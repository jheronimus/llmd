package sys1

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/jheronimus/llm/internal/sys1/calibration"
	"github.com/jheronimus/llm/internal/sys1/sequence"
	"github.com/jheronimus/llm/internal/sys1/tokenizer"
)

var (
	ortInitOnce sync.Once
	ortInitErr  error

	ErrClosed        = errors.New("llm: sys1 engine is closed")
	ErrInvalidSchema = errors.New("llm: invalid schema for decide")
)

type QuestionType int

const (
	TypeChoice QuestionType = iota
	TypeScore
	TypeNoul
)

func (t QuestionType) String() string {
	switch t {
	case TypeChoice:
		return "choice"
	case TypeScore:
		return "score"
	case TypeNoul:
		return "noul"
	default:
		return "unknown"
	}
}

type Question struct {
	Type         QuestionType
	Instructions string
	Criteria     any
	Labels       map[string]string
}

type Answer struct {
	Type          string             `json:"type"`
	Choice        string             `json:"choice,omitempty"`
	Score         float64            `json:"score,omitempty"`
	Noul          float64            `json:"noul,omitempty"`
	Confidence    float64            `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
}

type Engine struct {
	mu        sync.RWMutex
	cfg       ModelConfig
	tokenizer tokenizer.Tokenizer
	session   *ort.DynamicAdvancedSession
	closed    bool
}

type ModelConfig struct {
	ModelName            string             `json:"model_name"`
	MaxLen               int                `json:"max_len"`
	HeadMaxLen           int                `json:"head_max_len"`
	Temperature          []float64          `json:"temperature"`
	TemperatureByOptions map[string]float64 `json:"temperature_by_options"`
	ActCosts             map[string]float64 `json:"act_costs"`
}

// New creates and initializes the System 1 inference engine.
func New(modelDir, libPath string) (*Engine, error) {
	ortInitOnce.Do(func() {
		ort.SetSharedLibraryPath(libPath)
		if !ort.IsInitialized() {
			ortInitErr = ort.InitializeEnvironment()
		}
	})
	if ortInitErr != nil {
		return nil, fmt.Errorf("llm: failed initializing onnxruntime: %w", ortInitErr)
	}

	cfg := ModelConfig{
		ModelName:   "laya-multilingual",
		MaxLen:      1024,
		HeadMaxLen:  256,
		Temperature: []float64{1.0, 1.0, 1.0},
	}

	cfgPath := filepath.Join(modelDir, "config.json")
	if data, err := os.ReadFile(cfgPath); err == nil {
		_ = json.Unmarshal(data, &cfg)
	}

	tok, err := tokenizer.NewHFTokenizer(modelDir)
	if err != nil {
		return nil, fmt.Errorf("llm: tokenizer init failed: %w", err)
	}

	onnxPath := filepath.Join(modelDir, "model.onnx")
	session, err := ort.NewDynamicAdvancedSession(
		onnxPath,
		[]string{"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"},
		[]string{"logits", "act_logits"},
		nil,
	)
	if err != nil {
		_ = tok.Close()
		return nil, fmt.Errorf("llm: failed creating onnx session from %s: %w", onnxPath, err)
	}

	return &Engine{
		cfg:       cfg,
		tokenizer: tok,
		session:   session,
	}, nil
}

// Close releases the ONNX session and tokenizer.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true

	var errs []string
	if e.session != nil {
		if err := e.session.Destroy(); err != nil {
			errs = append(errs, err.Error())
		}
		e.session = nil
	}
	if e.tokenizer != nil {
		if err := e.tokenizer.Close(); err != nil {
			errs = append(errs, err.Error())
		}
		e.tokenizer = nil
	}

	if len(errs) > 0 {
		return errors.New(strings.Join(errs, "; "))
	}
	return nil
}

// Decide parses struct tags (`llm` or `laya`), executes zero-shot classification, and populates dest.
func (e *Engine) Decide(ctx context.Context, state any, dest any) error {
	v := reflect.ValueOf(dest)
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return fmt.Errorf("%w: dest must be a non-nil pointer to a struct", ErrInvalidSchema)
	}

	elem := v.Elem()
	if elem.Kind() != reflect.Struct {
		return fmt.Errorf("%w: dest must point to a struct, got %s", ErrInvalidSchema, elem.Kind())
	}

	elemType := elem.Type()
	questions := make(map[string]Question)
	fieldMap := make(map[string]int)

	for i := 0; i < elemType.NumField(); i++ {
		field := elemType.Field(i)
		tag := field.Tag.Get("llm")
		if tag == "" {
			tag = field.Tag.Get("laya")
		}
		if tag == "" || tag == "-" {
			continue
		}

		q, err := parseFieldTag(field.Name, tag)
		if err != nil {
			return fmt.Errorf("field %s: %w", field.Name, err)
		}

		qid := field.Name
		questions[qid] = q
		fieldMap[qid] = i
	}

	if len(questions) == 0 {
		return fmt.Errorf("%w: no fields tagged with `llm` or `laya` in %s", ErrInvalidSchema, elemType.Name())
	}

	answers, err := e.predict(ctx, state, questions)
	if err != nil {
		return err
	}

	for qid, idx := range fieldMap {
		ans, ok := answers[qid]
		if !ok {
			continue
		}

		fieldVal := elem.Field(idx)
		if !fieldVal.CanSet() {
			continue
		}

		if err := assignFieldValue(fieldVal, ans); err != nil {
			return fmt.Errorf("field %s: %w", qid, err)
		}
	}

	return nil
}

func (e *Engine) predict(ctx context.Context, state any, questions map[string]Question) (map[string]Answer, error) {
	e.mu.RLock()
	if e.closed || e.session == nil {
		e.mu.RUnlock()
		return nil, ErrClosed
	}
	session := e.session
	tok := e.tokenizer
	cfg := e.cfg
	e.mu.RUnlock()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	qids := slices.Sorted(maps.Keys(questions))
	if len(qids) == 0 {
		return make(map[string]Answer), nil
	}

	sequences := make([]*sequence.RawSequence, len(qids))
	maxSeqLen := 0
	maxMarkers := 0

	for i, qid := range qids {
		q := questions[qid]
		seq, err := sequence.BuildSequence(
			tok,
			qid,
			state,
			int(q.Type),
			q.Type.String(),
			q.Instructions,
			q.Criteria,
			q.Labels,
			cfg.MaxLen,
			cfg.HeadMaxLen,
			sequence.TruncateTail,
			false,
		)
		if err != nil {
			return nil, err
		}
		sequences[i] = seq
		if len(seq.InputIDs) > maxSeqLen {
			maxSeqLen = len(seq.InputIDs)
		}
		if len(seq.MarkerPositions) > maxMarkers {
			maxMarkers = len(seq.MarkerPositions)
		}
	}

	if maxMarkers == 0 {
		maxMarkers = 1
	}

	batch := len(sequences)
	paddedInputIDs := make([]int64, batch*maxSeqLen)
	paddedAttMask := make([]int64, batch*maxSeqLen)
	paddedMarkerPos := make([]int64, batch*maxMarkers)
	paddedMarkerMask := make([]bool, batch*maxMarkers)
	qtypes := make([]int64, batch)

	padID := tok.PADTokenID()
	for i := range paddedInputIDs {
		paddedInputIDs[i] = padID
	}

	for i, seq := range sequences {
		offsetSeq := i * maxSeqLen
		for j, id := range seq.InputIDs {
			paddedInputIDs[offsetSeq+j] = id
			paddedAttMask[offsetSeq+j] = 1
		}
		offsetMark := i * maxMarkers
		for j, pos := range seq.MarkerPositions {
			paddedMarkerPos[offsetMark+j] = int64(pos)
			paddedMarkerMask[offsetMark+j] = true
		}
		qtypes[i] = seq.QType
	}

	tInputIDs, err := ort.NewTensor(ort.NewShape(int64(batch), int64(maxSeqLen)), paddedInputIDs)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tInputIDs.Destroy() }()

	tAttMask, err := ort.NewTensor(ort.NewShape(int64(batch), int64(maxSeqLen)), paddedAttMask)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tAttMask.Destroy() }()

	tMarkerPos, err := ort.NewTensor(ort.NewShape(int64(batch), int64(maxMarkers)), paddedMarkerPos)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tMarkerPos.Destroy() }()

	tMarkerMask, err := ort.NewTensor(ort.NewShape(int64(batch), int64(maxMarkers)), paddedMarkerMask)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tMarkerMask.Destroy() }()

	tQTypes, err := ort.NewTensor(ort.NewShape(int64(batch)), qtypes)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tQTypes.Destroy() }()

	inputTensors := []ort.Value{tInputIDs, tAttMask, tMarkerPos, tMarkerMask, tQTypes}
	outputTensors := []ort.Value{nil, nil}

	err = session.Run(inputTensors, outputTensors)
	if err != nil {
		return nil, fmt.Errorf("llm: onnx forward pass failed: %w", err)
	}

	defer func() {
		for _, out := range outputTensors {
			if out != nil {
				_ = out.Destroy()
			}
		}
	}()

	logitsTensor, ok := outputTensors[0].(*ort.Tensor[float32])
	if !ok {
		return nil, errors.New("llm: unexpected logits tensor type from onnx")
	}

	logitsData := logitsTensor.GetData()
	answers := make(map[string]Answer, len(sequences))

	for r, qid := range qids {
		q := questions[qid]
		k := len(sequences[r].MarkerPositions)
		if k <= 0 {
			continue
		}

		startLogit := r * maxMarkers
		rowLogits := make([]float32, k)
		copy(rowLogits, logitsData[startLogit:startLogit+k])

		tScale := calibration.TemperatureScale(cfg.TemperatureByOptions, cfg.Temperature, q.Type.String(), int(q.Type), k)
		probs := calibration.Softmax(rowLogits, tScale)
		conf := calibration.ConfidenceFromProbs(probs, k)

		switch q.Type {
		case TypeChoice:
			keys := choiceKeys(q)
			probMap := make(map[string]float64, len(keys))
			bestIdx := 0
			bestProb := -1.0
			for i, key := range keys {
				p := 0.0
				if i < len(probs) {
					p = math.Round(probs[i]*10000) / 10000
				}
				probMap[key] = p
				if p > bestProb {
					bestProb = p
					bestIdx = i
				}
			}
			chosen := ""
			if bestIdx < len(keys) {
				chosen = keys[bestIdx]
			}
			answers[qid] = Answer{
				Type:          "choice",
				Choice:        chosen,
				Probabilities: probMap,
				Confidence:    conf,
			}

		case TypeScore:
			expScore := 0.0
			probMap := make(map[string]float64, len(probs))
			for i, p := range probs {
				roundedP := math.Round(p*10000) / 10000
				probMap[fmt.Sprintf("%d", i)] = roundedP
				expScore += float64(i) * roundedP
			}
			answers[qid] = Answer{
				Type:          "score",
				Score:         math.Round(expScore*10000) / 10000,
				Probabilities: probMap,
				Confidence:    conf,
			}

		case TypeNoul:
			noulProb := 0.0
			if len(probs) > 1 {
				noulProb = math.Round(probs[1]*10000) / 10000
			}
			noulConf := math.Round(math.Max(noulProb, 1.0-noulProb)*10000) / 10000
			answers[qid] = Answer{
				Type:       "noul",
				Noul:       noulProb,
				Confidence: noulConf,
			}
		}
	}

	return answers, nil
}

func choiceKeys(q Question) []string {
	if critMap, ok := q.Criteria.(map[string]string); ok {
		return slices.Sorted(maps.Keys(critMap))
	}
	if critSlice, ok := q.Criteria.([]string); ok {
		return slices.Clone(critSlice)
	}
	return nil
}

func parseFieldTag(fieldName, tag string) (Question, error) {
	parts := strings.Split(tag, ",")
	if len(parts) == 0 {
		return Question{}, errors.New("empty tag")
	}

	qtypeStr := strings.TrimSpace(parts[0])
	var prompt string
	var optionsPart string

	for _, p := range parts[1:] {
		p = strings.TrimSpace(p)
		if strings.HasPrefix(p, "prompt:") {
			prompt = strings.TrimPrefix(p, "prompt:")
		} else if optionsPart == "" {
			optionsPart = p
		}
	}

	if prompt == "" {
		prompt = fmt.Sprintf("Evaluate %s", fieldName)
	}

	switch qtypeStr {
	case "choice":
		if optionsPart == "" {
			return Question{}, fmt.Errorf("choice field %s requires options list (e.g. opt1|opt2)", fieldName)
		}
		opts := strings.Split(optionsPart, "|")
		m := make(map[string]string, len(opts))
		for _, o := range opts {
			trimmed := strings.TrimSpace(o)
			if trimmed != "" {
				m[trimmed] = trimmed
			}
		}
		return Question{Type: TypeChoice, Instructions: prompt, Criteria: m}, nil

	case "score":
		scale := []string{"0", "1", "2", "3", "4", "5"}
		if optionsPart != "" {
			if strings.Contains(optionsPart, "..") {
				rangeParts := strings.Split(optionsPart, "..")
				if len(rangeParts) == 2 {
					minV, _ := strconv.Atoi(rangeParts[0])
					maxV, _ := strconv.Atoi(rangeParts[1])
					if maxV > minV && (maxV-minV) <= 20 {
						scale = make([]string, maxV-minV+1)
						for v := minV; v <= maxV; v++ {
							scale[v-minV] = strconv.Itoa(v)
						}
					}
				}
			} else {
				scale = strings.Split(optionsPart, "|")
			}
		}
		return Question{Type: TypeScore, Instructions: prompt, Criteria: scale}, nil

	case "bool", "noul":
		return Question{Type: TypeNoul, Instructions: prompt}, nil

	default:
		return Question{}, fmt.Errorf("unknown question type: %s", qtypeStr)
	}
}

func assignFieldValue(val reflect.Value, ans Answer) error {
	switch val.Kind() {
	case reflect.String:
		if ans.Type == "choice" {
			val.SetString(ans.Choice)
		} else {
			val.SetString(fmt.Sprintf("%v", ans))
		}

	case reflect.Bool:
		if ans.Type == "noul" {
			val.SetBool(ans.Noul >= 0.5)
		} else {
			return fmt.Errorf("cannot assign non-noul answer to bool")
		}

	case reflect.Float32, reflect.Float64:
		switch ans.Type {
		case "score":
			val.SetFloat(ans.Score)
		case "noul":
			val.SetFloat(ans.Noul)
		default:
			val.SetFloat(ans.Confidence)
		}

	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		if ans.Type == "score" {
			val.SetInt(int64(ans.Score + 0.5))
		} else {
			return fmt.Errorf("cannot assign non-score answer to integer")
		}

	default:
		return fmt.Errorf("unsupported field type %s", val.Type())
	}
	return nil
}
