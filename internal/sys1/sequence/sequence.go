package sequence

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
)

var (
	ErrInvalidQuestion = errors.New("golaya: invalid question definition")
)

// ContextExceededError is returned when input exceeds model context and strict mode is enabled.
type ContextExceededError struct {
	MaxTokens   int
	HeadTokens  int
	StateTokens int
	TotalTokens int
}

func (e *ContextExceededError) Error() string {
	return fmt.Sprintf("golaya: input exceeds context budget: state=%d + head=%d total=%d > max=%d",
		e.StateTokens, e.HeadTokens, e.TotalTokens, e.MaxTokens)
}

// OptionsBudgetExceededError is returned when question options alone exceed head_max_len.
type OptionsBudgetExceededError struct {
	QuestionID string
	OptionsLen int
	HeadMaxLen int
}

func (e *OptionsBudgetExceededError) Error() string {
	return fmt.Sprintf("golaya: question %q options (%d tokens) exceed head_max_len budget (%d)",
		e.QuestionID, e.OptionsLen, e.HeadMaxLen)
}

// Tokenizer abstracts the tokenizer methods needed for sequence construction.
type Tokenizer interface {
	Encode(text string, addSpecialTokens bool) ([]int64, error)
	CLSTokenID() int64
	SEPTokenID() int64
	MASKTokenID() int64
	PADTokenID() int64
}

// Usage tracks sequence tokens and truncation.
type Usage struct {
	InputTokens   int
	StateTokens   int
	WasTruncated  bool
	DroppedTokens int
}

// RawSequence represents tokenized inputs ready for model tensor construction.
type RawSequence struct {
	InputIDs        []int64
	AttentionMask   []int64
	MarkerPositions []int64
	MarkerMask      []bool
	QType           int64
	Usage           Usage
}

// TruncateDirection defines where to clip input state when context exceeds budget.
type TruncateDirection int

const (
	TruncateTail TruncateDirection = 0
	TruncateHead TruncateDirection = 1
)

// SerializeState converts arbitrary state input into a string representation for tokenization.
func SerializeState(state any) string {
	if state == nil {
		return ""
	}
	switch v := state.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	default:
		data, err := json.Marshal(v)
		if err != nil {
			return fmt.Sprintf("%v", v)
		}
		return string(data)
	}
}

// RenderOptions formats question options matching upstream Laya.
func RenderOptions(qtype int, criteria any, labels map[string]string) ([]string, error) {
	switch qtype {
	case 0: // Choice
		if critMap, ok := criteria.(map[string]string); ok {
			keys := slices.Sorted(maps.Keys(critMap))
			opts := make([]string, len(keys))
			for i, k := range keys {
				v := critMap[k]
				if v == "" {
					opts[i] = k
				} else {
					opts[i] = fmt.Sprintf("%s: %s", k, v)
				}
			}
			return opts, nil
		}
		if critSlice, okSlice := criteria.([]string); okSlice {
			return slices.Clone(critSlice), nil
		}
		return nil, fmt.Errorf("%w: choice criteria must be map[string]string or []string", ErrInvalidQuestion)

	case 1: // Score
		scale, ok := criteria.([]string)
		if !ok {
			return nil, fmt.Errorf("%w: score criteria must be []string", ErrInvalidQuestion)
		}
		opts := make([]string, len(scale))
		for i, c := range scale {
			opts[i] = fmt.Sprintf("level %d: %s", i, c)
		}
		return opts, nil

	case 2: // Noul
		falseLabel := "false"
		trueLabel := "true"
		if labels != nil {
			if f, ok := labels["false"]; ok && f != "" {
				falseLabel = strings.TrimSpace(f)
			}
			if t, ok := labels["true"]; ok && t != "" {
				trueLabel = strings.TrimSpace(t)
			}
		}

		var falseCrit, trueCrit string
		if critMap, ok := criteria.(map[string]string); ok {
			falseCrit = critMap["false"]
			trueCrit = critMap["true"]
		}

		falseDesc := "no, the statement does not hold"
		if falseCrit != "" {
			falseDesc = falseCrit
		}
		trueDesc := "yes, the statement holds"
		if trueCrit != "" {
			trueDesc = trueCrit
		}

		return []string{
			fmt.Sprintf("%s: %s", falseLabel, falseDesc),
			fmt.Sprintf("%s: %s", trueLabel, trueDesc),
		}, nil

	default:
		return nil, fmt.Errorf("%w: unsupported question type %v", ErrInvalidQuestion, qtype)
	}
}

// BuildSequence implements upstream Laya state budgeting and marker positioning.
func BuildSequence(
	tok Tokenizer,
	qid string,
	state any,
	qtype int,
	qtypeName string,
	instructions string,
	criteria any,
	labels map[string]string,
	maxLen int,
	headMaxLen int,
	truncateDir TruncateDirection,
	strictMode bool,
) (*RawSequence, error) {
	if maxLen <= 0 {
		maxLen = 512
	}
	if headMaxLen <= 0 {
		headMaxLen = 192
	}

	opts, err := RenderOptions(qtype, criteria, labels)
	if err != nil {
		return nil, err
	}
	if len(opts) == 0 {
		return nil, fmt.Errorf("%w: question %q has no options", ErrInvalidQuestion, qid)
	}

	// 1. Encode question instructions: "<type> question: <instructions>"
	headPrompt := fmt.Sprintf("%s question: %s", qtypeName, instructions)
	headIDs, err := tok.Encode(headPrompt, false)
	if err != nil {
		return nil, fmt.Errorf("golaya: failed to tokenize question head: %w", err)
	}

	// 2. Encode options (each with leading [MASK] token, capped at 48 tokens per opt)
	optIDs := make([][]int64, len(opts))
	for i, opt := range opts {
		oTokens, err := tok.Encode(" "+opt, false)
		if err != nil {
			return nil, fmt.Errorf("golaya: failed to tokenize option %q: %w", opt, err)
		}
		if len(oTokens) > 48 {
			oTokens = oTokens[:48]
		}
		optIDs[i] = append([]int64{tok.MASKTokenID()}, oTokens...)
	}

	totalOptTokens := 0
	for _, o := range optIDs {
		totalOptTokens += len(o)
	}
	if totalOptTokens >= headMaxLen {
		return nil, &OptionsBudgetExceededError{
			QuestionID: qid,
			OptionsLen: totalOptTokens,
			HeadMaxLen: headMaxLen,
		}
	}

	optBudget := headMaxLen - totalOptTokens
	if optBudget < 16 {
		per := max(4, (headMaxLen-16)/max(1, len(optIDs)))
		for i := range optIDs {
			if len(optIDs[i]) > per {
				optIDs[i] = optIDs[i][:per]
			}
		}
		totalOptTokens = 0
		for _, o := range optIDs {
			totalOptTokens += len(o)
		}
		optBudget = headMaxLen - totalOptTokens
	}

	maxHeadBudget := max(8, optBudget)
	if len(headIDs) > maxHeadBudget {
		headIDs = headIDs[:maxHeadBudget]
	}

	// 3. Assemble sequence: [CLS] + headIDs + [SEP] + optIDs + [SEP]
	ids := make([]int64, 0, maxLen)
	ids = append(ids, tok.CLSTokenID())
	ids = append(ids, headIDs...)
	ids = append(ids, tok.SEPTokenID())

	markers := make([]int64, 0, len(optIDs))
	for _, o := range optIDs {
		markers = append(markers, int64(len(ids)))
		ids = append(ids, o...)
	}
	ids = append(ids, tok.SEPTokenID())

	// 4. Tokenize user state and perform state budgeting
	stateStr := SerializeState(state)
	stateIDs, err := tok.Encode(stateStr, false)
	if err != nil {
		return nil, fmt.Errorf("golaya: failed to tokenize state: %w", err)
	}

	room := maxLen - len(ids) - 1
	if room < 0 {
		room = 0
	}

	usage := Usage{
		StateTokens: len(stateIDs),
	}

	var finalStateIDs []int64
	if len(stateIDs) > room {
		if strictMode {
			return nil, &ContextExceededError{
				MaxTokens:   maxLen,
				HeadTokens:  len(ids) + 1,
				StateTokens: len(stateIDs),
				TotalTokens: len(ids) + len(stateIDs) + 1,
			}
		}
		usage.WasTruncated = true
		usage.DroppedTokens = len(stateIDs) - room

		if truncateDir == TruncateHead {
			finalStateIDs = stateIDs[len(stateIDs)-room:]
		} else {
			finalStateIDs = stateIDs[:room]
		}
	} else {
		finalStateIDs = stateIDs
	}

	ids = append(ids, finalStateIDs...)
	ids = append(ids, tok.SEPTokenID())

	if len(ids) > maxLen {
		ids = ids[:maxLen]
	}
	usage.InputTokens = len(ids)

	attMask := make([]int64, len(ids))
	for i := range attMask {
		attMask[i] = 1
	}

	markerMask := make([]bool, len(markers))
	validMarkers := make([]int64, 0, len(markers))
	for i, m := range markers {
		if m < int64(maxLen) {
			validMarkers = append(validMarkers, m)
			markerMask[i] = true
		} else {
			validMarkers = append(validMarkers, -1)
			markerMask[i] = false
		}
	}

	return &RawSequence{
		InputIDs:        ids,
		AttentionMask:   attMask,
		MarkerPositions: validMarkers,
		MarkerMask:      markerMask,
		QType:           int64(qtype),
		Usage:           usage,
	}, nil
}
