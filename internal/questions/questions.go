package questions

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
)

var ErrInvalidSchema = errors.New("llm: invalid schema")

type QuestionType string

const (
	TypeChoice QuestionType = "choice"
	TypeScore  QuestionType = "score"
	TypeNoul   QuestionType = "noul"
)

type Question struct {
	Type         QuestionType      `json:"type"`
	Instructions string            `json:"instructions"`
	Criteria     any               `json:"criteria,omitempty"`
	Labels       map[string]string `json:"labels,omitempty"`
}

type Answer struct {
	Answer     any     `json:"answer"`
	Confidence float64 `json:"confidence"`
	ActCost    float64 `json:"act_cost"`
}

// ParseQuestions inspects struct tags on dest and returns Question definitions and field indices.
func ParseQuestions(dest any) (map[string]Question, map[string]int, error) {
	v := reflect.ValueOf(dest)
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return nil, nil, fmt.Errorf("%w: dest must be a non-nil pointer to a struct", ErrInvalidSchema)
	}

	elem := v.Elem()
	if elem.Kind() != reflect.Struct {
		return nil, nil, fmt.Errorf("%w: dest must point to a struct, got %s", ErrInvalidSchema, elem.Kind())
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
			return nil, nil, fmt.Errorf("field %s: %w", field.Name, err)
		}

		qid := field.Name
		questions[qid] = q
		fieldMap[qid] = i
	}

	if len(questions) == 0 {
		return nil, nil, fmt.Errorf("%w: no fields tagged with `llm` or `laya` in %s", ErrInvalidSchema, elemType.Name())
	}

	return questions, fieldMap, nil
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

// PopulateDest fills dest fields from the returned answers map.
func PopulateDest(dest any, answers map[string]Answer, fieldMap map[string]int) error {
	v := reflect.ValueOf(dest).Elem()

	for qid, idx := range fieldMap {
		ans, ok := answers[qid]
		if !ok {
			continue
		}

		fieldVal := v.Field(idx)
		if !fieldVal.CanSet() {
			continue
		}

		if err := assignFieldValue(fieldVal, ans); err != nil {
			return fmt.Errorf("field %s: %w", qid, err)
		}
	}

	return nil
}

func assignFieldValue(val reflect.Value, ans Answer) error {
	switch val.Kind() {
	case reflect.String:
		if s, ok := ans.Answer.(string); ok {
			val.SetString(s)
		} else {
			val.SetString(fmt.Sprintf("%v", ans.Answer))
		}

	case reflect.Bool:
		switch v := ans.Answer.(type) {
		case bool:
			val.SetBool(v)
		case float64:
			val.SetBool(v >= 0.5)
		default:
			val.SetBool(ans.Confidence >= 0.5)
		}

	case reflect.Float32, reflect.Float64:
		switch v := ans.Answer.(type) {
		case float64:
			val.SetFloat(v)
		case json.Number:
			f, _ := v.Float64()
			val.SetFloat(f)
		default:
			val.SetFloat(ans.Confidence)
		}

	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		switch v := ans.Answer.(type) {
		case float64:
			val.SetInt(int64(math.Round(v)))
		case json.Number:
			i, _ := v.Int64()
			val.SetInt(i)
		case string:
			i, _ := strconv.ParseInt(v, 10, 64)
			val.SetInt(i)
		default:
			return fmt.Errorf("cannot assign non-number answer to integer: %v", ans.Answer)
		}

	default:
		return fmt.Errorf("unsupported field type %s", val.Type())
	}
	return nil
}
