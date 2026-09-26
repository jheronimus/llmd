package questions_test

import (
	"testing"

	"github.com/jheronimus/llm/internal/questions"
)

type TestDecision struct {
	Category string  `llm:"choice,tech|finance|health,prompt:Classify category"`
	Score    int     `llm:"score,1..5,prompt:Rate relevance"`
	IsSpam   bool    `llm:"bool,prompt:Is this spam"`
	Conf     float64 `llm:"choice,yes|no,prompt:Check yes"`
}

func TestParseQuestions(t *testing.T) {
	var d TestDecision
	qs, fieldMap, err := questions.ParseQuestions(&d)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(qs) != 4 {
		t.Fatalf("expected 4 questions, got %d", len(qs))
	}

	catQ, ok := qs["Category"]
	if !ok || catQ.Type != questions.TypeChoice {
		t.Fatalf("expected Choice question for Category: %+v", catQ)
	}

	scoreQ, ok := qs["Score"]
	if !ok || scoreQ.Type != questions.TypeScore {
		t.Fatalf("expected Score question for Score: %+v", scoreQ)
	}

	spamQ, ok := qs["IsSpam"]
	if !ok || spamQ.Type != questions.TypeNoul {
		t.Fatalf("expected Noul question for IsSpam: %+v", spamQ)
	}

	if len(fieldMap) != 4 {
		t.Fatalf("expected 4 field map entries, got %d", len(fieldMap))
	}
}

func TestPopulateDest(t *testing.T) {
	var d TestDecision
	_, fieldMap, err := questions.ParseQuestions(&d)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	answers := map[string]questions.Answer{
		"Category": {Answer: "tech", Confidence: 0.98},
		"Score":    {Answer: float64(4), Confidence: 0.85},
		"IsSpam":   {Answer: false, Confidence: 0.95},
		"Conf":     {Answer: "yes", Confidence: 0.99},
	}

	if err := questions.PopulateDest(&d, answers, fieldMap); err != nil {
		t.Fatalf("failed to populate dest: %v", err)
	}

	if d.Category != "tech" {
		t.Errorf("expected tech, got %s", d.Category)
	}
	if d.Score != 4 {
		t.Errorf("expected 4, got %d", d.Score)
	}
	if d.IsSpam != false {
		t.Errorf("expected false, got %v", d.IsSpam)
	}
	if d.Conf != 0.99 {
		t.Errorf("expected 0.99, got %f", d.Conf)
	}
}
