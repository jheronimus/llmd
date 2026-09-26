package llm_test

import (
	"context"
	"testing"
	"time"

	"github.com/jheronimus/llm"
)

type VacancyDecision struct {
	IsVacancy bool   `llm:"bool,prompt:Является ли данный текст объявлением о вакансии?"`
	WorkMode  string `llm:"choice,remote|office|hybrid|unclear,prompt:Какой формат работы указан?"`
}

type RankedJobItem struct {
	Title     string `json:"title"`
	Score     int    `json:"score"`
	FitReason string `json:"fit_reason"`
}

func TestLiveEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping live test in short mode")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// 1. Test System 1: Decide
	adText := `Ищем Senior Product Manager в финтех команду. Удаленная работа, гибкий график, зарплата от 350 000 руб.`
	dec, err := llm.Decide[VacancyDecision](ctx, adText)
	if err != nil {
		t.Fatalf("llm.Decide failed: %v", err)
	}

	t.Logf("System 1 Decision: IsVacancy=%v, WorkMode=%s", dec.IsVacancy, dec.WorkMode)
	if !dec.IsVacancy {
		t.Errorf("expected IsVacancy to be true, got false")
	}
	if dec.WorkMode != "remote" {
		t.Errorf("expected WorkMode to be remote, got %s", dec.WorkMode)
	}

	// 2. Test System 2: Extract (GBNF-constrained JSON)
	prompt := `Оцени вакансию "Senior Product Manager в финтех команду, удаленка" по критериям продукта (0-100) и верни JSON объект: title, score, fit_reason.`
	extracted, err := llm.Extract[RankedJobItem](ctx, prompt)
	if err != nil {
		t.Fatalf("llm.Extract failed: %v", err)
	}

	t.Logf("System 2 Extracted: Title=%s, Score=%d, Reason=%s", extracted.Title, extracted.Score, extracted.FitReason)
	if extracted.Score < 60 {
		t.Errorf("expected Score >= 60 for Senior PM, got %d", extracted.Score)
	}

	// 3. Test System 2: Generate
	resp, err := llm.Generate(ctx, llm.Request{
		Prompt:    "Ответь одним словом: столица Франции?",
		MaxTokens: 20,
	})
	if err != nil {
		t.Fatalf("llm.Generate failed: %v", err)
	}
	t.Logf("System 2 Generated: %s (%d tokens, %v)", resp.Text, resp.OutputTokens, resp.Duration)
}
