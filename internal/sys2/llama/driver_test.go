package llama_test

import (
	"context"
	"strings"
	"testing"

	"github.com/jheronimus/llm/internal/sys2/llama"
)

func TestInferenceSessionGenerate(t *testing.T) {
	mockDrv := llama.NewMockDriver("one two three four five")
	model, err := mockDrv.LoadModel("dummy.gguf", 4, 2048)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = model.Close() }()

	ctxObj, err := model.NewContext()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = ctxObj.Close() }()

	sess := llama.InferenceSession{
		Model: model,
		Ctx:   ctxObj,
	}

	var streamed []string
	out, promptLen, outTokens, err := sess.Generate(
		context.Background(),
		"hello world prompt",
		10,
		"",
		0.7,
		0.95,
		nil,
		func(p string) bool {
			streamed = append(streamed, p)
			return true
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if promptLen != 3 {
		t.Fatalf("expected 3 prompt tokens, got %d", promptLen)
	}
	if outTokens != 5 {
		t.Fatalf("expected 5 generated tokens, got %d", outTokens)
	}
	if len(streamed) != 5 {
		t.Fatalf("expected 5 streamed tokens, got %d", len(streamed))
	}
	if len(out) == 0 {
		t.Fatalf("expected non-empty output text")
	}
}

func TestInferenceSessionContextCancel(t *testing.T) {
	mockDrv := llama.NewMockDriver(strings.Repeat("word ", 100))
	model, err := mockDrv.LoadModel("dummy.gguf", 4, 2048)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = model.Close() }()

	ctxObj, err := model.NewContext()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = ctxObj.Close() }()

	sess := llama.InferenceSession{
		Model: model,
		Ctx:   ctxObj,
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, _, _, err = sess.Generate(ctx, "prompt", 100, "", 0.1, 0.95, nil, nil)
	if err != context.Canceled {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestInferenceSessionEarlyHaltStream(t *testing.T) {
	mockDrv := llama.NewMockDriver("one two three four five six seven")
	model, _ := mockDrv.LoadModel("dummy.gguf", 4, 2048)
	ctxObj, _ := model.NewContext()

	sess := llama.InferenceSession{Model: model, Ctx: ctxObj}

	count := 0
	_, _, outTokens, err := sess.Generate(
		context.Background(),
		"prompt",
		10,
		"",
		0.1,
		0.95,
		nil,
		func(p string) bool {
			count++
			return count < 3 // halt after 3 tokens
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if outTokens != 3 {
		t.Fatalf("expected early halt at 3 tokens, got %d", outTokens)
	}
}
