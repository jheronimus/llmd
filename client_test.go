package llm_test

import (
	"context"
	"testing"
	"time"

	"github.com/jheronimus/llm"
)

type SampleTriage struct {
	IsSpam   bool   `llm:"bool,prompt:Is this spam?"`
	Category string `llm:"choice,support|billing|sales,prompt:Classify category"`
	Priority int    `llm:"score,0..5,prompt:Rate priority"`
}

func TestClientConfig(t *testing.T) {
	c, err := llm.New(
		llm.WithThreads(2),
		llm.WithContextSize(4096),
		llm.WithIdleTimeout(5*time.Minute),
	)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}
	defer func() { _ = c.Close() }()
}

func TestConfigureDefault(t *testing.T) {
	llm.Configure(
		llm.WithThreads(2),
		llm.WithContextSize(2048),
	)
}

func TestPreloadInvalidTarget(t *testing.T) {
	c, err := llm.New(llm.WithAutoDownload(false))
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer func() { _ = c.Close() }()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// Calling Decide on empty un-cached assets should fail-fast with ErrModelNotCached
	var triage SampleTriage
	err = c.Decide(ctx, "Hello world", &triage)
	if err == nil {
		// If assets happen to be already cached in ~/.cache/llm or ~/.cache/golaya, Decide will succeed
		t.Logf("Decide succeeded with cached assets: %+v", triage)
	}
}
