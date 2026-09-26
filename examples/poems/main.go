package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/jheronimus/llm"
)

// System 1 struct for zero-shot poetic classification (<10 ms).
type PoemAnalysis struct {
	IsHaiku bool   `llm:"bool,prompt:Is this text a haiku (traditional Japanese 3-line poetry)?"`
	Genre   string `llm:"choice,haiku|limerick|sonnet|other,prompt:What is the poetic genre of this text?"`
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	genres := []string{
		"a Japanese haiku about autumn leaves",
		"a humorous five-line limerick about coding in Go",
		"a four-line Shakespearean sonnet stanza about the stars",
	}

	for i, genre := range genres {
		fmt.Printf("\n=== Generating poem %d: [%s] ===\n", i+1, genre)

		// 1. System 2: Autoregressive text generation
		resp, err := llm.Generate(ctx, llm.Request{
			System:      "You are a poetic master. Output only the poem text, without introductory words or notes.",
			Prompt:      fmt.Sprintf("Write %s.", genre),
			Temperature: 0.2,
			MaxTokens:   100,
		})
		if err != nil {
			log.Fatalf("generate failed: %v", err)
		}

		fmt.Printf("%s\n", resp.Text)

		// 2. System 1: Instant zero-shot classification (<10 ms)
		analysis, err := llm.Decide[PoemAnalysis](ctx, resp.Text)
		if err != nil {
			log.Fatalf("decide failed: %v", err)
		}

		fmt.Printf("--> [Sys 1 Decision] Is Haiku: %-5v | Detected Genre: %s\n",
			analysis.IsHaiku, analysis.Genre)
	}
}
