package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/jheronimus/llm"
)

// Hacker News Algolia response format.
type hnResponse struct {
	Hits []struct {
		Title string `json:"title"`
		URL   string `json:"url"`
	} `json:"hits"`
}

// System 1 struct for fast zero-shot filtering (<10 ms per story).
type StoryTriage struct {
	IsProductOrTool bool `llm:"bool,prompt:Is this link or post about a new software product, library, developer tool, or project release?"`
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	fmt.Println("1. Fetching last 100 stories from Hacker News...")
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://hn.algolia.com/api/v1/search_by_date?tags=story&hitsPerPage=100", nil)
	if err != nil {
		log.Fatalf("failed to build request: %v", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		log.Fatalf("failed to fetch HN stories: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	var data hnResponse
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		log.Fatalf("failed to decode HN response: %v", err)
	}

	fmt.Printf("2. Filtering %d stories with System 1 (zero-shot NLU)...\n", len(data.Hits))
	s1Start := time.Now()
	var candidates []string

	for _, hit := range data.Hits {
		text := fmt.Sprintf("Title: %s\nURL: %s", hit.Title, hit.URL)
		triage, err := llm.Decide[StoryTriage](ctx, text)
		if err != nil {
			continue
		}
		if triage.IsProductOrTool {
			candidates = append(candidates, fmt.Sprintf("- %s (%s)", hit.Title, hit.URL))
		}
	}
	s1Duration := time.Since(s1Start)
	fmt.Printf("--> Kept %d candidate launches in %v (~%v/story)\n\n",
		len(candidates), s1Duration, s1Duration/time.Duration(len(data.Hits)))

	if len(candidates) == 0 {
		fmt.Println("No candidate product launches found.")
		return
	}

	fmt.Println("3. Synthesizing developer product digest with System 2...")
	s2Start := time.Now()

	digest, err := llm.Generate(ctx, llm.Request{
		System: "You are a tech curator. From the provided list of candidate Hacker News posts, select ONLY the genuine software products, developer tools, or open-source releases. Output a clean markdown digest with 3 to 5 bullet points explaining what each tool is and why developers should care.",
		Prompt: fmt.Sprintf("Here are the candidate posts:\n%s\n\nWrite the product digest:", strings.Join(candidates, "\n")),
		MaxTokens: 350,
		Temperature: 0.2,
	})
	if err != nil {
		log.Fatalf("generation failed: %v", err)
	}
	s2Duration := time.Since(s2Start)

	fmt.Printf("--> Digest generated in %v:\n\n%s\n", s2Duration, digest.Text)
}
