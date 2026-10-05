package llmd_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/jheronimus/llmd"
)

func TestClient_Decide(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.URL.Path == "/decide" {
			res := map[string]any{
				"answers": map[string]any{
					"is_job": map[string]any{
						"answer":     true,
						"confidence": 0.95,
						"act_cost":   0.0,
						"escalated":  false,
					},
				},
				"duration_ms": 15,
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(res)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	client := llmd.NewClient(llmd.WithBaseURL(server.URL), llmd.WithTimeout(5*time.Second))
	defer client.Close()

	if !client.IsAlive(context.Background()) {
		t.Fatal("expected client to report alive")
	}

	questions := map[string]llmd.DecideQuestion{
		"is_job": {
			Type:         "choice",
			Instructions: "Is this a job?",
		},
	}

	answers, err := client.Decide(context.Background(), "Looking for Senior Go dev", questions)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	ans, ok := answers["is_job"]
	if !ok {
		t.Fatal("missing is_job answer")
	}

	if ans.Confidence < 0.9 {
		t.Fatalf("expected confidence >= 0.9, got %f", ans.Confidence)
	}
	if ans.Escalated {
		t.Fatal("expected escalated to be false")
	}
}

func TestClient_Generate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/generate" {
			res := map[string]any{
				"text":        `{"role":"developer"}`,
				"duration_ms": 120,
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(res)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	client := llmd.NewClient(llmd.WithBaseURL(server.URL))
	defer client.Close()

	resp, err := client.Generate(context.Background(), llmd.Request{
		Prompt: "Extract title: Senior developer",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.Text != `{"role":"developer"}` {
		t.Fatalf("unexpected text output: %s", resp.Text)
	}
}
