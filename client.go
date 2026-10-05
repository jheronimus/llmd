package llmd

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Client interacts with the llmd sidecar daemon over HTTP (Zero CGO).
type Client struct {
	mu      sync.Mutex
	cfg     Config
	httpCli *http.Client
	closed  bool
}

// NewClient initializes a new llmd client.
func NewClient(opts ...Option) *Client {
	cfg := Config{
		BaseURL: "http://127.0.0.1:8080",
		Timeout: 2 * time.Minute,
	}

	for _, opt := range opts {
		opt(&cfg)
	}

	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")

	return &Client{
		cfg: cfg,
		httpCli: &http.Client{
			Timeout: cfg.Timeout,
		},
	}
}

// IsAlive checks daemon liveness via /health.
func (c *Client) IsAlive(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.cfg.BaseURL+"/health", nil)
	if err != nil {
		return false
	}
	resp, err := c.httpCli.Do(req)
	if err != nil {
		return false
	}
	_ = resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

type decideReqEnvelope struct {
	State        any                       `json:"state"`
	Questions    map[string]DecideQuestion `json:"questions"`
	AutoEscalate bool                      `json:"auto_escalate"`
}

type decideRespEnvelope struct {
	Answers    map[string]DecideAnswer `json:"answers"`
	DurationMs uint64                  `json:"duration_ms"`
}

// Decide executes System 1 zero-shot classification with auto-escalation.
func (c *Client) Decide(ctx context.Context, state any, questions map[string]DecideQuestion) (map[string]DecideAnswer, error) {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil, ErrClientClosed
	}
	c.mu.Unlock()

	bodyBytes, err := json.Marshal(decideReqEnvelope{
		State:        state,
		Questions:    questions,
		AutoEscalate: true,
	})
	if err != nil {
		return nil, fmt.Errorf("llmd: marshal decide request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.BaseURL+"/decide", bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpCli.Do(req)
	if err != nil {
		return nil, fmt.Errorf("llmd: decide request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("llmd: daemon error %d: %s", resp.StatusCode, string(b))
	}

	var dResp decideRespEnvelope
	if err := json.NewDecoder(resp.Body).Decode(&dResp); err != nil {
		return nil, fmt.Errorf("llmd: decode decide response: %w", err)
	}

	return dResp.Answers, nil
}

type generateReqEnvelope struct {
	System      *string  `json:"system,omitempty"`
	Prompt      string   `json:"prompt"`
	Grammar     *string  `json:"grammar,omitempty"`
	Temperature *float32 `json:"temperature,omitempty"`
	TopP        *float32 `json:"top_p,omitempty"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
}

type generateRespEnvelope struct {
	Text       string `json:"text"`
	DurationMs uint64 `json:"duration_ms"`
}

// Generate runs System 2 generation.
func (c *Client) Generate(ctx context.Context, req Request) (Response, error) {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return Response{}, ErrClientClosed
	}
	c.mu.Unlock()

	gReq := generateReqEnvelope{
		Prompt: req.Prompt,
	}
	if req.System != "" {
		gReq.System = &req.System
	}
	if req.Grammar != "" {
		gReq.Grammar = &req.Grammar
	}
	if req.Temperature > 0 {
		gReq.Temperature = &req.Temperature
	}
	if req.TopP > 0 {
		gReq.TopP = &req.TopP
	}
	if req.MaxTokens > 0 {
		gReq.MaxTokens = &req.MaxTokens
	}

	bodyBytes, err := json.Marshal(gReq)
	if err != nil {
		return Response{}, fmt.Errorf("llmd: marshal generate request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.BaseURL+"/generate", bytes.NewReader(bodyBytes))
	if err != nil {
		return Response{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpCli.Do(httpReq)
	if err != nil {
		return Response{}, fmt.Errorf("llmd: generate request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return Response{}, fmt.Errorf("llmd: daemon error %d: %s", resp.StatusCode, string(b))
	}

	var gResp generateRespEnvelope
	if err := json.NewDecoder(resp.Body).Decode(&gResp); err != nil {
		return Response{}, fmt.Errorf("llmd: decode generate response: %w", err)
	}

	return Response{
		Text:     gResp.Text,
		Duration: time.Duration(gResp.DurationMs) * time.Millisecond,
	}, nil
}

// Close marks the client closed.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	c.httpCli.CloseIdleConnections()
	return nil
}
