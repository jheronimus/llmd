package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/jheronimus/llm/internal/asset"
	"github.com/jheronimus/llm/internal/grammar"
	"github.com/jheronimus/llm/internal/questions"
)

// Client coordinates inference interactions with the llmd sidecar daemon.
type Client struct {
	mu         sync.Mutex
	cfg        Config
	cacheDir   string
	socketPath string
	httpCli    *http.Client
	closed     bool
}

// New creates and initializes an LLM client.
func New(opts ...Option) (*Client, error) {
	cfg := Config{
		Threads:      runtime.NumCPU(),
		ContextSize:  8192,
		IdleTimeout:  10 * time.Minute,
		AutoDownload: true,
		AutoSpawn:    true,
		SocketPath:   "/tmp/llm.sock",
	}

	if envSock := os.Getenv("LLM_SOCKET_PATH"); envSock != "" {
		cfg.SocketPath = envSock
	}

	for _, opt := range opts {
		opt(&cfg)
	}

	cacheDir, err := asset.EnsureCacheDir(cfg.CacheDir)
	if err != nil {
		return nil, fmt.Errorf("llm: cache dir setup failed: %w", err)
	}

	httpCli := &http.Client{
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
				return (&net.Dialer{}).DialContext(ctx, "unix", cfg.SocketPath)
			},
		},
		Timeout: 5 * time.Minute,
	}

	return &Client{
		cfg:        cfg,
		cacheDir:   cacheDir,
		socketPath: cfg.SocketPath,
		httpCli:    httpCli,
	}, nil
}

// Preload ensures models and daemon binaries are present in cache.
func (c *Client) Preload(ctx context.Context, targets ...SystemTarget) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrClientClosed
	}

	// Locate or download daemon binary
	if _, err := asset.LocateOrDownloadDaemon(c.cfg.DaemonBinary, c.cacheDir, c.cfg.AutoDownload); err != nil {
		return fmt.Errorf("preload daemon: %w", err)
	}

	target := AllSystems
	if len(targets) > 0 {
		target = targets[0]
	}

	if target == AllSystems || target == System1 {
		if _, err := asset.LocateOrDownloadSys1Model(c.cfg.Sys1ModelDir, "", c.cacheDir, c.cfg.AutoDownload); err != nil {
			return fmt.Errorf("preload sys1 model: %w", err)
		}
	}

	if target == AllSystems || target == System2 {
		if _, err := asset.LocateOrDownloadSys2Model(c.cfg.Sys2ModelPath, c.cacheDir, c.cfg.AutoDownload); err != nil {
			return fmt.Errorf("preload sys2 model: %w", err)
		}
	}

	return nil
}

func (c *Client) isAlive(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://localhost/health", nil)
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

func (c *Client) ensureDaemon(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrClientClosed
	}

	// 1. Check if already responding
	healthCtx, cancel := context.WithTimeout(ctx, 300*time.Millisecond)
	alive := c.isAlive(healthCtx)
	cancel()
	if alive {
		return nil
	}

	if !c.cfg.AutoSpawn {
		return errors.New("llm: llmd daemon is not running and AutoSpawn is disabled")
	}

	// 2. Remove stale socket
	_ = os.Remove(c.socketPath)

	// 3. Locate or download llmd binary
	binPath, err := asset.LocateOrDownloadDaemon(c.cfg.DaemonBinary, c.cacheDir, c.cfg.AutoDownload)
	if err != nil {
		return err
	}

	// 4. Build command arguments
	args := []string{"--socket", c.socketPath}
	if c.cfg.IdleTimeout > 0 {
		args = append(args, "--idle-timeout", fmt.Sprintf("%d", int(c.cfg.IdleTimeout.Seconds())))
	}
	if c.cfg.Sys1ModelDir != "" {
		args = append(args, "--sys1-dir", c.cfg.Sys1ModelDir)
	}
	if c.cfg.Sys2ModelPath != "" {
		args = append(args, "--sys2-model", c.cfg.Sys2ModelPath)
	}

	cmd := exec.Command(binPath, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	logFile, err := os.OpenFile(filepath.Join(c.cacheDir, "llmd.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err == nil {
		cmd.Stdout = logFile
		cmd.Stderr = logFile
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("llm: failed to start llmd (%s): %w", binPath, err)
	}

	go func() {
		_ = cmd.Wait()
		if logFile != nil {
			_ = logFile.Close()
		}
	}()

	// 5. Poll /health until ready
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		pollCtx, pollCancel := context.WithTimeout(ctx, 200*time.Millisecond)
		if c.isAlive(pollCtx) {
			pollCancel()
			return nil
		}
		pollCancel()
		time.Sleep(50 * time.Millisecond)
	}

	return fmt.Errorf("llm: llmd started at %s but failed to respond to /health within 10s", c.socketPath)
}

type decideRequest struct {
	State     any                          `json:"state"`
	Questions map[string]questions.Question `json:"questions"`
}

type decideResponse struct {
	Answers    map[string]questions.Answer `json:"answers"`
	DurationMs uint64                       `json:"duration_ms"`
}

// Decide executes System 1 zero-shot classification and populates fields in dest.
func (c *Client) Decide(ctx context.Context, state any, dest any) error {
	qs, fieldMap, err := questions.ParseQuestions(dest)
	if err != nil {
		return err
	}

	if err := c.ensureDaemon(ctx); err != nil {
		return err
	}

	bodyBytes, err := json.Marshal(decideRequest{
		State:     state,
		Questions: qs,
	})
	if err != nil {
		return fmt.Errorf("llm: failed to marshal decide request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://localhost/decide", bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := c.httpCli.Do(httpReq)
	if err != nil {
		return fmt.Errorf("llm: decide request failed: %w", err)
	}
	defer func() { _ = httpResp.Body.Close() }()

	if httpResp.StatusCode != http.StatusOK {
		respBytes, _ := io.ReadAll(httpResp.Body)
		return fmt.Errorf("llm: daemon returned error %d: %s", httpResp.StatusCode, string(respBytes))
	}

	var dResp decideResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&dResp); err != nil {
		return fmt.Errorf("llm: failed to decode decide response: %w", err)
	}

	return questions.PopulateDest(dest, dResp.Answers, fieldMap)
}

type generateRequest struct {
	System      *string  `json:"system,omitempty"`
	Prompt      string   `json:"prompt"`
	Grammar     *string  `json:"grammar,omitempty"`
	Temperature *float32 `json:"temperature,omitempty"`
	TopP        *float32 `json:"top_p,omitempty"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
}

type generateResponse struct {
	Text       string `json:"text"`
	DurationMs uint64 `json:"duration_ms"`
}

// Generate executes text generation on System 2.
func (c *Client) Generate(ctx context.Context, req Request) (Response, error) {
	if err := c.ensureDaemon(ctx); err != nil {
		return Response{}, err
	}

	grammarStr := req.Grammar
	if grammarStr == "" && req.JSONSchema != "" {
		g, err := grammar.FromJSONSchema(req.JSONSchema)
		if err == nil {
			grammarStr = g
		}
	}

	gReq := generateRequest{
		Prompt: req.Prompt,
	}
	if req.System != "" {
		gReq.System = &req.System
	}
	if grammarStr != "" {
		gReq.Grammar = &grammarStr
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
		return Response{}, fmt.Errorf("llm: marshal generate request failed: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://localhost/generate", bytes.NewReader(bodyBytes))
	if err != nil {
		return Response{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := c.httpCli.Do(httpReq)
	if err != nil {
		return Response{}, fmt.Errorf("llm: generate request failed: %w", err)
	}
	defer func() { _ = httpResp.Body.Close() }()

	if httpResp.StatusCode != http.StatusOK {
		respBytes, _ := io.ReadAll(httpResp.Body)
		return Response{}, fmt.Errorf("llm: daemon returned error %d: %s", httpResp.StatusCode, string(respBytes))
	}

	var gResp generateResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&gResp); err != nil {
		return Response{}, fmt.Errorf("llm: decode generate response failed: %w", err)
	}

	return Response{
		Text:     cleanResponseText(gResp.Text),
		Duration: time.Duration(gResp.DurationMs) * time.Millisecond,
	}, nil
}

// GenerateStream executes text generation with streaming callback.
func (c *Client) GenerateStream(ctx context.Context, req Request, cb StreamCallback) (Response, error) {
	resp, err := c.Generate(ctx, req)
	if err != nil {
		return Response{}, err
	}
	cb(resp.Text)
	return resp, nil
}

// Extract generates structured output conforming to dest struct schema.
func (c *Client) Extract(ctx context.Context, prompt string, dest any) error {
	return c.ExtractWithSystem(ctx, "", prompt, dest)
}

// ExtractWithSystem generates structured output using an explicit system prompt.
func (c *Client) ExtractWithSystem(ctx context.Context, system, prompt string, dest any) error {
	if dest == nil {
		return errors.New("llm: extract destination cannot be nil")
	}

	gStr, err := grammar.FromType(dest)
	if err != nil || gStr == "" {
		gStr = grammar.GenericJSON()
	}

	req := Request{
		System:    system,
		Prompt:    prompt,
		Grammar:   gStr,
		MaxTokens: 2048,
	}

	resp, err := c.Generate(ctx, req)
	if err != nil {
		return err
	}

	clean := cleanResponseText(resp.Text)
	if err := json.Unmarshal([]byte(clean), dest); err != nil {
		return fmt.Errorf("llm: failed to unmarshal JSON output: %w (output: %s)", err, clean)
	}

	return nil
}

func cleanResponseText(raw string) string {
	raw = strings.TrimSpace(raw)
	if idx := strings.Index(raw, "</think>"); idx != -1 {
		raw = strings.TrimSpace(raw[idx+len("</think>"):])
	}
	if strings.HasPrefix(raw, "```json") {
		raw = strings.TrimPrefix(raw, "```json")
		if idx := strings.LastIndex(raw, "```"); idx != -1 {
			raw = raw[:idx]
		}
	} else if strings.HasPrefix(raw, "```") {
		raw = strings.TrimPrefix(raw, "```")
		if idx := strings.LastIndex(raw, "```"); idx != -1 {
			raw = raw[:idx]
		}
	}
	return strings.TrimSpace(raw)
}

// Close gracefully closes HTTP connections and marks client closed.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil
	}
	c.closed = true
	c.httpCli.CloseIdleConnections()
	return nil
}
