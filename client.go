package llm

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/jheronimus/llm/internal/asset"
	"github.com/jheronimus/llm/internal/sys1"
	"github.com/jheronimus/llm/internal/sys2"
)

// Client coordinates System 1 (Laya ONNX) and System 2 (Qwen3 GGUF) engines.
type Client struct {
	mu       sync.Mutex
	cfg      Config
	cacheDir string
	s1Engine *sys1.Engine
	s2Engine *sys2.Engine
	closed   bool
}

// New creates and initializes a Client.
func New(opts ...Option) (*Client, error) {
	cfg := Config{
		Threads:      runtime.NumCPU(),
		ContextSize:  8192,
		IdleTimeout:  10 * time.Minute,
		AutoDownload: true, // Default to true for developer convenience
	}

	for _, opt := range opts {
		opt(&cfg)
	}

	cacheDir, err := asset.EnsureCacheDir(cfg.CacheDir)
	if err != nil {
		return nil, fmt.Errorf("llm: cache dir error: %w", err)
	}

	return &Client{
		cfg:      cfg,
		cacheDir: cacheDir,
	}, nil
}

// Preload ensures models and native libraries are downloaded into the local cache.
func (c *Client) Preload(ctx context.Context, targets ...SystemTarget) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrClientClosed
	}

	target := AllSystems
	if len(targets) > 0 {
		target = targets[0]
	}

	if target == AllSystems || target == System1 {
		if _, err := asset.LocateOrDownloadONNX(c.cfg.Sys1LibPath, c.cacheDir, true); err != nil {
			return fmt.Errorf("preload onnx: %w", err)
		}
		if _, err := asset.LocateOrDownloadSys1Model(c.cfg.Sys1ModelDir, "", c.cacheDir, true); err != nil {
			return fmt.Errorf("preload sys1 model: %w", err)
		}
	}

	if target == AllSystems || target == System2 {
		if _, err := asset.LocateOrDownloadLlama(c.cfg.Sys2LibPath, c.cacheDir, true); err != nil {
			return fmt.Errorf("preload llama: %w", err)
		}
		if _, err := asset.LocateOrDownloadSys2Model(c.cfg.Sys2ModelPath, c.cacheDir, true); err != nil {
			return fmt.Errorf("preload sys2 model: %w", err)
		}
	}

	return nil
}

func (c *Client) getSys1() (*sys1.Engine, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, ErrClientClosed
	}

	if c.s1Engine != nil {
		return c.s1Engine, nil
	}

	libPath, err := asset.LocateOrDownloadONNX(c.cfg.Sys1LibPath, c.cacheDir, c.cfg.AutoDownload)
	if err != nil {
		return nil, err
	}

	modelDir, err := asset.LocateOrDownloadSys1Model(c.cfg.Sys1ModelDir, "", c.cacheDir, c.cfg.AutoDownload)
	if err != nil {
		return nil, err
	}

	eng, err := sys1.New(modelDir, libPath)
	if err != nil {
		return nil, err
	}

	c.s1Engine = eng
	return eng, nil
}

func (c *Client) getSys2() (*sys2.Engine, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, ErrClientClosed
	}

	if c.s2Engine != nil {
		return c.s2Engine, nil
	}

	libPath, err := asset.LocateOrDownloadLlama(c.cfg.Sys2LibPath, c.cacheDir, c.cfg.AutoDownload)
	if err != nil {
		return nil, err
	}

	modelPath, err := asset.LocateOrDownloadSys2Model(c.cfg.Sys2ModelPath, c.cacheDir, c.cfg.AutoDownload)
	if err != nil {
		return nil, err
	}

	eng, err := sys2.New(modelPath, libPath, c.cfg.Threads, c.cfg.ContextSize, c.cfg.IdleTimeout)
	if err != nil {
		return nil, err
	}

	c.s2Engine = eng
	return eng, nil
}

// Decide executes zero-shot classification on state and writes results into dest struct.
func (c *Client) Decide(ctx context.Context, state any, dest any) error {
	eng, err := c.getSys1()
	if err != nil {
		return err
	}
	return eng.Decide(ctx, state, dest)
}

// Extract generates structured output conforming to dest JSON schema via GBNF grammar.
func (c *Client) Extract(ctx context.Context, prompt string, dest any) error {
	return c.ExtractWithSystem(ctx, "", prompt, dest)
}

// ExtractWithSystem generates structured output using an explicit system prompt.
func (c *Client) ExtractWithSystem(ctx context.Context, system, prompt string, dest any) error {
	eng, err := c.getSys2()
	if err != nil {
		return err
	}
	return eng.ExtractWithSystem(ctx, system, prompt, dest)
}


// Generate executes autoregressive text generation.
func (c *Client) Generate(ctx context.Context, req Request) (Response, error) {
	eng, err := c.getSys2()
	if err != nil {
		return Response{}, err
	}
	r, err := eng.Generate(ctx, sys2.Request{
		System:      req.System,
		Prompt:      req.Prompt,
		RawPrompt:   req.RawPrompt,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MaxTokens:   req.MaxTokens,
		JSONSchema:  req.JSONSchema,
		Grammar:     req.Grammar,
		StopTokens:  req.StopTokens,
	})
	if err != nil {
		return Response{}, err
	}
	return Response{
		Text:         r.Text,
		PromptTokens: r.PromptTokens,
		OutputTokens: r.OutputTokens,
		Duration:     r.Duration,
		FinishReason: r.FinishReason,
	}, nil
}

// GenerateStream executes text generation with streaming callback.
func (c *Client) GenerateStream(ctx context.Context, req Request, cb StreamCallback) error {
	eng, err := c.getSys2()
	if err != nil {
		return err
	}
	_, err = eng.GenerateStream(ctx, sys2.Request{
		System:      req.System,
		Prompt:      req.Prompt,
		RawPrompt:   req.RawPrompt,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MaxTokens:   req.MaxTokens,
		JSONSchema:  req.JSONSchema,
		Grammar:     req.Grammar,
		StopTokens:  req.StopTokens,
	}, sys2.StreamCallback(cb))
	return err
}

// Close gracefully terminates active engines and releases allocated memory.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil
	}
	c.closed = true

	var errs []string
	if c.s1Engine != nil {
		if err := c.s1Engine.Close(); err != nil {
			errs = append(errs, err.Error())
		}
		c.s1Engine = nil
	}
	if c.s2Engine != nil {
		if err := c.s2Engine.Close(); err != nil {
			errs = append(errs, err.Error())
		}
		c.s2Engine = nil
	}

	if len(errs) > 0 {
		return fmt.Errorf("llm close errors: %s", errs)
	}
	return nil
}
