package sys2

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/jheronimus/llm/internal/sys2/grammar"
	"github.com/jheronimus/llm/internal/sys2/llama"
)

var (
	ErrClosed           = errors.New("llm: sys2 engine is closed")
	ErrGenerationFailed = errors.New("llm: text generation failed")
)

type StreamCallback func(token string) bool

type Request struct {
	System      string
	Prompt      string
	RawPrompt   bool
	Temperature float32
	TopP        float32
	MaxTokens   int
	JSONSchema  string
	Grammar     string
	StopTokens  []string
}

type Response struct {
	Text         string
	PromptTokens int
	OutputTokens int
	Duration     time.Duration
	FinishReason string
}

type Engine struct {
	mu           sync.Mutex
	driver       llama.Driver
	model        llama.Model
	activeCtx    llama.Context
	idleTimer    *time.Timer
	idleTimeout  time.Duration
	modelPath    string
	threads      int
	contextSize  int
	closed       bool
}

func New(modelPath, libPath string, threads, contextSize int, idleTimeout time.Duration) (*Engine, error) {
	if threads <= 0 {
		threads = runtime.NumCPU()
	}
	if contextSize <= 0 {
		contextSize = 8192
	}

	drv, err := llama.NewNativeDriver(libPath)
	if err != nil {
		return nil, fmt.Errorf("llm: failed to load llama driver from %s: %w", libPath, err)
	}

	return &Engine{
		driver:      drv,
		modelPath:   modelPath,
		threads:     threads,
		contextSize: contextSize,
		idleTimeout: idleTimeout,
	}, nil
}

func (e *Engine) acquireSession() (*llama.InferenceSession, func(), error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return nil, nil, ErrClosed
	}

	if e.idleTimer != nil {
		e.idleTimer.Stop()
	}

	if e.model == nil {
		m, err := e.driver.LoadModel(e.modelPath, e.threads, e.contextSize)
		if err != nil {
			return nil, nil, fmt.Errorf("llm: failed loading model from %s: %w", e.modelPath, err)
		}
		e.model = m
	}

	if e.activeCtx == nil {
		c, err := e.model.NewContext()
		if err != nil {
			return nil, nil, fmt.Errorf("llm: failed creating context: %w", err)
		}
		e.activeCtx = c
	}

	sess := &llama.InferenceSession{
		Model: e.model,
		Ctx:   e.activeCtx,
	}

	release := func() {
		e.mu.Lock()
		defer e.mu.Unlock()

		if e.activeCtx != nil {
			e.activeCtx.Reset()
		}

		if e.idleTimeout > 0 && !e.closed {
			e.idleTimer = time.AfterFunc(e.idleTimeout, func() {
				e.mu.Lock()
				defer e.mu.Unlock()
				if e.activeCtx != nil {
					_ = e.activeCtx.Close()
					e.activeCtx = nil
				}
				if e.model != nil {
					_ = e.model.Close()
					e.model = nil
				}
			})
		}
	}

	return sess, release, nil
}

func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return nil
	}
	e.closed = true

	if e.idleTimer != nil {
		e.idleTimer.Stop()
	}

	if e.activeCtx != nil {
		_ = e.activeCtx.Close()
		e.activeCtx = nil
	}
	if e.model != nil {
		_ = e.model.Close()
		e.model = nil
	}

	return e.driver.Close()
}

func (e *Engine) Generate(ctx context.Context, req Request) (Response, error) {
	return e.GenerateStream(ctx, req, nil)
}

func (e *Engine) GenerateStream(ctx context.Context, req Request, cb StreamCallback) (Response, error) {
	var empty Response
	start := time.Now()

	sess, release, err := e.acquireSession()
	if err != nil {
		return empty, err
	}
	defer release()

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 1024
	}

	temp := req.Temperature
	if temp < 0 {
		temp = 0.1
	}

	topP := req.TopP
	if topP <= 0 || topP > 1.0 {
		topP = 0.95
	}

	grammarStr := req.Grammar
	if grammarStr == "" && req.JSONSchema != "" {
		g, err := grammar.FromJSONSchema(req.JSONSchema)
		if err == nil {
			grammarStr = g
		}
	}

	formattedPrompt := req.Prompt
	if !req.RawPrompt {
		formattedPrompt = formatChatML(req.System, req.Prompt)
	}

	stopTokens := req.StopTokens
	hasEnd := false
	for _, s := range stopTokens {
		if s == "<|im_end|>" {
			hasEnd = true
			break
		}
	}
	if !hasEnd {
		stopTokens = append(stopTokens, "<|im_end|>")
	}

	text, promptTokens, outTokens, err := sess.Generate(
		ctx,
		formattedPrompt,
		maxTokens,
		grammarStr,
		temp,
		topP,
		stopTokens,
		llama.TokenStreamFunc(cb),
	)
	if err != nil {
		return empty, fmt.Errorf("%w: %w", ErrGenerationFailed, err)
	}

	finishReason := "stop"
	if outTokens >= maxTokens {
		finishReason = "length"
	}

	cleanText := cleanResponseText(text)

	return Response{
		Text:         cleanText,
		PromptTokens: promptTokens,
		OutputTokens: outTokens,
		Duration:     time.Since(start),
		FinishReason: finishReason,
	}, nil
}

func (e *Engine) Extract(ctx context.Context, prompt string, dest any) error {
	return e.ExtractWithSystem(ctx, "", prompt, dest)
}

func (e *Engine) ExtractWithSystem(ctx context.Context, system, prompt string, dest any) error {
	if dest == nil {
		return errors.New("llm: extract destination cannot be nil")
	}

	gStr := grammar.GenericJSON()

	req := Request{
		System:    system,
		Prompt:    prompt,
		Grammar:   gStr,
		MaxTokens: 2048,
	}

	resp, err := e.Generate(ctx, req)
	if err != nil {
		return err
	}

	clean := cleanResponseText(resp.Text)
	if err := json.Unmarshal([]byte(clean), dest); err != nil {
		return fmt.Errorf("llm: failed to unmarshal JSON output: %w (output: %s)", err, clean)
	}

	return nil
}


func formatChatML(system, user string) string {
	var b strings.Builder
	b.WriteString("<|im_start|>system\n")
	if strings.TrimSpace(system) != "" {
		b.WriteString(strings.TrimSpace(system))
		b.WriteString("\n")
	}
	b.WriteString("Do not output thoughts. Directly output valid answer only.\n<|im_end|>\n<|im_start|>user\n")
	b.WriteString(strings.TrimSpace(user))
	b.WriteString("\n<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n")
	return b.String()
}

func cleanResponseText(raw string) string {
	s := strings.TrimSpace(raw)
	s = strings.TrimSuffix(s, "<|im_end|>")
	s = strings.TrimSuffix(s, "<end_of_turn>")
	s = strings.TrimSpace(s)

	if idx := strings.Index(s, "</think>"); idx != -1 {
		s = strings.TrimSpace(s[idx+8:])
	}
	if strings.HasPrefix(s, "```json") {
		s = strings.TrimPrefix(s, "```json")
		s = strings.TrimSuffix(s, "```")
	} else if strings.HasPrefix(s, "```") {
		s = strings.TrimPrefix(s, "```")
		s = strings.TrimSuffix(s, "```")
	}
	return strings.TrimSpace(s)
}
