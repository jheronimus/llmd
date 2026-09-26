package llm

import (
	"errors"
	"time"

	"github.com/jheronimus/llm/internal/asset"
)

var (
	ErrModelNotCached = asset.ErrModelNotCached
	ErrLibNotCached   = asset.ErrLibNotCached
	ErrClientClosed   = errors.New("llm: client is closed")
)

// SystemTarget identifies an inference subsystem (System 1 or System 2).
type SystemTarget int

const (
	AllSystems SystemTarget = iota
	System1                 // Laya ONNX zero-shot classification
	System2                 // Qwen3 GGUF / llama.cpp generative & extraction
)

// StreamCallback receives newly generated text tokens during streaming.
// Returning false halts generation early.
type StreamCallback func(token string) bool

// Request specifies prompt inputs and sampling parameters for text generation.
type Request struct {
	// System provides guiding context or instructions.
	System string

	// Prompt contains the user input or query.
	Prompt string

	// RawPrompt bypasses ChatML template wrapping when true.
	RawPrompt bool

	// Temperature controls randomness (default: 0.1).
	Temperature float32

	// TopP sets nucleus sampling probability (default: 0.95).
	TopP float32

	// MaxTokens bounds generated output tokens (default: 1024).
	MaxTokens int

	// JSONSchema constrains sampling to match a specific schema.
	JSONSchema string

	// Grammar provides custom raw GBNF grammar for constrained sampling.
	Grammar string

	// StopTokens contains custom sequences that halt generation.
	StopTokens []string
}

// Response contains the result and execution metrics from Generate.
type Response struct {
	Text         string
	PromptTokens int
	OutputTokens int
	Duration     time.Duration
	FinishReason string
}

// Config holds runtime configuration options for Client.
type Config struct {
	CacheDir      string
	SocketPath    string
	DaemonBinary  string
	AutoSpawn     bool
	Threads       int
	ContextSize   int
	IdleTimeout   time.Duration
	AutoDownload  bool
	Sys1ModelDir  string
	Sys1LibPath   string
	Sys2ModelPath string
	Sys2LibPath   string
}

// Option modifies Client configuration.
type Option func(*Config)

// WithSocketPath overrides the Unix domain socket path (default: /tmp/llm.sock).
func WithSocketPath(path string) Option {
	return func(c *Config) {
		c.SocketPath = path
	}
}

// WithDaemonBinary sets explicit path to the llmd sidecar binary.
func WithDaemonBinary(path string) Option {
	return func(c *Config) {
		c.DaemonBinary = path
	}
}

// WithAutoSpawn controls whether to automatically launch llmd if not running (default: true).
func WithAutoSpawn(enable bool) Option {
	return func(c *Config) {
		c.AutoSpawn = enable
	}
}

// WithCacheDir overrides the root asset cache directory (~/.cache/llm).
func WithCacheDir(dir string) Option {
	return func(c *Config) {
		c.CacheDir = dir
	}
}

// WithThreads sets CPU thread count (default: runtime.NumCPU()).
func WithThreads(n int) Option {
	return func(c *Config) {
		c.Threads = n
	}
}

// WithContextSize sets token context window for System 2 (default: 8192).
func WithContextSize(size int) Option {
	return func(c *Config) {
		c.ContextSize = size
	}
}

// WithIdleTimeout unloads weights from RAM when inactive (0 disables, default: 10m).
func WithIdleTimeout(d time.Duration) Option {
	return func(c *Config) {
		c.IdleTimeout = d
	}
}

// WithAutoDownload enables automatic asset downloading when missing from cache.
func WithAutoDownload(enable bool) Option {
	return func(c *Config) {
		c.AutoDownload = enable
	}
}

// WithSys1Model sets a custom model directory for System 1 (ModernBERT/Laya).
func WithSys1Model(dir string) Option {
	return func(c *Config) {
		c.Sys1ModelDir = dir
	}
}

// WithSys2Model sets a custom GGUF model path for System 2.
func WithSys2Model(path string) Option {
	return func(c *Config) {
		c.Sys2ModelPath = path
	}
}
