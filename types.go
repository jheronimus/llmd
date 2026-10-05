package llmd

import (
	"errors"
	"time"
)

var (
	ErrClientClosed = errors.New("llmd: client is closed")
	ErrDaemonDown   = errors.New("llmd: daemon is unreachable")
)

// Request defines parameters for System 2 generation.
type Request struct {
	System      string   `json:"system,omitempty"`
	Prompt      string   `json:"prompt"`
	Grammar     string   `json:"grammar,omitempty"`
	JSONSchema  string   `json:"json_schema,omitempty"`
	Temperature float32  `json:"temperature,omitempty"`
	TopP        float32  `json:"top_p,omitempty"`
	MaxTokens   int      `json:"max_tokens,omitempty"`
}

// Response contains generated text and performance metrics.
type Response struct {
	Text       string        `json:"text"`
	Duration   time.Duration `json:"duration"`
}

// DecideQuestion defines a single System 1 question.
type DecideQuestion struct {
	Type         string            `json:"type"` // "choice", "score", "noul"
	Instructions string            `json:"instructions"`
	Criteria     any               `json:"criteria,omitempty"`
	Labels       map[string]string `json:"labels,omitempty"`
}

// DecideAnswer contains System 1 evaluation metrics.
type DecideAnswer struct {
	Answer     any     `json:"answer"`
	Confidence float32 `json:"confidence"`
	ActCost    float32 `json:"act_cost"`
	Escalated  bool    `json:"escalated"`
}

// Config defines LLMD client options.
type Config struct {
	BaseURL string
	Timeout time.Duration
}

// Option modifies Client configuration.
type Option func(*Config)

// WithBaseURL overrides the daemon HTTP endpoint (default: http://127.0.0.1:8080).
func WithBaseURL(url string) Option {
	return func(c *Config) {
		c.BaseURL = url
	}
}

// WithTimeout sets HTTP client timeout (default: 2 minutes).
func WithTimeout(d time.Duration) Option {
	return func(c *Config) {
		c.Timeout = d
	}
}
