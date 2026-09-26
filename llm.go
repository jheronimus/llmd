package llm

import (
	"context"
	"sync"
)

var (
	defaultClientMu sync.RWMutex
	defaultClient   *Client
	defaultOpts     []Option
)

// Configure sets global options for the default client.
func Configure(opts ...Option) {
	defaultClientMu.Lock()
	defer defaultClientMu.Unlock()
	defaultOpts = append(defaultOpts, opts...)
	if defaultClient != nil {
		_ = defaultClient.Close()
		defaultClient = nil
	}
}

func getDefaultClient() (*Client, error) {
	defaultClientMu.RLock()
	c := defaultClient
	defaultClientMu.RUnlock()
	if c != nil {
		return c, nil
	}

	defaultClientMu.Lock()
	defer defaultClientMu.Unlock()
	if defaultClient != nil {
		return defaultClient, nil
	}

	c, err := New(defaultOpts...)
	if err != nil {
		return nil, err
	}
	defaultClient = c
	return c, nil
}

// Decide executes System 1 zero-shot classification on state, returning a populated struct of type T.
func Decide[T any](ctx context.Context, state any) (T, error) {
	var result T
	c, err := getDefaultClient()
	if err != nil {
		return result, err
	}
	if err := c.Decide(ctx, state, &result); err != nil {
		return result, err
	}
	return result, nil
}

// Extract executes System 2 structured JSON extraction, returning decoded struct of type T.
func Extract[T any](ctx context.Context, prompt string) (T, error) {
	var result T
	c, err := getDefaultClient()
	if err != nil {
		return result, err
	}
	if err := c.Extract(ctx, prompt, &result); err != nil {
		return result, err
	}
	return result, nil
}

// Generate executes System 2 text generation using the default client.
func Generate(ctx context.Context, req Request) (Response, error) {
	c, err := getDefaultClient()
	if err != nil {
		return Response{}, err
	}
	return c.Generate(ctx, req)
}

// GenerateStream executes System 2 streaming text generation with callback.
func GenerateStream(ctx context.Context, req Request, cb StreamCallback) error {
	c, err := getDefaultClient()
	if err != nil {
		return err
	}
	return c.GenerateStream(ctx, req, cb)
}

// Preload downloads all required assets for the default client into local cache.
func Preload(ctx context.Context, targets ...SystemTarget) error {
	c, err := getDefaultClient()
	if err != nil {
		return err
	}
	return c.Preload(ctx, targets...)
}
