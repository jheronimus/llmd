// Package llama defines the low-level model and context driver interfaces
// interacting with llama.cpp C runtimes.
package llama

import (
	"context"
	"errors"
)

var (
	ErrContextFull = errors.New("llama: context limit reached")
	ErrClosed      = errors.New("llama: driver is closed")
)

// Driver manages model loading and global backend lifecycle.
type Driver interface {
	LoadModel(modelPath string, nThreads int, nCtx int) (Model, error)
	Close() error
}

// Model represents loaded GGUF neural weights in RAM.
type Model interface {
	NewContext() (Context, error)
	Tokenize(text string, addSpecial bool) ([]int32, error)
	TokenToPiece(token int32) string
	Close() error
}

// Context manages runtime KV cache, prompt evaluation, and token sampling.
type Context interface {
	Eval(tokens []int32) error
	Sample(grammarStr string, temp float32, topP float32) (int32, error)
	IsEOG(token int32) bool
	Reset()
	Close() error
}

// TokenStreamFunc receives each decoded piece. Return false to cancel.
type TokenStreamFunc func(piece string) bool

// InferenceSession orchestrates prompt tokenization, evaluation, and generation loop.
type InferenceSession struct {
	Model Model
	Ctx   Context
}

// Generate runs an end-to-end generation loop with context cancellation.
func (s *InferenceSession) Generate(
	ctx context.Context,
	prompt string,
	maxTokens int,
	grammarStr string,
	temp, topP float32,
	stopTokens []string,
	onPiece TokenStreamFunc,
) (string, int, int, error) {
	if s.Model == nil || s.Ctx == nil {
		return "", 0, 0, ErrClosed
	}

	tokens, err := s.Model.Tokenize(prompt, true)
	if err != nil {
		return "", 0, 0, err
	}

	promptLen := len(tokens)
	if err := s.Ctx.Eval(tokens); err != nil {
		return "", promptLen, 0, err
	}

	var (
		outPieces []string
		outTokens int
	)

	for outTokens < maxTokens {
		select {
		case <-ctx.Done():
			return joinPieces(outPieces), promptLen, outTokens, ctx.Err()
		default:
		}

		tok, err := s.Ctx.Sample(grammarStr, temp, topP)
		if err != nil {
			return joinPieces(outPieces), promptLen, outTokens, err
		}

		if s.Ctx.IsEOG(tok) {
			break
		}

		piece := s.Model.TokenToPiece(tok)
		outPieces = append(outPieces, piece)
		outTokens++

		if onPiece != nil {
			if !onPiece(piece) {
				break
			}
		}

		// Check custom stop sequences
		combined := joinPieces(outPieces)
		stopped := false
		for _, stop := range stopTokens {
			if stop != "" && len(combined) >= len(stop) && combined[len(combined)-len(stop):] == stop {
				stopped = true
				break
			}
		}
		if stopped {
			break
		}

		// Feed generated token back into context
		if err := s.Ctx.Eval([]int32{tok}); err != nil {
			return joinPieces(outPieces), promptLen, outTokens, err
		}
	}

	return joinPieces(outPieces), promptLen, outTokens, nil
}

func joinPieces(pieces []string) string {
	var total int
	for _, p := range pieces {
		total += len(p)
	}
	b := make([]byte, 0, total)
	for _, p := range pieces {
		b = append(b, p...)
	}
	return string(b)
}
