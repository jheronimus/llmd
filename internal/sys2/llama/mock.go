package llama

import (
	"strings"
	"sync"
)

// MockDriver simulates model loading and generation for deterministic unit tests.
type MockDriver struct {
	Mu          sync.Mutex
	Closed      bool
	NextOutput  string
	TokensLimit int
}

func NewMockDriver(sampleOutput string) *MockDriver {
	return &MockDriver{
		NextOutput:  sampleOutput,
		TokensLimit: 2048,
	}
}

func (m *MockDriver) LoadModel(modelPath string, nThreads, nCtx int) (Model, error) {
	m.Mu.Lock()
	defer m.Mu.Unlock()
	if m.Closed {
		return nil, ErrClosed
	}

	pieces := splitIntoPieces(m.NextOutput)
	return &MockModel{
		Output: m.NextOutput,
		pieces: pieces,
	}, nil
}

func (m *MockDriver) Close() error {
	m.Mu.Lock()
	defer m.Mu.Unlock()
	m.Closed = true
	return nil
}

type MockModel struct {
	Output string
	pieces []string
	Closed bool
}

func (m *MockModel) NewContext() (Context, error) {
	return &MockContext{
		pieces: m.pieces,
	}, nil
}

func (m *MockModel) Tokenize(text string, addSpecial bool) ([]int32, error) {
	words := strings.Fields(text)
	toks := make([]int32, len(words))
	for i := range words {
		toks[i] = int32(i + 1)
	}
	return toks, nil
}

func (m *MockModel) TokenToPiece(token int32) string {
	idx := int(token) - 1
	if idx >= 0 && idx < len(m.pieces) {
		return m.pieces[idx]
	}
	return ""
}

func (m *MockModel) Close() error {
	m.Closed = true
	return nil
}

type MockContext struct {
	pieces []string
	idx    int
}

func (c *MockContext) Eval(tokens []int32) error {
	return nil
}

func (c *MockContext) Sample(grammarStr string, temp, topP float32) (int32, error) {
	if c.idx >= len(c.pieces) {
		return -1, nil // EOG
	}
	c.idx++
	return int32(c.idx), nil
}

func (c *MockContext) IsEOG(token int32) bool {
	return token == -1
}

func (c *MockContext) Reset() {
	c.idx = 0
}

func (c *MockContext) Close() error {
	return nil
}

func splitIntoPieces(s string) []string {
	if s == "" {
		return nil
	}
	words := strings.Fields(s)
	if len(words) > 1 {
		pieces := make([]string, len(words))
		for i, w := range words {
			if i < len(words)-1 {
				pieces[i] = w + " "
			} else {
				pieces[i] = w
			}
		}
		return pieces
	}
	return []string{s}
}
