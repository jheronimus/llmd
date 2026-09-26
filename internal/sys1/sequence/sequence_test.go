package sequence

import (
	"errors"
	"strings"
	"testing"
)

type mockTokenizer struct{}

func (m *mockTokenizer) Encode(text string, addSpecialTokens bool) ([]int64, error) {
	words := strings.Fields(text)
	res := make([]int64, len(words))
	for i := range words {
		res[i] = int64(100 + i)
	}
	return res, nil
}

func (m *mockTokenizer) CLSTokenID() int64  { return 1 }
func (m *mockTokenizer) SEPTokenID() int64  { return 2 }
func (m *mockTokenizer) PADTokenID() int64  { return 0 }
func (m *mockTokenizer) MASKTokenID() int64 { return 3 }

func TestRenderOptions(t *testing.T) {
	critChoice := map[string]string{
		"refund": "money back",
		"tech":   "technical help",
	}
	opts, err := RenderOptions(0, critChoice, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(opts) != 2 {
		t.Fatalf("expected 2 options, got %d", len(opts))
	}
	if opts[0] != "refund: money back" || opts[1] != "tech: technical help" {
		t.Errorf("unexpected choice render: %v", opts)
	}

	critScore := []string{"calm", "annoyed", "angry"}
	opts, err = RenderOptions(1, critScore, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(opts) != 3 || opts[0] != "level 0: calm" {
		t.Errorf("unexpected score render: %v", opts)
	}

	opts, err = RenderOptions(2, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(opts) != 2 || !strings.Contains(opts[0], "false: no") || !strings.Contains(opts[1], "true: yes") {
		t.Errorf("unexpected noul render: %v", opts)
	}
}

func TestBuildSequenceBudgeting(t *testing.T) {
	tok := &mockTokenizer{}

	maxLen := 50
	headMaxLen := 40
	longState := "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twentyone twentytwo twentythree twentyfour twentyfive twentysix twentyseven twentyeight twentynine thirty"

	// 1. Safe truncation
	seq, err := BuildSequence(tok, "q1", longState, 2, "noul", "Is urgent?", nil, nil, maxLen, headMaxLen, TruncateTail, false)
	if err != nil {
		t.Fatalf("unexpected error with truncation: %v", err)
	}
	if !seq.Usage.WasTruncated {
		t.Errorf("expected sequence to be truncated")
	}
	if seq.Usage.DroppedTokens <= 0 {
		t.Errorf("expected positive dropped tokens, got %d", seq.Usage.DroppedTokens)
	}
	if len(seq.InputIDs) > maxLen {
		t.Errorf("input IDs length %d exceeds maxLen %d", len(seq.InputIDs), maxLen)
	}

	// 2. Strict mode
	_, err = BuildSequence(tok, "q1", longState, 2, "noul", "Is urgent?", nil, nil, maxLen, headMaxLen, TruncateTail, true)
	if err == nil {
		t.Fatalf("expected ContextExceededError in strict mode, got nil")
	}
	var ctxErr *ContextExceededError
	if !errors.As(err, &ctxErr) {
		t.Errorf("expected *ContextExceededError, got %T: %v", err, err)
	}

	// 3. Option budget exceeded
	tinyHeadMaxLen := 5
	_, err = BuildSequence(tok, "q1", "short state", 2, "noul", "Is urgent?", nil, nil, maxLen, tinyHeadMaxLen, TruncateTail, false)
	if err == nil {
		t.Fatalf("expected OptionsBudgetExceededError, got nil")
	}
	var optErr *OptionsBudgetExceededError
	if !errors.As(err, &optErr) {
		t.Errorf("expected *OptionsBudgetExceededError, got %T: %v", err, err)
	}
}
