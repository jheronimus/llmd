package tokenizer

/*
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/lib/darwin_arm64 -framework CoreFoundation -framework Security
#cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/lib/darwin_amd64 -framework CoreFoundation -framework Security
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/lib/linux_amd64
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/lib/linux_arm64
*/
import "C"


import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	hf "github.com/daulet/tokenizers"
)

var (
	ErrSessionClosed = errors.New("golaya: tokenizer session is closed")
	ErrModelNotFound = errors.New("golaya: tokenizer file not found")
)

// Tokenizer defines the tokenization interface required by golaya.
type Tokenizer interface {
	Encode(text string, addSpecialTokens bool) ([]int64, error)
	Decode(tokens []int64, skipSpecialTokens bool) (string, error)
	CLSTokenID() int64
	SEPTokenID() int64
	MASKTokenID() int64
	PADTokenID() int64
	Close() error
}

// HFTokenizer wraps Hugging Face's Rust tokenizers engine.
type HFTokenizer struct {
	inner       *hf.Tokenizer
	clsTokenID  int64
	sepTokenID  int64
	maskTokenID int64
	padTokenID  int64
}

// NewHFTokenizer loads a tokenizer from tokenizer.json in modelDir.
func NewHFTokenizer(modelDir string) (*HFTokenizer, error) {
	tokPath := filepath.Join(modelDir, "tokenizer.json")
	if _, err := os.Stat(tokPath); err != nil {
		altPath := filepath.Join(modelDir, "tokenizer", "tokenizer.json")
		if _, errAlt := os.Stat(altPath); errAlt == nil {
			tokPath = altPath
		} else {
			return nil, fmt.Errorf("%w: %s", ErrModelNotFound, tokPath)
		}
	}

	raw, err := hf.FromFile(tokPath)
	if err != nil {
		return nil, fmt.Errorf("golaya: failed to load tokenizer from %s: %w", tokPath, err)
	}

	t := &HFTokenizer{
		inner:       raw,
		clsTokenID:  50281, // Default ModernBERT/mmBERT fallback
		sepTokenID:  50282,
		padTokenID:  50283,
		maskTokenID: 50284,
	}

	cfgPath := filepath.Join(modelDir, "tokenizer_config.json")
	if data, err := os.ReadFile(cfgPath); err == nil {
		t.parseSpecialTokensConfig(data)
	} else {
		t.resolveSpecialTokensViaEncode()
	}

	return t, nil
}

func (t *HFTokenizer) parseSpecialTokensConfig(data []byte) {
	var cfg struct {
		CLSToken  any `json:"cls_token"`
		SEPToken  any `json:"sep_token"`
		PADToken  any `json:"pad_token"`
		MASKToken any `json:"mask_token"`
	}
	if err := json.Unmarshal(data, &cfg); err == nil {
		t.resolveSpecialTokensViaEncode()
	}
}

func (t *HFTokenizer) resolveSpecialTokensViaEncode() {
	tokens := map[string]*int64{
		"[CLS]":  &t.clsTokenID,
		"[SEP]":  &t.sepTokenID,
		"[PAD]":  &t.padTokenID,
		"[MASK]": &t.maskTokenID,
	}
	for str, idPtr := range tokens {
		ids, _, err := t.inner.EncodeErr(str, false)
		if err == nil && len(ids) == 1 {
			*idPtr = int64(ids[0])
		}
	}
}

// Encode encodes a string to token IDs.
func (t *HFTokenizer) Encode(text string, addSpecialTokens bool) ([]int64, error) {
	if t.inner == nil {
		return nil, ErrSessionClosed
	}
	uids, _, err := t.inner.EncodeErr(text, addSpecialTokens)
	if err != nil {
		return nil, err
	}
	res := make([]int64, len(uids))
	for i, u := range uids {
		res[i] = int64(u)
	}
	return res, nil
}

// Decode converts token IDs back to a string.
func (t *HFTokenizer) Decode(tokens []int64, skipSpecialTokens bool) (string, error) {
	if t.inner == nil {
		return "", ErrSessionClosed
	}
	uids := make([]uint32, len(tokens))
	for i, id := range tokens {
		uids[i] = uint32(id)
	}
	return t.inner.DecodeErr(uids, skipSpecialTokens)
}

func (t *HFTokenizer) CLSTokenID() int64  { return t.clsTokenID }
func (t *HFTokenizer) SEPTokenID() int64  { return t.sepTokenID }
func (t *HFTokenizer) MASKTokenID() int64 { return t.maskTokenID }
func (t *HFTokenizer) PADTokenID() int64  { return t.padTokenID }

// Close releases the tokenizer resources.
func (t *HFTokenizer) Close() error {
	if t.inner != nil {
		err := t.inner.Close()
		t.inner = nil
		return err
	}
	return nil
}
