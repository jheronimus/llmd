package tokenizer

/*
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/lib/darwin_arm64 -ltokenizers -lm -lstdc++ -framework CoreFoundation -framework Security
#cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/lib/darwin_amd64 -ltokenizers -lm -lstdc++ -framework CoreFoundation -framework Security
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/lib/linux_amd64 -ltokenizers -ldl -lm -lstdc++
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/lib/linux_arm64 -ltokenizers -ldl -lm -lstdc++

#include <stdlib.h>
#include "tokenizers.h"

extern void tokenizers_version_1_26_0(void);
static void (*tokenizers_version_check)(void) = &tokenizers_version_1_26_0;
*/
import "C"

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"unsafe"
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

// HFTokenizer wraps Hugging Face's Rust tokenizers engine directly via Cgo.
type HFTokenizer struct {
	mu          sync.Mutex
	handle      unsafe.Pointer
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

	cPath := C.CString(tokPath)
	defer C.free(unsafe.Pointer(cPath))

	var errPtr *C.char
	handle := C.tokenizers_from_file(cPath, &errPtr)
	if handle == nil {
		if errPtr != nil {
			errStr := C.GoString(errPtr)
			C.tokenizers_free_string(errPtr)
			return nil, fmt.Errorf("golaya: failed to load tokenizer: %s", errStr)
		}
		return nil, fmt.Errorf("golaya: failed to load tokenizer from %s", tokPath)
	}

	t := &HFTokenizer{
		handle:      handle,
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
		ids, err := t.encodeRaw(str, false)
		if err == nil && len(ids) == 1 {
			*idPtr = ids[0]
		}
	}
}

func (t *HFTokenizer) encodeRaw(text string, addSpecialTokens bool) ([]int64, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.handle == nil {
		return nil, ErrSessionClosed
	}

	cStr := C.CString(text)
	defer C.free(unsafe.Pointer(cStr))

	opts := C.struct_tokenizers_encode_options{
		add_special_tokens: C.bool(addSpecialTokens),
		return_tokens:     C.bool(false),
	}

	buf := C.tokenizers_encode(t.handle, cStr, &opts)
	n := int(buf.len)
	if n == 0 {
		return nil, nil
	}
	defer C.tokenizers_free_buffer(buf)

	slice := unsafe.Slice(buf.ids, n)
	res := make([]int64, n)
	for i, id := range slice {
		res[i] = int64(id)
	}
	return res, nil
}

// Encode encodes a string to token IDs.
func (t *HFTokenizer) Encode(text string, addSpecialTokens bool) ([]int64, error) {
	return t.encodeRaw(text, addSpecialTokens)
}

// Decode converts token IDs back to a string.
func (t *HFTokenizer) Decode(tokens []int64, skipSpecialTokens bool) (string, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.handle == nil {
		return "", ErrSessionClosed
	}
	if len(tokens) == 0 {
		return "", nil
	}

	uids := make([]C.uint32_t, len(tokens))
	for i, id := range tokens {
		uids[i] = C.uint32_t(id)
	}

	cStr := C.tokenizers_decode(t.handle, &uids[0], C.uint32_t(len(uids)), C.bool(skipSpecialTokens))
	if cStr == nil {
		return "", fmt.Errorf("golaya: failed to decode tokens")
	}
	defer C.tokenizers_free_string(cStr)

	return C.GoString(cStr), nil
}

func (t *HFTokenizer) CLSTokenID() int64  { return t.clsTokenID }
func (t *HFTokenizer) SEPTokenID() int64  { return t.sepTokenID }
func (t *HFTokenizer) MASKTokenID() int64 { return t.maskTokenID }
func (t *HFTokenizer) PADTokenID() int64  { return t.padTokenID }

// Close releases tokenizer resources.
func (t *HFTokenizer) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.handle != nil {
		C.tokenizers_free_tokenizer(t.handle)
		t.handle = nil
	}
	return nil
}
