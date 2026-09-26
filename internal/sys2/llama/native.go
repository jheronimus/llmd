//go:build darwin || linux

package llama

/*
#cgo CFLAGS: -O2
#cgo darwin LDFLAGS: -ldl
#cgo linux LDFLAGS: -ldl

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdbool.h>
#include "llama.h"

// Function pointer typedefs matching official llama.h
typedef void (*fn_llama_log_set)(ggml_log_callback log_callback, void * user_data);
typedef void (*fn_llama_backend_init)(void);
typedef void (*fn_llama_backend_free)(void);
typedef struct llama_model_params (*fn_llama_model_default_params)(void);
typedef struct llama_model * (*fn_llama_model_load_from_file)(const char * path, struct llama_model_params params);
typedef void (*fn_llama_model_free)(struct llama_model * model);
typedef const struct llama_vocab * (*fn_llama_model_get_vocab)(const struct llama_model * model);
typedef struct llama_context_params (*fn_llama_context_default_params)(void);
typedef struct llama_context * (*fn_llama_init_from_model)(struct llama_model * model, struct llama_context_params params);
typedef void (*fn_llama_free)(struct llama_context * ctx);

typedef int32_t (*fn_llama_tokenize)(const struct llama_vocab * vocab, const char * text, int32_t text_len, llama_token * tokens, int32_t n_tokens_max, bool add_special, bool parse_special);
typedef int32_t (*fn_llama_token_to_piece)(const struct llama_vocab * vocab, llama_token token, char * buf, int32_t length, int32_t lstrip, bool special);
typedef bool (*fn_llama_token_is_eog)(const struct llama_vocab * vocab, llama_token token);

typedef struct llama_batch (*fn_llama_batch_init)(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
typedef void (*fn_llama_batch_free)(struct llama_batch batch);
typedef int32_t (*fn_llama_decode)(struct llama_context * ctx, struct llama_batch batch);

typedef struct llama_sampler_chain_params (*fn_llama_sampler_chain_default_params)(void);
typedef struct llama_sampler * (*fn_llama_sampler_chain_init)(struct llama_sampler_chain_params params);
typedef void (*fn_llama_sampler_chain_add)(struct llama_sampler * chain, struct llama_sampler * smpl);
typedef struct llama_sampler * (*fn_llama_sampler_init_temp)(float temp);
typedef struct llama_sampler * (*fn_llama_sampler_init_top_p)(float p, size_t min_keep);
typedef struct llama_sampler * (*fn_llama_sampler_init_grammar)(const struct llama_vocab * vocab, const char * grammar_str, const char * grammar_root);
typedef struct llama_sampler * (*fn_llama_sampler_init_dist)(uint32_t seed);
typedef struct llama_sampler * (*fn_llama_sampler_init_greedy)(void);
typedef llama_token (*fn_llama_sampler_sample)(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx);
typedef void (*fn_llama_sampler_free)(struct llama_sampler * smpl);
typedef llama_memory_t (*fn_llama_get_memory)(const struct llama_context * ctx);
typedef void (*fn_llama_memory_clear)(llama_memory_t mem, bool data);

typedef struct {
    void * handle;
    fn_llama_backend_init backend_init;
    fn_llama_backend_free backend_free;
    fn_llama_model_default_params model_default_params;
    fn_llama_model_load_from_file model_load_from_file;
    fn_llama_model_free model_free;
    fn_llama_model_get_vocab model_get_vocab;
    fn_llama_context_default_params context_default_params;
    fn_llama_init_from_model init_from_model;
    fn_llama_free free_context;
    fn_llama_tokenize tokenize;
    fn_llama_token_to_piece token_to_piece;
    fn_llama_token_is_eog token_is_eog;
    fn_llama_batch_init batch_init;
    fn_llama_batch_free batch_free;
    fn_llama_decode decode;
    fn_llama_sampler_chain_default_params sampler_chain_default_params;
    fn_llama_sampler_chain_init sampler_chain_init;
    fn_llama_sampler_chain_add sampler_chain_add;
    fn_llama_sampler_init_temp sampler_init_temp;
    fn_llama_sampler_init_top_p sampler_init_top_p;
    fn_llama_sampler_init_grammar sampler_init_grammar;
    fn_llama_sampler_init_dist sampler_init_dist;
    fn_llama_sampler_init_greedy sampler_init_greedy;
    fn_llama_sampler_sample sampler_sample;
    fn_llama_sampler_free sampler_free;
    fn_llama_get_memory get_memory;
    fn_llama_memory_clear memory_clear;
    fn_llama_log_set log_set;
} llama_binding_t;

static llama_binding_t g_binding;

static void c_llama_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void)user_data;
    static int checked_env = 0;
    static int verbose = 0;
    static int last_was_error = 0;

    if (!checked_env) {
        const char * env = getenv("LLM_DEBUG");
        if (!env) {
            env = getenv("LLAMA_LOG_VERBOSE");
        }
        if (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) {
            verbose = 1;
        }
        checked_env = 1;
    }

    if (verbose) {
        fputs(text, stderr);
        return;
    }

    if (level == GGML_LOG_LEVEL_ERROR) {
        last_was_error = 1;
        fputs(text, stderr);
    } else if (level == GGML_LOG_LEVEL_CONT && last_was_error) {
        fputs(text, stderr);
    } else {
        last_was_error = 0;
    }
}

static int load_llama_symbols(const char * lib_path, char * err_buf, size_t err_size) {
    void * h = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        const char * err = dlerror();
        if (err) {
            snprintf(err_buf, err_size, "%s", err);
        } else {
            snprintf(err_buf, err_size, "unknown dlopen error");
        }
        return -1;
    }

    g_binding.handle = h;
    #define LOAD_SYM(name) \
        g_binding.name = (fn_llama_##name)dlsym(h, "llama_" #name); \
        if (!g_binding.name) { \
            snprintf(err_buf, err_size, "missing symbol llama_" #name); \
            return -2; \
        }

    LOAD_SYM(backend_init)
    LOAD_SYM(backend_free)
    LOAD_SYM(model_default_params)
    LOAD_SYM(model_load_from_file)
    LOAD_SYM(model_free)
    LOAD_SYM(model_get_vocab)
    LOAD_SYM(context_default_params)
    LOAD_SYM(init_from_model)
    LOAD_SYM(tokenize)
    LOAD_SYM(token_to_piece)
    LOAD_SYM(token_is_eog)
    LOAD_SYM(batch_init)
    LOAD_SYM(batch_free)
    LOAD_SYM(decode)
    LOAD_SYM(sampler_chain_default_params)
    LOAD_SYM(sampler_chain_init)
    LOAD_SYM(sampler_chain_add)
    LOAD_SYM(sampler_init_temp)
    LOAD_SYM(sampler_init_top_p)
    LOAD_SYM(sampler_init_grammar)
    LOAD_SYM(sampler_init_dist)
    LOAD_SYM(sampler_init_greedy)
    LOAD_SYM(sampler_sample)
    LOAD_SYM(sampler_free)
    LOAD_SYM(get_memory)
    LOAD_SYM(memory_clear)

    g_binding.free_context = (fn_llama_free)dlsym(h, "llama_free");
    g_binding.log_set = (fn_llama_log_set)dlsym(h, "llama_log_set");
    if (g_binding.log_set) {
        g_binding.log_set(c_llama_log_callback, NULL);
    }

    #undef LOAD_SYM

    g_binding.backend_init();
    return 0;
}

static void close_llama_lib(void) {
    if (g_binding.backend_free) {
        g_binding.backend_free();
    }
    if (g_binding.handle) {
        dlclose(g_binding.handle);
        g_binding.handle = NULL;
    }
}

// C Helper callers
static void * c_llama_model_load(const char * path) {
    if (!g_binding.model_load_from_file || !g_binding.model_default_params) return NULL;
    return g_binding.model_load_from_file(path, g_binding.model_default_params());
}

static void c_llama_model_free(void * model) {
    if (g_binding.model_free) g_binding.model_free(model);
}

static void * c_llama_model_get_vocab(void * model) {
    if (g_binding.model_get_vocab) return (void *)g_binding.model_get_vocab(model);
    return NULL;
}


static void * c_llama_init_ctx(void * model, uint32_t n_ctx, int32_t n_threads) {
    if (!g_binding.init_from_model || !g_binding.context_default_params) return NULL;
    struct llama_context_params params = g_binding.context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = 4096;
    params.n_ubatch = 512;
    params.n_threads = n_threads;
    params.n_threads_batch = n_threads;
    return g_binding.init_from_model((struct llama_model *)model, params);
}

static void c_llama_free_ctx(void * ctx) {
    if (g_binding.free_context) g_binding.free_context((struct llama_context *)ctx);
}

static int32_t c_llama_tokenize(const void * vocab, const char * text, int32_t text_len, int32_t * tokens, int32_t max_toks, bool add_special) {
    if (!g_binding.tokenize) return -1;
    return g_binding.tokenize((const struct llama_vocab *)vocab, text, text_len, (llama_token *)tokens, max_toks, add_special, true);
}

static int32_t c_llama_token_to_piece(const void * vocab, int32_t token, char * buf, int32_t length) {
    if (!g_binding.token_to_piece) return -1;
    return g_binding.token_to_piece((const struct llama_vocab *)vocab, (llama_token)token, buf, length, 0, false);
}

static bool c_llama_token_is_eog(const void * vocab, int32_t token) {
    if (!g_binding.token_is_eog) return false;
    return g_binding.token_is_eog((const struct llama_vocab *)vocab, (llama_token)token);
}

static struct llama_batch c_llama_batch_init(int32_t n_tokens) {
    if (!g_binding.batch_init) {
        struct llama_batch empty = {0};
        return empty;
    }
    return g_binding.batch_init(n_tokens, 0, 1);
}

static void c_llama_batch_free(struct llama_batch batch) {
    if (g_binding.batch_free) g_binding.batch_free(batch);
}

static int32_t c_llama_decode(void * ctx, struct llama_batch batch) {
    if (!g_binding.decode) return -1;
    return g_binding.decode((struct llama_context *)ctx, batch);
}

static void * c_llama_init_sampler(const void * vocab, const char * grammar_str, float temp, float top_p) {
    if (!g_binding.sampler_chain_init) return NULL;

    struct llama_sampler_chain_params sparams = {0};
    if (g_binding.sampler_chain_default_params) {
        sparams = g_binding.sampler_chain_default_params();
    }
    struct llama_sampler * chain = g_binding.sampler_chain_init(sparams);
    if (!chain) return NULL;

    if (grammar_str && grammar_str[0] != '\0' && g_binding.sampler_init_grammar) {
        struct llama_sampler * g_smpl = g_binding.sampler_init_grammar((const struct llama_vocab *)vocab, grammar_str, "root");
        if (g_smpl) {
            g_binding.sampler_chain_add(chain, g_smpl);
        }
    }

    if (top_p > 0.0f && top_p < 1.0f && g_binding.sampler_init_top_p) {
        struct llama_sampler * p_smpl = g_binding.sampler_init_top_p(top_p, 1);
        if (p_smpl) {
            g_binding.sampler_chain_add(chain, p_smpl);
        }
    }

    if (temp > 0.0f && g_binding.sampler_init_temp) {
        struct llama_sampler * t_smpl = g_binding.sampler_init_temp(temp);
        if (t_smpl) {
            g_binding.sampler_chain_add(chain, t_smpl);
        }
    }

    if (temp <= 0.0f && g_binding.sampler_init_greedy) {
        struct llama_sampler * g_smpl = g_binding.sampler_init_greedy();
        if (g_smpl) {
            g_binding.sampler_chain_add(chain, g_smpl);
        }
    } else if (g_binding.sampler_init_dist) {
        struct llama_sampler * d_smpl = g_binding.sampler_init_dist(1337);
        if (d_smpl) {
            g_binding.sampler_chain_add(chain, d_smpl);
        }
    }

    return chain;
}

static int32_t c_llama_sample_token(void * sampler, void * ctx) {
    if (!g_binding.sampler_sample || !sampler || !ctx) return -1;
    llama_token tok = g_binding.sampler_sample((struct llama_sampler *)sampler, (struct llama_context *)ctx, -1);
    return (int32_t)tok;
}

static void c_llama_free_sampler(void * sampler) {
    if (g_binding.sampler_free && sampler) {
        g_binding.sampler_free((struct llama_sampler *)sampler);
    }
}

static void c_llama_ctx_reset(void * ctx) {
    if (g_binding.get_memory && g_binding.memory_clear && ctx) {
        g_binding.memory_clear(g_binding.get_memory((const struct llama_context *)ctx), true);
    }
}
*/
import "C"

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"
)

var (
	nativeMu    sync.Mutex
	nativeLoads int
)

// NativeDriver connects to an active libllama shared object using dlopen.
type NativeDriver struct {
	libPath string
}

// NewNativeDriver initializes the native llama backend from libPath.
func NewNativeDriver(libPath string) (*NativeDriver, error) {
	nativeMu.Lock()
	defer nativeMu.Unlock()

	cLibPath := C.CString(libPath)
	defer C.free(unsafe.Pointer(cLibPath))

	var errBuf [512]C.char
	code := C.load_llama_symbols(cLibPath, &errBuf[0], C.size_t(len(errBuf)))
	if code != 0 {
		return nil, fmt.Errorf("gogemma: failed to load %s: %s", libPath, C.GoString(&errBuf[0]))
	}

	nativeLoads++
	return &NativeDriver{libPath: libPath}, nil
}

func (d *NativeDriver) LoadModel(modelPath string, nThreads, nCtx int) (Model, error) {
	nativeMu.Lock()
	defer nativeMu.Unlock()

	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	modelPtr := C.c_llama_model_load(cPath)
	if modelPtr == nil {
		return nil, fmt.Errorf("gogemma: failed to load model from %s", modelPath)
	}

	vocabPtr := C.c_llama_model_get_vocab(modelPtr)

	return &NativeModel{
		modelPtr: modelPtr,
		vocabPtr: vocabPtr,
		nThreads: nThreads,
		nCtx:     nCtx,
	}, nil
}

func (d *NativeDriver) Close() error {
	nativeMu.Lock()
	defer nativeMu.Unlock()

	if nativeLoads > 0 {
		nativeLoads--
		if nativeLoads == 0 {
			C.close_llama_lib()
		}
	}
	return nil
}

type NativeModel struct {
	mu       sync.Mutex
	modelPtr unsafe.Pointer
	vocabPtr unsafe.Pointer
	nThreads int
	nCtx     int
	closed   bool
}

func (m *NativeModel) NewContext() (Context, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, ErrClosed
	}

	ctxPtr := C.c_llama_init_ctx(m.modelPtr, C.uint32_t(m.nCtx), C.int32_t(m.nThreads))
	if ctxPtr == nil {
		return nil, errors.New("gogemma: failed initializing context from model")
	}

	batch := C.c_llama_batch_init(C.int32_t(m.nCtx))

	return &NativeContext{
		model:    m,
		ctxPtr:   ctxPtr,
		batch:    batch,
		maxBatch: m.nCtx,
	}, nil
}

func (m *NativeModel) Tokenize(text string, addSpecial bool) ([]int32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, ErrClosed
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	// Initial buffer estimate
	maxTokens := len(text) + 64
	tokens := make([]int32, maxTokens)

	n := C.c_llama_tokenize(
		m.vocabPtr,
		cText,
		C.int32_t(len(text)),
		(*C.int32_t)(unsafe.Pointer(&tokens[0])),
		C.int32_t(maxTokens),
		C.bool(addSpecial),
	)

	if n < 0 {
		// Need larger buffer
		tokens = make([]int32, -n)
		n = C.c_llama_tokenize(
			m.vocabPtr,
			cText,
			C.int32_t(len(text)),
			(*C.int32_t)(unsafe.Pointer(&tokens[0])),
			C.int32_t(-n),
			C.bool(addSpecial),
		)
	}

	if n < 0 {
		return nil, errors.New("gogemma: tokenization error")
	}

	return tokens[:n], nil
}

func (m *NativeModel) TokenToPiece(token int32) string {
	var buf [256]C.char
	n := C.c_llama_token_to_piece(
		m.vocabPtr,
		C.int32_t(token),
		&buf[0],
		C.int32_t(len(buf)),
	)
	if n <= 0 {
		return ""
	}
	return C.GoStringN(&buf[0], n)
}

func (m *NativeModel) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true
	if m.modelPtr != nil {
		C.c_llama_model_free(m.modelPtr)
		m.modelPtr = nil
	}
	return nil
}

type NativeContext struct {
	mu         sync.Mutex
	model      *NativeModel
	ctxPtr     unsafe.Pointer
	samplerPtr unsafe.Pointer
	batch      C.struct_llama_batch
	maxBatch   int
	nPast      int
	closed     bool
}

func (c *NativeContext) Eval(tokens []int32) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return ErrClosed
	}

	total := len(tokens)
	if total == 0 {
		return nil
	}

	chunkSize := 512
	for offset := 0; offset < total; offset += chunkSize {
		end := offset + chunkSize
		if end > total {
			end = total
		}
		chunk := tokens[offset:end]
		n := len(chunk)

		c.batch.n_tokens = C.int32_t(n)
		for i, tok := range chunk {
			*(*C.int32_t)(unsafe.Pointer(uintptr(unsafe.Pointer(c.batch.token)) + uintptr(i)*4)) = C.int32_t(tok)
			*(*C.int32_t)(unsafe.Pointer(uintptr(unsafe.Pointer(c.batch.pos)) + uintptr(i)*4)) = C.int32_t(c.nPast + offset + i)
			*(*C.int32_t)(unsafe.Pointer(uintptr(unsafe.Pointer(c.batch.n_seq_id)) + uintptr(i)*4)) = 1
			*(*int8)(unsafe.Pointer(uintptr(unsafe.Pointer(c.batch.logits)) + uintptr(i))) = 0
		}
		// Logits enabled only for the final token of the entire prompt
		if end == total {
			*(*int8)(unsafe.Pointer(uintptr(unsafe.Pointer(c.batch.logits)) + uintptr(n-1))) = 1
		}

		if code := C.c_llama_decode(c.ctxPtr, c.batch); code != 0 {
			return fmt.Errorf("gogemma: llama_decode failed with code %d", code)
		}
	}
	c.nPast += total

	return nil
}

func (c *NativeContext) Sample(grammarStr string, temp, topP float32) (int32, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return -1, ErrClosed
	}

	if c.samplerPtr == nil {
		var cGrammar *C.char
		if grammarStr != "" {
			cGrammar = C.CString(grammarStr)
			defer C.free(unsafe.Pointer(cGrammar))
		}
		c.samplerPtr = C.c_llama_init_sampler(c.model.vocabPtr, cGrammar, C.float(temp), C.float(topP))
		if c.samplerPtr == nil {
			return -1, fmt.Errorf("gogemma: failed to initialize sampler chain")
		}
	}

	tok := C.c_llama_sample_token(c.samplerPtr, c.ctxPtr)
	return int32(tok), nil
}

func (c *NativeContext) IsEOG(token int32) bool {
	return bool(C.c_llama_token_is_eog(c.model.vocabPtr, C.int32_t(token)))
}

func (c *NativeContext) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.nPast = 0
	if c.samplerPtr != nil {
		C.c_llama_free_sampler(c.samplerPtr)
		c.samplerPtr = nil
	}
	if c.ctxPtr != nil {
		C.c_llama_ctx_reset(c.ctxPtr)
	}
}

func (c *NativeContext) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	if c.samplerPtr != nil {
		C.c_llama_free_sampler(c.samplerPtr)
		c.samplerPtr = nil
	}
	C.c_llama_batch_free(c.batch)
	if c.ctxPtr != nil {
		C.c_llama_free_ctx(c.ctxPtr)
		c.ctxPtr = nil
	}
	return nil
}
