// Package llm provides an embedded, local-first artificial intelligence runtime
// for Go applications, unifying System 1 (fast zero-shot decision classification)
// and System 2 (grammar-constrained generation and JSON extraction).
//
// The package requires no external server daemons (such as Ollama or Docker).
// Models and dynamic C/C++ runtimes (ONNX Runtime and llama.cpp with Metal/AVX2
// acceleration) are managed automatically within the local cache directory
// (~/.cache/llm).
//
// # Dual-System Architecture
//
// The design separates high-frequency, low-latency categorical decisions from
// slower, expressive autoregressive generation:
//
//   - System 1 (Laya / ModernBERT ONNX): sub-10ms zero-shot classification into
//     strongly-typed Go structs. Supports boolean flags, categorical choices,
//     and bounded numerical scores calibrated for confidence.
//   - System 2 (Qwen3-1.7B / llama.cpp): grammar-constrained JSON extraction and
//     text generation. Uses GBNF grammars derived from Go struct schemas to
//     guarantee syntactically valid JSON output with zero schema hallucinations.
//
// # Struct Tags for System 1
//
// System 1 struct fields are annotated with the "llm" (or "laya") struct tag:
//
//	type Triage struct {
//	    IsSpam   bool   `llm:"bool,prompt:Is this incoming message spam?"`
//	    Category string `llm:"choice,support|billing|sales,prompt:Select the primary category"`
//	    Priority int    `llm:"score,1..5,prompt:Rate urgency from 1 to 5"`
//	}
//
// Supported tag formats:
//   - `bool,prompt:<text>`: Populates a bool field based on yes/no semantic classification.
//   - `choice,<opt1>|<opt2>|...,prompt:<text>`: Populates a string field with the closest option.
//   - `score,<min>..<max>,prompt:<text>`: Populates an int field calibrated between min and max.
//
// # Package-Level vs Instance API
//
// For standard applications, the package-level functions [Decide], [Extract],
// [Generate], and [GenerateStream] operate on a lazily-initialized singleton
// client. Global settings can be adjusted before first use via [Configure].
//
// For applications requiring distinct lifecycles, custom cache directories, or
// independent thread pools, create isolated instances using [New]:
//
//	client, err := llm.New(
//	    llm.WithCacheDir("/var/cache/llm"),
//	    llm.WithThreads(4),
//	    llm.WithContextSize(4096),
//	    llm.WithIdleTimeout(5 * time.Minute),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer client.Close()
//
// # Asset Management and Preloading
//
// On first invocation, missing model weights and native dynamic libraries are
// automatically fetched into the cache directory. In latency-sensitive
// production deployments (such as HTTP servers or worker pools), call [Preload]
// during startup to eliminate cold-start spikes:
//
//	if err := llm.Preload(ctx, llm.AllSystems); err != nil {
//	    log.Fatalf("failed to preload model assets: %v", err)
//	}
package llm
