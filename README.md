# llm

`github.com/jheronimus/llm` is an embedded, local-first artificial intelligence library for Go that unifies System 1 (fast zero-shot decision classification) and System 2 (grammar-constrained generation and JSON extraction) into a single, self-contained binary without external server daemons or Docker dependencies.

Under the hood, System 1 utilizes ModernBERT executed via ONNX Runtime to deliver sub-10ms typed struct classifications (booleans, categorical choices, and ordinal scores), while System 2 runs quantized Qwen3 models over native `llama.cpp` (with Metal and AVX2 hardware acceleration) using GBNF grammars to guarantee mathematically valid JSON extraction. All runtime libraries and model weights are self-managing and automatically cached under `~/.cache/llm/`.

Install via `go get github.com/jheronimus/llm` and immediately call package-level primitives like `llm.Decide[T]` for zero-shot tagging or `llm.Extract[T]` and `llm.Generate` for generative tasks, with production services calling `llm.Preload(ctx)` at startup to eliminate cold-start latency; see [pkg.go.dev/github.com/jheronimus/llm](https://pkg.go.dev/github.com/jheronimus/llm) for complete package documentation and API examples.
