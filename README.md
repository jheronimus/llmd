# llm

An embedded library that uses two local models to be invoked directly in the code:

laya-multilingual for llm.Decide() calls using ONNX Runtime.
Qwen3 1.7B for llm.Generate() and llm.Extract() calls using llama.cpp.

Install via `go get github.com/jheronimus/llm`.
`llm.Preload(ctx)` is used at startup to download the models locally to ~/.cache/llm.

See [pkg.go.dev/github.com/jheronimus/llm](https://pkg.go.dev/github.com/jheronimus/llm) for complete package documentation and API examples.
