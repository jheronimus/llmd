# llm

Embed two local LLMs directly into Go applications without Cgo:

- **System 1 (`llm.Decide`)**: Fast zero-shot classification & scoring (<10ms) powered by ModernBERT / Laya multilingual.
- **System 2 (`llm.Extract`, `llm.Generate`)**: Constrained JSON extraction and reasoning powered by Qwen3 1.7B with Apple Metal / AVX GPU acceleration.

## Architecture

```
+-------------------------------------------------------------+
|                     Your Go Application                     |
|                                                             |
|   dec, _ := llm.Decide[Triage](ctx, text)                   |
|   item, _ := llm.Extract[JobSpec](ctx, prompt)              |
+------------------------------+------------------------------+
                               | Pure Go (CGO_ENABLED=0)
                               | HTTP over Unix Domain Socket (/tmp/llm.sock)
+------------------------------v------------------------------+
|                   llmd (Rust Sidecar Daemon)                |
|                                                             |
|   • System 1: ONNX Runtime (ModernBERT / Laya)              |
|   • System 2: llama.cpp (Qwen3 1.7B + Metal / NEON / AVX)   |
|   • GBNF Grammar Constrained Decoding                       |
|   • Auto-spawned on demand, auto-shutdown when idle         |
+-------------------------------------------------------------+
```

- **Zero Cgo:** `github.com/jheronimus/llm` is pure Go standard library (`net/http`, `net/dial`). Compiles in milliseconds, cross-compiles without a C toolchain.
- **Single Native Daemon:** The `llmd` Rust sidecar runs as a local background daemon. It auto-spawns on the first call, serves requests over `/tmp/llm.sock` with sub-millisecond IPC overhead, and unloads after 10 minutes of inactivity.
- **Full Hardware Acceleration:** Metal GPU acceleration on macOS, NEON on ARM64, and AVX2 on x86_64.

## Quick Start

Install the Go package:

```bash
go get github.com/jheronimus/llm
```

### System 1: Fast Zero-Shot Classification (`llm.Decide`)

Define struct tags to express categorical choices, numerical rating scales, or booleans:

```go
type SpamTriage struct {
    IsSpam   bool   `llm:"bool,prompt:Is this email spam or phishing?"`
    Category string `llm:"choice,support|billing|sales,prompt:Classify intent"`
    Priority int    `llm:"score,1..5,prompt:Rate urgency from 1 to 5"`
}

decision, err := llm.Decide[SpamTriage](ctx, emailBody)
if err != nil {
    log.Fatal(err)
}
if decision.IsSpam {
    // Drop spam in <10ms
}
```

### System 2: Structured JSON Extraction (`llm.Extract`)

Constrain output to Go types using token-level GBNF grammar:

```go
type Candidate struct {
    Name       string   `json:"name"`
    Experience int      `json:"experience_years"`
    Skills     []string `json:"skills"`
}

candidate, err := llm.Extract[Candidate](ctx, resumeText)
```

### System 2: Freeform Generation (`llm.Generate`)

```go
resp, err := llm.Generate(ctx, llm.Request{
    Prompt:    "Summarize this incident in 2 sentences.",
    MaxTokens: 120,
})
```

## Daemon Binary (`llmd`)

The Go client locates `llmd` in:
1. Custom path specified via `llm.WithDaemonBinary(path)` or `$LLMD_PATH`
2. Local development build (`daemon/target/release/llmd`)
3. `~/.cache/llm/bin/llmd`
4. System `$PATH`
5. GitHub Releases (auto-downloaded on first run when `WithAutoDownload(true)`)

To build `llmd` manually:

```bash
cd daemon
cargo build --release
# Binary will be placed in daemon/target/release/llmd
```
