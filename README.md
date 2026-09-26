# llm

An embedded, local-first Go framework unifying **System 1** (fast zero-shot decision classification) and **System 2** (grammar-constrained generation and JSON extraction).

Zero external server daemons (no Ollama/Docker requirement). Everything runs directly inside your Go binary via native ONNX Runtime and `llama.cpp` AVX2/Metal acceleration.

## Features

- **Package-level API:** Call `llm.Decide`, `llm.Extract`, and `llm.Generate` directly without managing client instances.
- **System 1 (Laya / ONNX):** Sub-10ms zero-shot classification for booleans, categorical choices, and ordinal scores into native Go structs.
- **System 2 (Qwen3 / llama.cpp):** Fast, memory-efficient (~1.05 GB RAM) text generation and GBNF-constrained JSON extraction into typed Go structs.
- **Self-managing Assets:** Automatically downloads dynamic runtime libraries and quantized model weights into `~/.cache/llm/`.
- **Production Preloading:** Explicit `llm.Preload(ctx)` prevents first-call latency spikes in background daemons and HTTP servers.

---

## Installation

```bash
go get github.com/jheronimus/llm
```

---

## Quickstart

### 1. System 1: Fast Zero-Shot Classification (`<10 ms`)

Annotate your Go struct with `llm` tags:

```go
package main

import (
    "context"
    "fmt"
    "github.com/jheronimus/llm"
)

type VacancyDecision struct {
    IsVacancy bool   `llm:"bool,prompt:Является ли данный текст объявлением о вакансии?"`
    WorkMode  string `llm:"choice,remote|hybrid|office|unclear,prompt:Какой формат работы указан?"`
    Urgency   int    `llm:"score,0..5,prompt:Срочность закрытия позиции"`
}

func main() {
    ctx := context.Background()
    ad := "Ищем Senior Go Developer в финтех. Удаленка, вилка 350-450k."

    decision, err := llm.Decide[VacancyDecision](ctx, ad)
    if err != nil {
        panic(err)
    }

    fmt.Printf("Is Vacancy: %v, Work Mode: %s\n", decision.IsVacancy, decision.WorkMode)
}
```

---

### 2. System 2: Structured JSON Extraction (`GBNF`)

Extract structured data from unstructured text with mathematically guaranteed JSON validity:

```go
type RankedRole struct {
    Title     string `json:"title"`
    Score     int    `json:"score"`
    FitReason string `json:"fit_reason"`
}

func main() {
    ctx := context.Background()
    prompt := "Оцени вакансию Go Developer для тимлида и верни JSON: title, score, fit_reason."

    role, err := llm.Extract[RankedRole](ctx, prompt)
    if err != nil {
        panic(err)
    }

    fmt.Printf("Role: %s, Score: %d, Reason: %s\n", role.Title, role.Score, role.FitReason)
}
```

---

### 3. System 2: Text Generation & Streaming

```go
func main() {
    ctx := context.Background()

    // Freeform completion
    resp, err := llm.Generate(ctx, llm.Request{
        Prompt:    "Напиши 3 ключевых навыка для Backend Tech Lead.",
        MaxTokens: 128,
    })
    fmt.Println(resp.Text)

    // Streaming completion
    err = llm.GenerateStream(ctx, req, func(chunk string) bool {
        fmt.Print(chunk)
        return true // return false to cancel stream
    })
}
```

---

### 4. Daemon Preloading & Custom Clients

In background services, pre-warm assets during startup to eliminate runtime latency:

```go
func main() {
    ctx := context.Background()

    // Global configuration
    llm.Configure(
        llm.WithThreads(4),
        llm.WithIdleTimeout(10 * time.Minute),
    )

    // Preload both System 1 and System 2 models into cache
    if err := llm.Preload(ctx, llm.AllSystems); err != nil {
        log.Fatalf("Asset preloading failed: %v", err)
    }

    // Isolated client instance if needed
    customClient, err := llm.New(
        llm.WithCacheDir("/custom/cache"),
        llm.WithContextSize(4096),
    )
    defer customClient.Close()
}
```

---

## Architecture & Models

| Subsystem | Model | Runtime Engine | Active Memory | Primary Use Case |
|---|---|---|---|---|
| **System 1** | **Laya** (ModernBERT) | ONNX Runtime (`libonnxruntime`) | ~60 MB | Fast, zero-shot struct decision classification |
| **System 2** | **Qwen3-1.7B** (Q4_K_M) | `llama.cpp` (`libllama`) | ~1.05 GB | Structured JSON extraction, reasoning, ranking |

## License

MIT
