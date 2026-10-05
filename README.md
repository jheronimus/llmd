# llmd

High-performance, local-first dual-system LLM daemon and Go client.

- **System 1**: Zero-shot cross-encoder NLI classification (<15ms) using ModernBERT ONNX.
- **System 2**: Autoregressive structured extraction & generation using Qwen3 0.6B via `llama.cpp`.
- **Transparent Auto-Escalation**: Ambiguous System 1 predictions automatically escalate to System 2.
- **OpenAI Compatible**: Exposes `/v1/chat/completions` as a drop-in replacement for OpenAI tools.
- **Pure Go Client**: Zero CGO, high-level client library for Go callers in the root package.
- **Resource Efficient**: Serialized inference lock keeps peak RAM bounded under 1.1 GB.

## Usage (Go Client)

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/jheronimus/llmd"
)

func main() {
    client := llmd.NewClient(llmd.WithBaseURL("http://localhost:8080"))
    defer client.Close()

    // System 1: Fast NLI Triage
    answers, err := client.Decide(context.Background(), "Looking for Senior Go dev", map[string]llmd.DecideQuestion{
        "is_job": {
            Type: "choice",
            Instructions: "Is this a job posting?",
        },
    })
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Is Job: %v (confidence: %f, escalated: %v)\n", 
        answers["is_job"].Answer, answers["is_job"].Confidence, answers["is_job"].Escalated)

    // System 2: Structured JSON Generation
    resp, err := client.Generate(context.Background(), llmd.Request{
        Prompt: "Extract role from: Senior Go dev, $5000",
        JSONSchema: `{"type":"object","properties":{"role":{"type":"string"}},"required":["role"]}`,
    })
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(resp.Text)
}
```

## Running the Daemon

```sh
# Local build
task build
./target/release/llmd --port 8080

# Or via Docker
task docker:build
docker run -p 8080:8080 -v ~/.cache/llmd/models:/root/.cache/llmd/models llmd:latest
```
