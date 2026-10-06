# TODO

- [x] **JSON Schema to GBNF Compilation & Enforcement**
  - Accept standard JSON Schema in API / client requests (`json_schema` field).
  - Automatically compile JSON Schema into a GBNF grammar string before passing to `llama.cpp` (`LlamaSampler::grammar`).
  - Wire `json_schema` through client envelopes (`client.go`) to daemon endpoints (`src/server.rs` & `src/sys2.rs`).
  - Enforce hard validation on output before returning to client (reject or repair invalid completions).

- [x] **Switch Sys2 Default Model to Qwen 3.5 0.8B (GGUF)**
  - Update model fetch / download logic to pull `Qwen3.5-0.8B` (Q4_K_M GGUF).
  - Ensure ChatML prompt template closes thinking tags (`<think>\n</think>`) to bypass chain-of-thought overhead.
  - Verify RAM footprint remains under 1GB and benchmark inference speed on target N100/RPi hardware.

