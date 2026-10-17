# Model chat templates — the distill vs Gemma

Reference for the two local models the harness runs. These are the **chat templates** — how
`src/harness/backends/mlx_backend.py::_render` turns a `[{role, content}, …]` message list into the
raw prompt string (via the tokenizer's `apply_chat_template`, with `enable_thinking` set from the
chat's `think=full|brief|off`). Captured by rendering each model's real tokenizer — not from memory.

Why this matters: the two models delimit turns and reasoning with **different tokens**, so anything
that parses the model's output (above all `_strip_think`) must handle both. Swapping models
(`HARNESS_MLX_REPO=…`) must not need any other code change.

---

## Qwen3.8-2B-Distill — `qwen38-2b-distill-4bit` (the default)

**ChatML**, Qwen-style `<think>` reasoning. One system + one user turn renders as:

```text
<|im_start|>system
You are a Python coding assistant.<|im_end|>
<|im_start|>user
add two numbers<|im_end|>
<|im_start|>assistant
<think>
```

The model then writes its reasoning, closes with `</think>`, writes the answer, and ends the turn
with `<|im_end|>`.

**Thinking modes** (the tail of the prompt):
- `enable_thinking=True`  → `…<|im_start|>assistant\n<think>\n`      (reasons, then answers)
- `enable_thinking=False` → `…<|im_start|>assistant\n<think>\n\n</think>\n\n`  (empty think → straight to answer)

| | |
|---|---|
| Turn format | `<|im_start|>{role}\n{content}<|im_end|>\n` |
| Roles | `system` / `user` / `assistant` |
| Reasoning | `<think>…</think>` |
| Vision (it is a VLM) | `<|vision_start|><|image_pad|><|vision_end|>` per image |
| Tools | `<tools>…</tools>` + `<tool_call><function=name><parameter=…>…</parameter></function></tool_call>` |
| Special | **no BOS** · EOS `<|im_end|>` · PAD `<|endoftext|>` |

Full Jinja source: `models/qwen38-2b-distill-4bit/chat_template.jinja`.

---

## Gemma 4 E2B — `mlx-community/gemma-4-e2b-it-4bit` (swap-in)

Gemma's own `<|turn>…<turn|>` delimiters, role **`model`** (not `assistant`), and a `<|think|>`
reasoning opener. Same two-message turn renders as:

```text
<bos><|turn>system
<|think|>
You are a Python coding assistant.<turn|>
<|turn>user
add two numbers<turn|>
<|turn>model
```

**Thinking modes:**
- `enable_thinking=True`  → system turn carries `<|think|>`; prompt ends `…<|turn>model\n`
- `enable_thinking=False` → no `<|think|>`; prompt ends `…<|turn>model\n`

| | |
|---|---|
| Turn format | `<|turn>{role}\n{content}<turn|>\n` |
| Roles | `system` / `user` / **`model`** |
| Reasoning | prompt opener `<|think|>`; the model **emits** `<|channel>thought … <channel|>` |
| Special | BOS `<bos>` · EOS `<eos>` · PAD `<pad>` |

---

## What the harness does with this

- `_render(messages, enable_thinking)` calls `apply_chat_template(..., enable_thinking=…)` and falls
  back to a call without the kwarg for tokenizers that don't accept it.
- `_strip_think` removes **both** reasoning forms before anything downstream sees the text:
  `<think>…</think>` (Qwen) **and** `<|channel>thought…<channel|>` (Gemma). That one helper is why a
  model swap needs no other change — see `src/harness/lab.py`.
- Regenerate this doc's renderings any time with:

  ```bash
  .venv-315/bin/python -c "from transformers import AutoTokenizer as T; \
    m=[{'role':'system','content':'You are a Python coding assistant.'},{'role':'user','content':'add two numbers'}]; \
    print(repr(T.from_pretrained('models/qwen38-2b-distill-4bit').apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True)))"
  ```
