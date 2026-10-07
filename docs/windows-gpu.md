# Windows + GPU: performance, setup, and the vLLM/Docker route

This note records how the harness runs on an NVIDIA GPU on Windows, the measured
performance, why this particular model is the way it is, and how to set up the
Docker + GPU path for faster inference engines (vLLM). Verified on a Lenovo Legion
(RTX 4060 Laptop, 8 GB) running Windows 11 Enterprise, driver 610.88.

---

## TL;DR

- The installer **auto-detects an NVIDIA GPU** and installs the CUDA build of PyTorch,
  so the model runs on the GPU in float16. No manual step.
- On the RTX 4060 the GPU is **~5× faster than CPU** (28.5 vs 5.5 tok/s). It is still
  only ~28 tok/s because this model's linear-attention layers fall back to reference
  kernels on Windows (see below).
- A faster engine (vLLM) means running under **WSL2 / Docker** — vLLM has no native
  Windows support. The Docker + GPU path is set up and verified here.

---

## How inference works today

Plain **Hugging Face Transformers, eager mode** — no serving engine
([`src/harness/backends/torch_backend.py`](../src/harness/backends/torch_backend.py)):

- `from_pretrained(...)` loads the safetensors into VRAM (float16 on CUDA, float32 on CPU).
- Generation is `model.generate()` with a `TextIteratorStreamer`, **batch size 1, one
  token at a time** — standard autoregressive decoding.
- No paged attention, no continuous batching, no custom kernels, no quantization.

This is deliberate: the repo exists to *teach* the harness loop, so it uses the simplest
possible generate call.

## Which model, and why it's not blazing fast

The repo pins **`empero-ai/Qwen3.8-2B-Distill`** (`model.py`). Its config reports:

- `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: "qwen3_5"` — a
  **Qwen3.5 / Qwen3-Next-style hybrid**, and a **vision-language model** (`vision_config`).
- `layer_types`: 3× `linear_attention` then 1× `full_attention`, repeated over 24 layers
  (`full_attention_interval: 4`). The linear layers are **Gated DeltaNet + causal conv**
  (`linear_conv_kernel_dim: 4`, `mamba_ssm_dtype: "float32"`).

Those linear-attention layers need special kernels — `causal_conv1d` and
`flash-linear-attention`. They are **not installed** (effectively Linux-only; no prebuilt
Windows wheels), so Transformers runs a **sequential float32 recurrence in reference
PyTorch**. On the GPU each per-token op is tiny and dominated by launch/host-sync
overhead, so the GPU sits ~30 W / partially idle rather than saturated. That is the
ceiling on Windows for this architecture, not a setup mistake.

## Measured performance (A/B)

Same prompt, 120 new tokens, greedy decoding, `HARNESS_THREADS=4`:

| Backend         | Precision | tok/s | Wall  |
|-----------------|-----------|-------|-------|
| GPU (RTX 4060)  | fp16      | 28.5  | 4.2 s |
| CPU             | fp32      | 5.5   | 21.6 s |

**GPU wins ~5.2×.** Reproduce by hiding CUDA for the CPU run:

```bash
# GPU
./.venv-notebook/Scripts/python.exe -c "import sys;sys.path.insert(0,'src');from harness import load_model;m=load_model();print(m.memory()['device']);m.generate([{'role':'user','content':'hi'}],max_tokens=8);print(m.generate([{'role':'user','content':'Write five paragraphs about sailing ships.'}],max_tokens=120) and m.last_stats)"
# CPU: prefix with  CUDA_VISIBLE_DEVICES=-1
```

### Thread count matters — even on the GPU
This model's reference kernels dispatch **per-token CPU work**, so thread contention
hurts. `HARNESS_THREADS` defaults to **4** and is applied on the CUDA path too (not just
CPU). Raising it to all cores made generation *slower*, not faster. Leave it at 4 unless
you measure otherwise.

---

## Faster inference: the vLLM + Docker + GPU route

vLLM ships the exact kernels we're missing — it integrates the **Flash Linear Attention
Triton kernels** and a hybrid KV-cache manager for Qwen3-Next's Gated DeltaNet + full
attention. So vLLM is the real fix for the slowness. Caveats:

- **No native Windows vLLM.** Paths are WSL2 (recommended), Docker Model Runner with the
  WSL2 backend (Dec 2025), or the community `SystemPanic/vllm-windows` fork.
- **Architecture match is unconfirmed.** vLLM supports Qwen3-**Next** (text MoE). This
  model is `qwen3_5` **with vision** — the gated-delta *family* is supported, but this
  exact VLM `model_type` may not be wired up yet. Load-test before building on it.
- It replaces the teaching loop with a black-box server — fine for speed, counter to the
  repo's purpose.

### Enabling WSL2 (Windows 11)

The quick path, in an **Administrator** terminal, then reboot:

```powershell
wsl --install
```

If `wsl` fails with **`0x80070422`** ("service cannot be started, either because it is
disabled…"), the `WSLService` is **Disabled**. Check the services:

```powershell
Get-Service WSLService,vmcompute,HvHost | Select Name,Status,StartType
```

Re-enable and start it (Administrator), then retry `wsl --install`:

```powershell
Set-Service WSLService -StartupType Manual
Start-Service WSLService
```

A one-liner that self-elevates (UAC) from a normal shell and does all of it:

```powershell
Start-Process powershell -Verb RunAs -ArgumentList '-NoExit','-Command','Set-Service WSLService -StartupType Manual; Start-Service WSLService; wsl --install'
```

If `WSLService` stays Disabled after this, Group Policy is pinning it (common on
Enterprise). Check `reg query "HKLM\SYSTEM\CurrentControlSet\Services\WSLService" /v Start`
— `4` = Disabled by policy, `3` = Manual.

Verify after reboot:

```powershell
wsl --status          # Default Version: 2
wsl -l -v             # a distro at VERSION 2
wsl -d Ubuntu -- uname -a
```

### Docker Desktop + GPU passthrough

1. Launch **Docker Desktop**. Settings → General → **Use the WSL 2 based engine** (on).
2. Confirm the daemon and the NVIDIA runtime:

```bash
docker info --format 'Server {{.ServerVersion}} | OSType={{.OSType}}'
docker info | grep -i runtimes        # expect: ... nvidia runc
```

3. Prove the GPU reaches a container:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

Expected (verified on this machine): the container prints
`NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 610.88`.

> Prerequisite: virtualization (Intel VT-x / VT-d) must be enabled in BIOS — WSL2 and
> Docker both require it.

### Next step (not yet done)
With Docker + GPU verified, a vLLM container can be launched with `--gpus all`. **First**
do a load-test to confirm vLLM accepts the `qwen3_5` vision model before building a
workflow around it — this is the open question above.
