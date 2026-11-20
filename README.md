# Inkling 2 Mini on Apple Silicon (MLX)

MLX-VLM with support for [Thinking Machines' Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), the
looped mixture-of-experts model (26.4B parameters, about 3.4B active per token; the middle 12 of its 24 layers run three
times per token). Text and image input, its DSpark speculative-decoding draft, tool calls, and the released NVFP4 / MXFP8
weights loaded as they are.

This repository is [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) pinned at commit `25ead5d` plus three commits:

1. `mlx_vlm/models/inkling2_loop/`: the looped model (`model_type: inkling2_loop`), built on MLX-VLM's Inkling 2 code and
   adapted from the Inkling2Loop implementation in Hugging Face Transformers; the DSpark draft reads the looped model's
   48 layer passes.
2. `mlx_vlm/utils.py`: mixed ModelOpt checkpoints (NVFP4 experts in layers 0–19, MXFP8 in layers 20–23) are read per tensor
   from `hf_quant_config.json`.
3. `mlx_vlm/tools/parsers/inkling.py`: Inkling's tool-call format, so `mlx_vlm.server` returns OpenAI `tool_calls`.

Published by [NetProjects](https://netprojects.com) (Internet Projects Ltd). Not affiliated with Thinking Machines.

## Install

Python 3.10 or newer on an Apple Silicon Mac:

```bash
pip install "git+https://github.com/internet-projects/inkling2-mlx@v0.1.0" torch torchvision
```

This installs `mlx-vlm` version `0.8.0+inkling2mini.1`. PyTorch and torchvision are only used on the CPU by the image
processor.

## Models

| Model | Size | Notes |
|---|---|---|
| [netprojects/Inkling-2-Mini-MLX-4-8bit](https://huggingface.co/netprojects/Inkling-2-Mini-MLX-4-8bit) | 20.2 GB + 0.64 GB draft | 4-bit NVFP4 + 8-bit MXFP8 experts as released (lossless), recommended |

The original checkpoint also loads directly: `load("thinkingmachines/Inkling-2-Mini")`.

## Use

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("netprojects/Inkling-2-Mini-MLX-4-8bit")
prompt = apply_chat_template(processor, model.config, "Explain in two sentences why the sky is blue.", reasoning_effort="low")
print(generate(model, processor, prompt, max_tokens=2048, temperature=0.0).text)
```

`reasoning_effort` is part of Inkling's chat template: `none`, `minimal`, `low`, `medium`, `high` (default), `xhigh`, `max`,
or a number from 0.0 to 0.99. The model can think at length even at `none`, so always set `max_tokens`.

Images: pass `num_images=1` to `apply_chat_template` and `image=["picture.png"]` to `generate`.

DSpark speculative decoding, with the draft bundled in the model repository:

```python
from mlx_vlm.speculative.drafters import load_drafter
draft, kind = load_drafter("netprojects/Inkling-2-Mini-MLX-4-8bit")
print(generate(model, processor, prompt, max_tokens=2048, draft_model=draft, draft_kind=kind).text)
```

OpenAI-compatible server (tool calls returned as `tool_calls`, thinking as `reasoning_content`):

```bash
mlx_vlm.server --model netprojects/Inkling-2-Mini-MLX-4-8bit --port 8000
```

## Checks

On a MacBook Pro M5 Max (128 GB), 20 November 2026, against Thinking Machines' Transformers implementation (pull request
#51050) running on the CPU with the released weights, 12 short chat texts (488 answer positions):

| | Same next token | Mean KL | Greedy text (3 prompts × 24 tokens) |
|---|---|---|---|
| 4/8-bit release in this MLX-VLM | 98.6% | 0.0018 | identical |

GSM8K, 50 fixed questions (reasoning `none`): 47/50. Tool calls, coding tasks, thinking and image prompts pass. The model
tests (`pytest mlx_vlm/tests/test_inkling2_loop.py`) cover the loop schedule, its caches, speculative rollback and lossless
greedy DSpark decoding.

Known limitation: when the model calls a tool through `mlx_vlm.server`, `content` can also carry a short sentence of its
reasoning and the tool name.

## License

MLX-VLM is under the MIT License (`LICENSE`). The Inkling 2 Mini additions are under the Apache License 2.0
(`LICENSE-APACHE-2.0`); `NOTICE` lists them and credits Hugging Face Transformers, whose Inkling2Loop implementation they
adapt. Inkling 2 Mini itself is Apache-2.0 with Thinking Machines'
[Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/).

MLX-VLM's own README is in [`README-mlx-vlm.md`](README-mlx-vlm.md).
