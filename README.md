# Patchwork

Patchwork is an experimental CLI for editing the weights of Qwen3.5-0.8B from a plain-language goal. Claude inspects a small set of local tensors and proposes a direct edit. Patchwork measures the original and edited model on the same prompts, then saves the candidate checkpoint and the results.

The approach draws on [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). The authors' math experiment used a larger set of analysis and practice tools. Patchwork starts with bounded edits to columns of Qwen's MLP down-projection matrices.

## Install

With Python 3.11–3.14 and [uv](https://docs.astral.sh/uv/), clone the repository and install the CLI.

```sh
gh repo clone chrisdten3/patchwork
cd patchwork
uv tool install '.[local]'
patchwork --help
```

The first model download and each saved checkpoint require a few GB of disk space. The default device is CPU. Pass `--device mps` on supported Apple Silicon hardware or `--device cuda` with a CUDA-enabled PyTorch install.

## Run your first patch

1. Set `ANTHROPIC_API_KEY` in your shell using a key from the [Claude Console](https://platform.claude.com/). Patchwork reads the key from the environment. Claude requests use [API billing](https://platform.claude.com/docs/en/about-claude/pricing). Anthropic restricts third-party Claude.ai subscription login. [Authentication policy](https://code.claude.com/docs/en/legal-and-compliance)
2. Verify the key and copy a Claude model ID from `patchwork models`.

   ```sh
   patchwork login
   patchwork models
   ```

3. Download [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), then run a patch. Replace `MODEL_ID` with an ID from step 2.

   ```sh
   patchwork download
   patchwork patch \
     --claude-model MODEL_ID \
     --prompt "Reduce repetitive reasoning loops on short arithmetic problems." \
     --eval examples/math-smoke.json \
     --thinking --max-new-tokens 256 \
     --output runs/math
   ```

   The eval file is a JSON array of objects with `prompt` and `answer` strings. Include both target and regression cases. The bundled set is a small smoke test. Use an independent held-out set for meaningful comparisons and ask for short final answers that can be scored by exact match.

Claude receives the goal, checkpoint revision, tensor names and shapes, and up to six requested summaries of local weight slices. Evaluation prompts and outputs stay local. Claude can propose column scaling or removing the direction of a reference column from a contiguous set of columns. Patchwork validates the proposed edits and caps the number of touched parameters. The workflow uses direct edits without gradients or training.

Each run writes a Transformers checkpoint under `checkpoint/` and a `report.json` with the source revision, Claude model, patch operations, per-case outputs, and before-and-after exact-match scores, generation time, and token counts. Patchwork saves a valid candidate even if its score drops. Compare runs using identical device, token limit, thinking mode, and evaluation data.

## Run from source

```sh
uv run --extra local patchwork --help
```
