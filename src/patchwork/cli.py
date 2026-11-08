import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys

from . import claude
from .benchmark import evaluate, read_cases


QWEN = "Qwen/Qwen3.5-0.8B"
DOWNLOAD_FILES = ["*.json", "*.safetensors", "*.jinja", "*.txt"]


def snapshot(*, revision, local_only):
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=QWEN,
        revision=revision,
        allow_patterns=DOWNLOAD_FILES,
        local_files_only=local_only,
    ))


def patch(args):
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from .weights import Weights

    if not args.prompt.strip() or args.max_new_tokens < 1:
        raise ValueError("Provide a patch goal and a positive token limit.")
    if args.output.exists() or args.output.is_symlink():
        raise ValueError(f"Output already exists: {args.output}")
    cases = read_cases(args.eval)
    checkpoint = snapshot(revision=args.revision, local_only=True)
    if not any(file.is_file() for file in checkpoint.glob("*.safetensors")):
        raise ValueError(f"Checkpoint is incomplete. Run patchwork download --revision {args.revision}.")
    print(f"Loading {QWEN}@{checkpoint.name} on {args.device}…", flush=True)
    with claude.client() as api:
        api.models.retrieve(args.claude_model)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            checkpoint, local_files_only=True, dtype="float32",
        ).to(args.device).eval()
        weights = Weights(model)
        settings = {"max_new_tokens": args.max_new_tokens, "thinking": args.thinking}
        print(f"Benchmarking baseline on {len(cases)} cases…", flush=True)
        before = evaluate(model, tokenizer, cases, **settings)
        print(f"Requesting a weight patch from {args.claude_model}…", flush=True)
        proposal, resolved_model = claude.propose(
            api, args.claude_model, args.prompt, f"{QWEN}@{checkpoint.name}", weights,
        )
    changes = weights.apply(proposal)
    print(f"Benchmarking candidate with {changes['parameters_changed']:,} changed weights…", flush=True)
    after = evaluate(model, tokenizer, cases, **settings)
    report = {
        "base_model": QWEN,
        "base_revision": checkpoint.name,
        "claude_model": resolved_model,
        "goal": args.prompt,
        "patch": proposal,
        **changes,
        "evaluation": {
            **settings,
            "device": args.device,
            "dtype": "float32",
            "decoding": "greedy",
            "torch": version("torch"),
            "transformers": version("transformers"),
        },
        "before": before,
        "after": after,
    }

    args.output.mkdir(parents=True, exist_ok=False)
    try:
        model.save_pretrained(args.output / "checkpoint")
        tokenizer.save_pretrained(args.output / "checkpoint")
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        shutil.rmtree(args.output)
        raise

    for name, result in (("Before", before), ("After", after)):
        print(
            f"{name:6} {result['exact_match']:.1%} exact match "
            f"({result['correct']}/{result['total']}) · "
            f"{result['mean_generation_seconds']:.2f}s per case"
        )
    print(f"Saved {args.output / 'checkpoint'} and {args.output / 'report.json'}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="patchwork", description="Edit and benchmark Qwen3.5-0.8B weights.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("login", help="Verify your Anthropic API key.")
    commands.add_parser("models", help="List Claude model IDs available to your API key.")
    download = commands.add_parser("download", help="Download the Qwen checkpoint.")
    download.add_argument("--revision", default="main", help="Hugging Face branch or commit (default: main).")
    run = commands.add_parser("patch", help="Propose, apply, and benchmark a direct weight edit.")
    run.add_argument("--claude-model", required=True, help="Claude model ID from patchwork models.")
    run.add_argument("--prompt", required=True, help="Describe the desired behavior change.")
    run.add_argument("--eval", required=True, type=Path, help="Held-out prompt/answer JSON file.")
    run.add_argument("--output", required=True, type=Path, help="New directory for the checkpoint and report.")
    run.add_argument("--revision", default="main", help="Previously downloaded Hugging Face branch or commit.")
    run.add_argument("--max-new-tokens", type=int, default=128)
    run.add_argument("--thinking", action="store_true", help="Enable Qwen's reasoning template.")
    run.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    try:
        if args.command in ("login", "models"):
            with claude.client() as api:
                models = api.models.list()
                if args.command == "login":
                    print("Anthropic API key verified.")
                else:
                    for model in models:
                        print(f"{model.id}\t{model.display_name}")
        else:
            try:
                from huggingface_hub.errors import LocalEntryNotFoundError
            except ImportError:
                raise ValueError("Install local model support with uv tool install '.[local]'.") from None
            try:
                if args.command == "download":
                    print(snapshot(revision=args.revision, local_only=False))
                else:
                    patch(args)
            except LocalEntryNotFoundError:
                raise ValueError(f"Download the checkpoint first: patchwork download --revision {args.revision}") from None
    except Exception as exc:
        print(f"patchwork: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
