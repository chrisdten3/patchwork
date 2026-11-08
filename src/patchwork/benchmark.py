import json
from time import perf_counter


def read_cases(path):
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation file must contain a nonempty JSON array.")
    for i, case in enumerate(cases, 1):
        if not isinstance(case, dict) or any(
            not isinstance(case.get(key), str) or not case[key].strip()
            for key in ("prompt", "answer")
        ):
            raise ValueError(f"Evaluation case {i} needs nonempty prompt and answer strings.")
    return cases


def normalize(text):
    return " ".join(text.casefold().split())


def evaluate(model, tokenizer, cases, *, max_new_tokens, thinking):
    import torch
    from transformers import GenerationConfig

    original_config = model.generation_config
    generation = GenerationConfig(
        eos_token_id=original_config.eos_token_id,
        pad_token_id=original_config.pad_token_id,
        bos_token_id=original_config.bos_token_id,
        do_sample=False,
        use_cache=True,
    )

    def generate(prompt, limit):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            start = perf_counter()
            generated = model.generate(
                **inputs, generation_config=generation, max_new_tokens=limit,
            )
            tokens = generated[0, inputs.input_ids.shape[1]:].tolist()
            seconds = perf_counter() - start
        token_count = len(tokens)
        raw = tokenizer.decode(tokens, skip_special_tokens=False)
        think_end = tokenizer.convert_tokens_to_ids("</think>")
        if think_end in tokens:
            tokens = tokens[tokens.index(think_end) + 1:]
        elif thinking:
            tokens = []
        return raw, tokenizer.decode(tokens, skip_special_tokens=True), seconds, token_count

    generate("Reply OK.", 4)
    results = []
    for case in cases:
        raw, answer, seconds, tokens = generate(case["prompt"], max_new_tokens)
        results.append({
            "prompt": case["prompt"],
            "expected": case["answer"],
            "raw_output": raw,
            "answer": answer,
            "correct": normalize(answer) == normalize(case["answer"]),
            "generation_seconds": seconds,
            "generated_tokens": tokens,
            "reached_token_limit": tokens == max_new_tokens,
        })
    seconds = sum(row["generation_seconds"] for row in results)
    tokens = sum(row["generated_tokens"] for row in results)
    correct = sum(row["correct"] for row in results)
    return {
        "exact_match": correct / len(results),
        "correct": correct,
        "total": len(results),
        "mean_generation_seconds": seconds / len(results),
        "generated_tokens_per_second": tokens / seconds,
        "reached_token_limit": sum(row["reached_token_limit"] for row in results),
        "cases": results,
    }
