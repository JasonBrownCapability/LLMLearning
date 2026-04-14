"""Evaluation across reasoning benchmarks."""

import json
import os
from pathlib import Path

import torch
from transformers import GenerationConfig

from .data import load_gsm8k_test, load_math_test, PROMPT_TEMPLATE
from .rewards import extract_gsm8k_answer, extract_math_answer


def evaluate_gsm8k(model, tokenizer, max_samples=None, num_samples_pass_at_k=1,
                   batch_size=8, max_new_tokens=512):
    """Evaluate the model on GSM8K test set.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer.
        max_samples: Limit evaluation to this many samples (None = all).
        num_samples_pass_at_k: Number of samples for pass@k metric.
        batch_size: Batch size for generation.
        max_new_tokens: Maximum tokens to generate per problem.

    Returns:
        Dictionary with accuracy metrics and per-example results.
    """
    dataset = load_gsm8k_test()
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    model.eval()
    results = []
    correct_at_1 = 0
    correct_at_k = 0
    total = 0

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=num_samples_pass_at_k > 1,
        temperature=0.7 if num_samples_pass_at_k > 1 else 0.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Process in batches
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        prompts = batch["prompt"]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            # Greedy decode for pass@1
            greedy_outputs = model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                ),
            )

            greedy_texts = tokenizer.batch_decode(
                greedy_outputs[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            # Sampled decodes for pass@k (if k > 1)
            sampled_correct = [False] * len(prompts)
            if num_samples_pass_at_k > 1:
                for _ in range(num_samples_pass_at_k):
                    sample_outputs = model.generate(
                        **inputs,
                        generation_config=gen_config,
                    )
                    sample_texts = tokenizer.batch_decode(
                        sample_outputs[:, inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    for j, text in enumerate(sample_texts):
                        pred = extract_gsm8k_answer(text)
                        true = extract_gsm8k_answer(batch["answer"][j])
                        if pred is not None and true is not None:
                            if abs(pred - true) < 1e-3:
                                sampled_correct[j] = True

        # Score greedy results
        for j, text in enumerate(greedy_texts):
            predicted = extract_gsm8k_answer(text)
            true_answer = extract_gsm8k_answer(batch["answer"][j])

            is_correct = (predicted is not None and true_answer is not None
                          and abs(predicted - true_answer) < 1e-3)

            if is_correct:
                correct_at_1 += 1
            if is_correct or sampled_correct[j]:
                correct_at_k += 1
            total += 1

            results.append({
                "question": batch["question"][j],
                "true_answer": true_answer,
                "predicted_answer": predicted,
                "correct": is_correct,
                "generated_text": text[:2048],  # Truncate for storage
            })

        if (i // batch_size) % 10 == 0:
            print(f"  Evaluated {min(i + batch_size, len(dataset))}/{len(dataset)} "
                  f"(acc so far: {correct_at_1}/{total} = {correct_at_1/max(total,1):.3f})")

    accuracy_at_1 = correct_at_1 / max(total, 1)
    accuracy_at_k = correct_at_k / max(total, 1)

    return {
        "benchmark": "gsm8k",
        "total": total,
        "correct_at_1": correct_at_1,
        "accuracy_at_1": accuracy_at_1,
        "correct_at_k": correct_at_k,
        "accuracy_at_k": accuracy_at_k,
        "k": num_samples_pass_at_k,
        "examples": results,
    }


def _normalize_math(answer: str) -> str:
    """Normalize a math answer string for comparison."""
    s = answer.strip().lower()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\frac", "frac").replace("\\sqrt", "sqrt")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace(" ", "")
    try:
        return str(float(s))
    except ValueError:
        return s


def evaluate_math(model, tokenizer, max_samples=None, batch_size=8,
                  max_new_tokens=1024):
    """Evaluate the model on MATH test set.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer.
        max_samples: Limit evaluation to this many samples (None = all).
        batch_size: Batch size for generation.
        max_new_tokens: Maximum tokens to generate per problem.

    Returns:
        Dictionary with accuracy metrics and per-example results.
    """
    dataset = load_math_test()
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    model.eval()
    results = []
    correct = 0
    total = 0

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        prompts = batch["prompt"]

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                ),
            )
            texts = tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

        for j, text in enumerate(texts):
            predicted = extract_math_answer(text)
            true_answer = extract_math_answer(batch["solution"][j])

            is_correct = (predicted is not None and true_answer is not None
                          and _normalize_math(predicted) == _normalize_math(true_answer))

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "problem": batch["problem"][j],
                "true_answer": true_answer,
                "predicted_answer": predicted,
                "correct": is_correct,
                "generated_text": text[:2048],
            })

        if (i // batch_size) % 10 == 0:
            print(f"  MATH: Evaluated {min(i + batch_size, len(dataset))}/{len(dataset)} "
                  f"(acc so far: {correct}/{total} = {correct/max(total,1):.3f})")

    accuracy = correct / max(total, 1)
    return {
        "benchmark": "math",
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "examples": results,
    }


def run_full_evaluation(model, tokenizer, output_dir, condition_name,
                        num_samples_pass_at_k=8, max_samples=None):
    """Run evaluation on all benchmarks and save results."""
    output_path = Path(output_dir) / condition_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Evaluating condition: {condition_name}")
    print(f"{'='*60}")

    # GSM8K evaluation
    print("\nRunning GSM8K evaluation...")
    gsm8k_results = evaluate_gsm8k(
        model, tokenizer,
        max_samples=max_samples,
        num_samples_pass_at_k=num_samples_pass_at_k,
    )
    print(f"GSM8K pass@1: {gsm8k_results['accuracy_at_1']:.4f}")
    print(f"GSM8K pass@{num_samples_pass_at_k}: {gsm8k_results['accuracy_at_k']:.4f}")

    # Save GSM8K results
    results_file = output_path / "gsm8k_results.json"
    with open(results_file, "w") as f:
        json.dump(gsm8k_results, f, indent=2, default=str)
    print(f"Results saved to {results_file}")

    # MATH evaluation
    print("\nRunning MATH evaluation...")
    math_results = evaluate_math(
        model, tokenizer,
        max_samples=max_samples,
    )
    print(f"MATH accuracy: {math_results['accuracy']:.4f}")

    math_results_file = output_path / "math_results.json"
    with open(math_results_file, "w") as f:
        json.dump(math_results, f, indent=2, default=str)
    print(f"Results saved to {math_results_file}")

    # Summary
    summary = {
        "condition": condition_name,
        "gsm8k_pass_at_1": gsm8k_results["accuracy_at_1"],
        "gsm8k_pass_at_k": gsm8k_results["accuracy_at_k"],
        "math_accuracy": math_results["accuracy"],
    }

    summary_file = output_path / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary
