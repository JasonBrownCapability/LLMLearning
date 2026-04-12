"""Reward functions for RL training on reasoning benchmarks."""

import re


def extract_gsm8k_answer(text: str) -> float | None:
    """Extract the final numerical answer from a GSM8K-style response.

    Looks for "#### <number>" format first, then falls back to the last
    number in the text.
    """
    # Try the standard GSM8K format: #### <number>
    match = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", text)
    if match:
        return _parse_number(match.group(1))

    # Fallback: last number in the text
    numbers = re.findall(r"[+-]?\d[\d,]*\.?\d*", text)
    if numbers:
        return _parse_number(numbers[-1])

    return None


def extract_math_answer(text: str) -> str | None:
    r"""Extract the answer from a MATH-style response.

    Looks for \boxed{...} format first, then falls back to last expression
    after "answer is" or similar.
    """
    # Try \boxed{...} format
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()

    # Try "the answer is ..." format
    match = re.search(r"(?:the answer is|therefore|thus)[:\s]+(.+?)(?:\.|$)",
                      text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def _parse_number(s: str) -> float:
    """Parse a number string, handling commas."""
    return float(s.replace(",", ""))


def _has_reasoning_steps(text: str) -> bool:
    """Check if the response shows intermediate reasoning steps."""
    indicators = [
        r"\bstep\s+\d",
        r"\bfirst\b.*\bthen\b",
        r"\bso\b",
        r"\btherefore\b",
        r"\bwe\s+(need|know|can|get|have)\b",
        r"\blet'?s\b",
        r"\n",  # Multi-line responses likely show work
    ]
    score = sum(1 for pattern in indicators if re.search(pattern, text, re.IGNORECASE))
    return score >= 2


def gsm8k_reward_fn(completions: list[list[dict]], answer: list[str], **kwargs) -> list[float]:
    """Reward function for GSM8K problems.

    Args:
        completions: List of completions in chat format. Each completion is a
            list of message dicts with "role" and "content" keys. The generated
            text is in completions[i][-1]["content"].
        answer: List of ground truth answer strings. This parameter name must
            match the dataset column name ("answer" in GSM8K). The GRPOTrainer
            passes extra dataset columns as keyword arguments matched by name.

    Returns:
        List of float rewards, one per completion.
    """
    rewards = []
    for completion, truth in zip(completions, answer):
        # Extract text from the chat-format completion
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        # Extract ground truth number
        true_answer = extract_gsm8k_answer(truth)

        # Extract predicted answer
        predicted = extract_gsm8k_answer(text)

        # Score
        reward = 0.0
        if predicted is not None and true_answer is not None:
            if abs(predicted - true_answer) < 1e-3:
                reward = 1.0

        # Format bonus for showing work
        if _has_reasoning_steps(text):
            reward += 0.1

        rewards.append(reward)

    return rewards
