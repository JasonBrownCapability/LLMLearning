"""Dataset loading and prompt formatting for training and evaluation."""

from datasets import load_dataset


PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Show your reasoning, then give your final answer after ####.\n\n"
    "Problem: {question}\n\n"
    "Solution:"
)


def load_gsm8k_train():
    """Load GSM8K training set, formatted for GRPO training."""
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    return dataset.map(_format_gsm8k_example)


def load_gsm8k_test():
    """Load GSM8K test set for evaluation."""
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    return dataset.map(_format_gsm8k_example)


def _format_gsm8k_example(example):
    """Format a GSM8K example into a prompt.

    The GRPOTrainer expects a 'prompt' column containing either a string
    or a list of chat messages.
    """
    example["prompt"] = PROMPT_TEMPLATE.format(question=example["question"])
    # Keep the answer column for the reward function
    return example


def load_gsm8k_sft():
    """Load GSM8K training set formatted for supervised fine-tuning.

    Returns examples with 'text' column containing the full
    prompt + solution for teacher-forced training.
    """
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    def format_sft(example):
        prompt = PROMPT_TEMPLATE.format(question=example["question"])
        example["text"] = prompt + " " + example["answer"]
        return example

    return dataset.map(format_sft)
