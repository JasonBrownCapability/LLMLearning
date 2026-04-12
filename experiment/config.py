"""Centralised configuration for all experiment conditions."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str = "meta-llama/Meta-Llama-3-8B"
    quantize_4bit: bool = True
    dtype: str = "float16"
    max_seq_length: int = 1024
    trust_remote_code: bool = True


@dataclass
class InsertedLayerConfig:
    num_layers: int = 2
    positions: list[int] = field(default_factory=lambda: [10, 21])
    init_strategy: str = "zero"  # "zero", "small_random", "copy"
    small_random_std: float = 0.01


@dataclass
class LoRAConfig:
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "gate_proj", "down_proj",
    ])


@dataclass
class GRPOConfig:
    num_rollouts: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    max_steps: int = 2000
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    gradient_checkpointing: bool = True
    kl_coef: float = 0.01  # Set to 0.0 to skip loading reference model (saves memory)
    temperature: float = 0.7
    max_completion_length: int = 512
    logging_steps: int = 10
    save_steps: int = 500


@dataclass
class SFTConfig:
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    max_steps: int = 2000
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 500


@dataclass
class TwoStageConfig:
    # Stage 1: LoRA + RL
    lora_rank: int = 32
    lora_alpha: int = 64
    stage1_max_steps: int = 1000
    # Rollout generation
    num_rollouts_to_generate: int = 50000
    rollout_batch_size: int = 32
    # Stage 2: Inserted layers + RL
    stage2_max_steps: int = 1000


@dataclass
class EvalConfig:
    benchmarks: list[str] = field(default_factory=lambda: [
        "gsm8k", "math", "arc_challenge", "logiqa", "humaneval",
    ])
    num_samples_pass_at_8: int = 8
    eval_batch_size: int = 8


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    inserted_layers: InsertedLayerConfig = field(default_factory=InsertedLayerConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    two_stage: TwoStageConfig = field(default_factory=TwoStageConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output_dir: str = "./results"
    wandb_project: str = "rl-inserted-layers"
    seed: int = 42
    use_wandb: bool = True
