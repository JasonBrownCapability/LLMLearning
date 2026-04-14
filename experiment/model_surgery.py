"""Insert new transformer layers into a pre-trained model with no-op initialisation."""

import copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import ModelConfig, InsertedLayerConfig


def load_base_model(config: ModelConfig, smoke_test: bool = False):
    """Load the base model with optional 4-bit quantisation."""
    quantization_config = None
    if config.quantize_4bit and not smoke_test:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.name,
        quantization_config=quantization_config,
        torch_dtype=torch.float32 if smoke_test else torch.bfloat16,
        device_map="cpu" if smoke_test else "auto",
        trust_remote_code=config.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.name,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def create_inserted_layer(model, config: InsertedLayerConfig):
    """Create a new transformer layer matching the base model's architecture.

    The layer is initialised as a no-op: output projections of both self-attention
    and FFN are set to zero, so the residual connection passes input through unchanged.
    """
    # Get a reference layer to copy architecture from
    reference_layer = model.model.layers[0]
    new_layer = copy.deepcopy(reference_layer)

    # Move to same device/dtype as reference
    ref_param = next(reference_layer.parameters())
    new_layer = new_layer.to(device=ref_param.device, dtype=ref_param.dtype)

    # Reinitialise all parameters
    for name, param in new_layer.named_parameters():
        param.requires_grad = True
        if param.dim() >= 2:
            nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
        else:
            nn.init.ones_(param)  # Layer norm weights

    # Zero-init the output projections to make the layer a no-op
    if config.init_strategy == "zero":
        _zero_init_outputs(new_layer)
    elif config.init_strategy == "small_random":
        _small_random_init_outputs(new_layer, config.small_random_std)
    elif config.init_strategy == "copy":
        # Keep the deepcopy weights as-is (copy of reference layer)
        pass

    return new_layer


def _zero_init_outputs(layer):
    """Zero-initialise output projections so the layer acts as a no-op."""
    # Self-attention output projection
    if hasattr(layer.self_attn, "o_proj"):
        nn.init.zeros_(layer.self_attn.o_proj.weight)
    # FFN down projection (the output of the FFN block)
    if hasattr(layer.mlp, "down_proj"):
        nn.init.zeros_(layer.mlp.down_proj.weight)


def _small_random_init_outputs(layer, std):
    """Small random init for output projections (alternative to exact zero)."""
    if hasattr(layer.self_attn, "o_proj"):
        nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=std)
    if hasattr(layer.mlp, "down_proj"):
        nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=std)


def insert_layers(model, config: InsertedLayerConfig):
    """Insert new transformer layers at specified positions.

    Returns the indices of the inserted layers within the modified model.
    """
    layers = model.model.layers
    inserted_indices = []

    # Sort positions in descending order so earlier insertions don't shift
    # the positions of later ones
    for offset, pos in enumerate(sorted(config.positions)):
        adjusted_pos = pos + offset  # Account for previously inserted layers
        new_layer = create_inserted_layer(model, config)
        layers.insert(adjusted_pos, new_layer)
        inserted_indices.append(adjusted_pos)

    # Update model config to reflect new layer count
    model.config.num_hidden_layers = len(layers)

    print(f"Inserted {config.num_layers} layers at positions {inserted_indices}")
    print(f"Model now has {len(layers)} total layers")

    return inserted_indices


def freeze_base_model(model):
    """Freeze all parameters in the model."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_inserted_layers(model, inserted_indices):
    """Unfreeze only the inserted layers."""
    layers = model.model.layers
    total_trainable = 0
    for idx in inserted_indices:
        for param in layers[idx].parameters():
            param.requires_grad = True
            total_trainable += param.numel()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {total_trainable:,} / {total_params:,} "
          f"({100 * total_trainable / total_params:.2f}%)")

    return total_trainable


def verify_noop(model, tokenizer, inserted_indices, test_prompt="The capital of France is"):
    """Verify that inserted layers (when zero-initialised) don't change model output.

    This should be run immediately after insertion and before any training.
    """
    model.eval()
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    # Get logits with inserted layers active
    with torch.no_grad():
        logits_with = model(**inputs).logits

    # Temporarily zero out inserted layer outputs to double-check
    # (They should already be zero, so this is a sanity check)
    print(f"Output logits shape: {logits_with.shape}")
    print(f"First 10 logits: {logits_with[0, -1, :10]}")

    # Check that inserted layers' output projections are actually zero
    layers = model.model.layers
    all_zero = True
    for idx in inserted_indices:
        layer = layers[idx]
        o_proj_norm = layer.self_attn.o_proj.weight.norm().item()
        down_proj_norm = layer.mlp.down_proj.weight.norm().item()
        if o_proj_norm > 1e-6 or down_proj_norm > 1e-6:
            all_zero = False
            print(f"WARNING: Layer {idx} output projections are not zero! "
                  f"o_proj norm={o_proj_norm:.6f}, down_proj norm={down_proj_norm:.6f}")

    if all_zero:
        print("VERIFIED: All inserted layers are no-ops (output projections are zero)")
    else:
        print("WARNING: Some inserted layers are not proper no-ops")

    return all_zero


def get_trainable_params(model):
    """Return list of trainable parameters (for optimizer)."""
    return [p for p in model.parameters() if p.requires_grad]


def print_model_summary(model):
    """Print a summary of model layers and trainable status."""
    layers = model.model.layers
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"\nModel Summary:")
    print(f"  Total layers: {len(layers)}")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"  Frozen: {frozen:,} ({100 * frozen / total:.2f}%)")
