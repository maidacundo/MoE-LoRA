# MoE-LoRA: Mixture-of-Experts Adaptation using Parameter Efficient Fine-tuning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training](#training)
- [Advanced Usage](#advanced-usage)
- [How It Works](#how-it-works)
- [Performance & Efficiency](#performance--efficiency)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Contributing](#contributing)

## Overview

**MoE-LoRA** transforms standard decoder-only language models (like Mistral 7B) into efficient Mixture-of-Experts (MoE) models (similar to Mixtral 8x7B) using **Parameter Efficient Fine-Tuning (PEFT)** with LoRA (Low-Rank Adaptation).

Instead of training billions of parameters from scratch, MoE-LoRA injects trainable LoRA adapters into the Feed-Forward Network (FFN) layers, creating multiple "expert" pathways while keeping the base model frozen. This approach dramatically reduces training costs while enabling MoE capabilities.

### What is MoE?
Mixture-of-Experts models route each token to a subset of specialized "expert" networks, allowing the model to be larger while keeping computational costs manageable. Only a few experts process each token, providing efficiency at scale.

### What is LoRA?
Low-Rank Adaptation (LoRA) freezes pre-trained model weights and injects trainable low-rank matrices into each layer, reducing the number of trainable parameters by orders of magnitude while maintaining model quality.

### Why MoE-LoRA?
By combining MoE and LoRA, you can:
- Convert existing models to MoE architecture without full retraining
- Train with minimal GPU memory using quantization
- Achieve parameter efficiency (train <1% of total parameters)
- Experiment with different expert configurations rapidly
- Deploy expert systems on consumer hardware

## Key Features

- **Parameter Efficient**: Train only LoRA adapters (~0.1-1% of model parameters)
- **Memory Efficient**: Supports 4-bit and 8-bit quantization via bitsandbytes
- **Flexible Architecture**: Configure number of experts, routing strategy, and expert rank
- **Compatible**: Works with any Mistral-based or LLaMA-based model
- **Router Learning**: Trainable gating network with optional auxiliary loss
- **Production Ready**: Includes training scripts for OpenAssistant and Wikipedia datasets

## Architecture

```
┌─────────────────────────────────────────┐
│         Input Embeddings                │
└──────────────┬──────────────────────────┘
               │
      ┌────────▼────────┐
      │  Self-Attention  │ (Frozen)
      └────────┬─────────┘
               │
      ┌────────▼────────────────────────────┐
      │     MoE-LoRA Block                  │
      │  ┌──────────────────────────────┐   │
      │  │  Router (Gating Network)     │   │
      │  └──────┬───────────────────────┘   │
      │         │                            │
      │    ┌────▼──────┐ Top-K Selection    │
      │    │  Expert 1  │ (LoRA Adapters)   │
      │    │  Expert 2  │                    │
      │    │    ...     │                    │
      │    │  Expert N  │                    │
      │    └───────────┘                     │
      │         │                            │
      │    Weighted Sum                      │
      └────────┬───────────────────────────┘
               │
         ┌─────▼──────┐
         │   Output   │
         └────────────┘
```

Each **LoraExpert** wraps the frozen FFN with three LoRA adapters:
- `gate_lora`: Low-rank adaptation for gate projection
- `up_lora`: Low-rank adaptation for up projection
- `down_lora`: Low-rank adaptation for down projection

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/maidacundo/MoE-LoRA.git
cd MoE-LoRA/

# Install dependencies
pip install -r requirements.txt

# Login to required services
wandb login           # For experiment tracking
huggingface-cli login # For model downloads
```

### Requirements
```
transformers>=4.38.0
datasets
accelerate
evaluate
wandb
bitsandbytes
peft
```

## Quick Start

### Basic Usage

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from lora_moe import LoraMoeConfig, LoraMoeModel
import torch

# Configure 4-bit quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto",
)

# Configure MoE-LoRA
moe_config = LoraMoeConfig.from_pretrained("mistralai/Mistral-7B-v0.1")
moe_config.experts_rank = 8              # LoRA rank (higher = more capacity)
moe_config.experts_scale = 1.0           # LoRA scaling factor
moe_config.num_experts_per_tok = 2       # Experts active per token
moe_config.num_local_experts = 8         # Total number of experts
moe_config.output_router_logits = True   # Enable router loss

# Wrap model with MoE-LoRA
moe_model = LoraMoeModel(base_model, moe_config)

# Freeze base model, train only LoRA experts
moe_model.make_experts_trainable()

# Use like any Hugging Face model
outputs = moe_model(input_ids=input_ids, labels=labels)
loss = outputs.loss
```

### Training

#### Train on OpenAssistant Dataset
```bash
accelerate launch train_openassistant.py
```

#### Train on Wikipedia Dataset
```bash
accelerate launch train_wikipedia.py
```

Both scripts use the configurations in `training/training_config.py`.

## Configuration

### MoE-LoRA Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experts_rank` | int | 8 | Rank of LoRA projection matrices (controls expert capacity) |
| `experts_scale` | float | 1.0 | Scaling factor applied to LoRA outputs |
| `num_experts_per_tok` | int | 2 | Number of experts activated per token (top-k routing) |
| `num_local_experts` | int | 8 | Total number of expert modules |
| `output_router_logits` | bool | False | Whether to return routing weights (needed for auxiliary loss) |
| `router_aux_loss_coef` | float | 0.001 | Weight of load-balancing auxiliary loss |

### Training Parameters

Edit `training/training_config.py` to customize training:

```python
@dataclass
class TrainingConfig:
    # Dataset
    dataset: str = "openassistant"  # or "wikipedia"

    # LoRA MoE
    experts_rank: int = 8
    experts_scale: float = 1.0
    num_experts_per_tok: int = 2
    num_local_experts: int = 8

    # Training
    num_epochs: int = 1
    train_batch_size: int = 1
    learning_rate: float = 1e-4
    context_length: int = 64

    # Model
    base_model_id: str = "mistralai/Mistral-7B-v0.1"
    quantize: bool = True
    mixed_precision: str = "fp16"

    # Logging
    project_name: str = "lora_moe"
    run_name: str = "experiment_1"
```

## Training

The training pipeline uses Hugging Face Accelerate for distributed training and mixed precision:

```python
from training import train, TrainingConfig

# Create custom config
config = TrainingConfig(
    dataset="openassistant",
    experts_rank=16,
    num_local_experts=4,
    learning_rate=2e-4,
)

# Launch training
train(config)
```

### Multi-GPU Training
```bash
accelerate config  # Configure distributed setup
accelerate launch --num_processes=2 train_openassistant.py
```

## Advanced Usage

### Selective Layer Wrapping
Apply MoE-LoRA to specific transformer layers:

```python
# Only wrap layers 10-20
moe_model = LoraMoeModel(
    base_model,
    moe_config,
    layer_ids=list(range(10, 20))
)
```

### Custom Expert Implementation
Extend the `LoraExpert` class to create specialized expert architectures:

```python
from lora_moe.peft_experts import LoraExpert

class CustomExpert(LoraExpert):
    def __init__(self, config):
        super().__init__(config)
        # Add custom layers

    def forward(self, hidden_states, mlp):
        # Custom expert logic
        return output
```

### Inference
```python
# Generate text
input_ids = tokenizer("Once upon a time", return_tensors="pt").input_ids
outputs = moe_model.generate(
    input_ids,
    max_length=100,
    temperature=0.7,
    top_p=0.9,
)
print(tokenizer.decode(outputs[0]))
```

## How It Works

### 1. Model Wrapping
MoE-LoRA wraps each transformer decoder layer's FFN with a `LoraMoeBlock` containing:
- **Router**: Learns to assign tokens to experts using a noisy top-k gating mechanism
- **LoRA Experts**: Multiple low-rank adapter modules that process token representations

### 2. Expert Routing
For each token:
1. Router computes logits for all experts
2. Top-k experts are selected based on highest routing weights
3. Token representation is processed by selected experts
4. Expert outputs are combined via weighted sum

### 3. Training
Only LoRA adapter parameters and router weights are trained:
```
Total parameters: ~7B
Trainable parameters: ~50M (0.7%)
```

The auxiliary load-balancing loss encourages even expert utilization:
```python
aux_loss = load_balancing_loss_func(router_logits, num_experts, top_k)
total_loss = task_loss + router_aux_loss_coef * aux_loss
```

## Performance & Efficiency

### Memory Usage
| Configuration | VRAM (Training) | VRAM (Inference) |
|---------------|-----------------|------------------|
| 7B base, 8 experts, rank 8, 4-bit | ~12 GB | ~6 GB |
| 7B base, 8 experts, rank 16, 4-bit | ~14 GB | ~7 GB |
| 7B base, 16 experts, rank 8, 4-bit | ~16 GB | ~8 GB |

### Computation
- **Sparse Activation**: Only 2/8 experts active per token (25% of expert capacity)
- **Efficient Routing**: Block-sparse operations avoid padding overhead
- **Gradient Efficiency**: Only ~1% of parameters receive gradients

## Troubleshooting

### Out of Memory
- Reduce `experts_rank` (e.g., 4 or 8)
- Reduce `num_local_experts` (e.g., 4 instead of 8)
- Enable gradient checkpointing
- Reduce `train_batch_size`
- Use deeper quantization (4-bit instead of 8-bit)

### Router Collapse (All tokens to one expert)
- Increase `router_aux_loss_coef` (e.g., 0.01)
- Verify `output_router_logits=True` in config
- Check that router weights are being updated

### Slow Training
- Ensure CUDA is available: `torch.cuda.is_available()`
- Use `torch.compile()` for PyTorch 2.0+
- Enable Flash Attention 2 if available
- Reduce `context_length` for faster iterations

### Import Errors
```bash
pip install --upgrade transformers accelerate peft
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{moe_lora_2024,
  author = {maidacundo},
  title = {MoE-LoRA: Mixture-of-Experts Adaptation using Parameter Efficient Fine-tuning},
  year = {2024},
  url = {https://github.com/maidacundo/MoE-LoRA}
}
```

### Related Work
- **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Mixtral**: [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- **Switch Transformers**: [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
pip install -r requirements.txt
pip install pytest black flake8  # Dev dependencies
```

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Acknowledgments

- Built on [Hugging Face Transformers](https://github.com/huggingface/transformers)
- Inspired by [Mixtral](https://mistral.ai/news/mixtral-of-experts/) and [LoRA](https://github.com/microsoft/LoRA)
- Uses [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for quantization
