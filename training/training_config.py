from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingConfiguration:

    # LoRA MoE parameters
    experts_rank: int = 8
    experts_scale: float = 1.0
    num_experts_per_tok: int = 2
    num_local_experts: int = 8

    # Training parameters
    seed: int = 42
    num_epochs: int = 1
    train_batch_size: int = 8
    eval_batch_size: int = 8
    context_length: int = 512 # number of tokens to use as context
    num_train_texts: int = 5000 # number of wikipedia articles to use for training
    learning_rate: float = 1e-4 
    eval_steps: int = 100 # how often to evaluate the model

    # Model parameters
    mixed_precision: bool = "fp16"
    use_8bit_adam: bool = False
    quantize: bool = False
    base_model_id: str = "mistralai/Mistral-7B-v0.1"

    # Logging parameters
    resume_from: Optional[str] = None # TODO
    checkpoint_folder: str = "checkpoints" 
    num_checkpoint_limit: int = 1 # number of checkpoints to keep
    logdir: str = "logs"
    project_name: str = "lora_moe"
    run_name: Optional[str] = "test_run"
