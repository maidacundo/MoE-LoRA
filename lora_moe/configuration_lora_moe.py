"""
MoE-LoRA Model Configuration

This module defines the configuration class for MoE-LoRA models, extending the
Mistral configuration with parameters specific to Mixture-of-Experts and LoRA adaptation.
"""

from transformers.models.mistral.modeling_mistral import MistralConfig

class LoraMoeConfig(MistralConfig):
    r"""
    Configuration class for MoE-LoRA models.

    This configuration extends MistralConfig with additional parameters for configuring
    the Mixture-of-Experts architecture using LoRA (Low-Rank Adaptation) experts.

    Args:
        experts_rank (`int`, *optional*, defaults to 8):
            The rank of the LoRA projection matrices. Higher values provide more
            capacity but increase parameters. Typical values: 4, 8, 16, 32.
        experts_scale (`float`, *optional*, defaults to 1.0):
            Scaling factor applied to LoRA adapter outputs. Controls the magnitude
            of the adaptation. Equivalent to alpha/rank in standard LoRA.
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            Number of experts activated per token (top-k routing). Higher values
            increase computation but may improve quality. Typically 1-4.
        num_local_experts (`int`, *optional*, defaults to 8):
            Total number of expert modules in each MoE layer. Common values: 4, 8, 16.
            More experts provide greater specialization but increase memory usage.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether to return router logits in model outputs. Must be True to compute
            the auxiliary load-balancing loss during training.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            Coefficient for the auxiliary load-balancing loss. Encourages even
            distribution of tokens across experts. Higher values enforce stronger
            load balancing. Typical range: 0.0001 to 0.01.

    Example:
        ```python
        from lora_moe import LoraMoeConfig

        config = LoraMoeConfig.from_pretrained("mistralai/Mistral-7B-v0.1")
        config.experts_rank = 16
        config.num_local_experts = 8
        config.num_experts_per_tok = 2
        config.output_router_logits = True
        ```
    """

    def __init__(
        self,
        experts_rank=8,
        experts_scale=1.0,
        num_experts_per_tok=2,
        num_local_experts=8,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        **kwargs
    ):
        # lora parameters
        self.experts_rank = experts_rank
        self.experts_scale = experts_scale
        
        # moe parameters
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        super().__init__(**kwargs)
