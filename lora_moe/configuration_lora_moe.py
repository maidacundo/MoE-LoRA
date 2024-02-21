""" MoE LoRA model configuration"""

from transformers.models.mistral.modeling_mistral import MistralConfig


class LoraMoeConfig(MistralConfig):
    r"""
        experts_ranks (`int`, *optional*, defaults to 8):
            The rank of the LoRA experts
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            The number of experts to root per-token, can be also interpreted as the `top-p` routing
            parameter
        num_local_experts (`int`, *optional*, defaults to 8):
            Number of experts per Sparse MLP layer.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabeling this will also
            allow the model to output the auxiliary loss. See [here]() for more details
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
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
