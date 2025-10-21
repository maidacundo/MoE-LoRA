"""
LoRA Expert Modules

This module implements LoRA-adapted expert networks for MoE-LoRA models.
Each expert consists of low-rank projection matrices applied to the FFN layers.
"""

from torch import nn

from transformers.activations import ACT2FN
from transformers.models.mistral.modeling_mistral import (
    MistralMLP,
)

from .configuration_lora_moe import LoraMoeConfig

class LoraInjectedLinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) linear layer.

    Implements a low-rank decomposition of a linear transformation as:
    h = W_0(x) + s * B(A(x))

    where W_0 is the frozen pretrained weight, A and B are trainable low-rank
    matrices, and s is a scaling factor.

    Args:
        in_features (int): Size of input features
        out_features (int): Size of output features
        r (int, optional): Rank of the low-rank decomposition. Defaults to 8.
        scale (float, optional): Scaling factor for LoRA output. Defaults to 1.0.
        dropout (float, optional): Dropout probability. Defaults to 0.1.

    Attributes:
        A: Down-projection matrix (in_features -> r)
        B: Up-projection matrix (r -> out_features)
        scale: Output scaling factor
        dropout: Dropout layer applied to input
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 8,
            scale: float = 1.0,
            dropout: float = 0.1,
            ):

        super().__init__()
        self.r = r

        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)
        self.scale = scale
        self.dropout = nn.Dropout(dropout)

        # Initialize A with random values, B with zeros (identity at initialization)
        nn.init.normal_(self.A.weight, std=1 / r)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        """
        Forward pass through LoRA adapter.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        x = self.dropout(x)
        x = self.A(x)
        x = self.B(x)
        x = x * self.scale
        return x

class LoraExpert(nn.Module):
    """
    LoRA-adapted expert module for MoE-LoRA.

    Each expert wraps the frozen base MLP with trainable LoRA adapters on the
    gate, up, and down projections. This allows parameter-efficient specialization
    of experts without modifying the base model weights.

    Architecture:
        gate_out = gate_proj(x) + gate_lora(x)
        up_out = up_proj(x) + up_lora(x)
        intermediate = activation(gate_out * up_out)
        output = down_proj(intermediate) + down_lora(intermediate)

    Args:
        config (LoraMoeConfig): Configuration containing expert parameters

    Attributes:
        gate_lora: LoRA adapter for gate projection
        up_lora: LoRA adapter for up projection
        down_lora: LoRA adapter for down projection
        activation_fn: Activation function (e.g., SiLU)
    """
    def __init__(self, config: LoraMoeConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.rank = config.experts_rank
        self.scale = config.experts_scale

        self.gate_lora = LoraInjectedLinear(config.hidden_size, config.intermediate_size, self.rank, self.scale)
        self.up_lora = LoraInjectedLinear(config.hidden_size, config.intermediate_size, self.rank, self.scale)
        self.down_lora = LoraInjectedLinear(config.intermediate_size, config.hidden_size, self.rank, self.scale)

        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, mlp: MistralMLP):
        """
        Forward pass combining frozen MLP with LoRA adapters.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            mlp: Frozen base MLP module containing pre-trained weights

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Apply frozen projections + LoRA adapters
        up = mlp.up_proj(hidden_states)
        up += self.up_lora(hidden_states)
        gate = mlp.gate_proj(hidden_states)
        gate += self.gate_lora(hidden_states)

        # Gated activation
        act = self.activation_fn(gate * up)

        # Down projection with LoRA adapter
        down = mlp.down_proj(act)
        down += self.down_lora(act)
        return down


# TODO implement DoraInjectedLinear well
class DoraInjectedLinear(nn.Module):
    def __init__(
            self, 
            in_features: int,
            out_features: int,
            r=8,
            scale=1.0,
            ):
        
        super().__init__()
        self.r = r

        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        x = self.lora_down(x)
        x = self.lora_up(x)
        x = x * self.scale
        return x

class DoraExpert(nn.Module):
    def __init__(self, config: LoraMoeConfig, scale=1.0,):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.rank = config.experts_rank
        self.activation_fn = ACT2FN[config.hidden_act]
        self.gate_lora = DoraInjectedLinear(config.hidden_size, config.intermediate_size, self.rank, scale)
        self.up_lora = DoraInjectedLinear(config.hidden_size, config.intermediate_size, self.rank, scale)
        self.down_lora = DoraInjectedLinear(config.intermediate_size, config.hidden_size, self.rank, scale)

    def forward(self, hidden_states, mlp: MistralMLP):
        up = mlp.up_proj(hidden_states)
        up += self.up_lora(hidden_states)
        gate = mlp.gate_proj(hidden_states)
        gate += self.gate_lora(hidden_states)
        act = self.activation_fn(gate * up)
        down = mlp.down_proj(act)
        down += self.down_lora(act)
        return down

