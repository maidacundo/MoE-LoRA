from torch import nn

from transformers.activations import ACT2FN
from transformers.models.mistral.modeling_mistral import (
    MistralMLP,
)

from configuration_lora_moe import LoraMoeConfig

class LoraInjectedLinear(nn.Module):
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

        nn.init.normal_(self.A.weight, std=1 / r)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        x = self.dropout(x)
        x = self.A(x)
        x = self.B(x)
        x = x * self.scale
        return x

class LoraExpert(nn.Module):
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
        up = mlp.up_proj(hidden_states)
        up += self.up_lora(hidden_states)
        gate = mlp.gate_proj(hidden_states)
        gate += self.gate_lora(hidden_states)
        act = self.activation_fn(gate * up)
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

