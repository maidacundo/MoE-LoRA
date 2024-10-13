# MoE-LoRA: Mixture-of-Experts Adaptation of LLM using Parameter Efficient Method

This implementation adapts a LLama-like model (like Mistral 7B) to a Mixture-of-Experts model (like Mixstral 8x7B), using Parameter Efficient finetuning (LoRA).
LoRA adapters are injected in the FFN to mimic finetuning of Mixstral.

```python
from transformers import AutoModelForCausalLM
from lora_moe import LoraMoeConfig, LoraMoeModel

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_id,
    quantization_config=bnb_config,
)

model_config = LoraMoeConfig.from_pretrained(config.base_model_id)
model_config.experts_rank = 8 # rank of LoRA experts
model_config.experts_scale = 1 # LoRA scale
model_config.num_experts_per_tok = 2 # number of expert to use for each token
model_config.num_local_experts = 8 # numer of LoRA experts to initialize
model_config.output_router_logits = True

moe_model = LoraMoeModel(model, model_config) # injects MoE-LoRA adapters in the FFN
moe_model.make_experts_trainable() # train only the adapters
```

```bash
git clone https://github.com/maidacundo/MoE-LoRA.git
cd MoE-LoRA/
pip install -r requirements.txt
wandb login
huggingface-cli login
accelerate launch train_openassistant.py
```
