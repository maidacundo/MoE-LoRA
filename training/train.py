
from wikipedia.dataset import get_datasets

from lora_moe import LoraMoeConfig, LoraMoeModel 
from training_config import TrainingConfiguration

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_scheduler
from peft import prepare_model_for_kbit_training

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
import numpy as np
import os
from tqdm import tqdm
import math

from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from accelerate import Accelerator, DistributedDataParallelKwargs

logger = get_logger(__name__)


def evaluate(model, accelerator, eval_dataloader):
    losses = []
    aux_losses = []
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                output_router_logits=True,
                )
        losses.append(accelerator.gather(outputs.loss))
        aux_losses.append(accelerator.gather(outputs.aux_loss))
    loss = torch.mean(torch.cat(losses))
    aux_loss = torch.mean(torch.cat(aux_losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")

    return loss.item(), aux_loss.item(), perplexity.item()

def train(config: TrainingConfiguration):

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
        logger.info(f"Resuming from {config.resume_from}")

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config, # TODO try to add the grad_accumulation_steps
        split_batches=True,
        # kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.project_name,
            config=config,
            init_kwargs={"wandb": {
                "name": config.run_name,
                "entity": "maidacundo",
                }},
        )

    set_seed(config.seed)

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    with accelerator.main_process_first():
        tokenized_datasets = get_datasets(
            tokenizer, 
            context_length=config.context_length, 
            num_train_texts=config.num_train_texts,
            )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_dataloader = DataLoader(
        tokenized_datasets['train'], 
        batch_size=config.train_batch_size, 
        collate_fn=data_collator, 
        shuffle=True,)
    
    eval_dataloader = DataLoader(
        tokenized_datasets['test'], 
        batch_size=config.eval_batch_size, 
        collate_fn=data_collator,)

    bnb_config = None
    if config.quantize:
        bnb_config = get_qlora_bnb_config()

    model_config = LoraMoeConfig.from_pretrained(config.base_model_id)
    model_config.experts_rank = config.experts_rank
    model_config.experts_scale = config.experts_scale
    model_config.num_experts_per_tok = config.num_experts_per_tok
    model_config.num_local_experts = config.num_local_experts
    model_config.output_router_logits = True
       
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        quantization_config=bnb_config,
        config=model_config,
    )

    moe_model = LoraMoeModel(model, config)

    if config.quantize:
        moe_model = prepare_model_for_kbit_training(moe_model, use_gradient_checkpointing=True)
    

    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(moe_model.parameters(), lr=config.learning_rate)

    train_steps_num = config.num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        "cosine_with_restarts",
        optimizer,
        num_warmup_steps=math.ceil(train_steps_num * 0.05), # 5% of warmup
        num_training_steps=train_steps_num,
    )

    if not os.path.exists(config.checkpoint_folder):
        os.makedirs(config.checkpoint_folder)
        print(f"Directory '{config.checkpoint_folder}' created.")
    else:
        print(f"Directory '{config.checkpoint_folder}' already exists.")


    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        global_step = int(config.resume_from.split("_")[-1]) + 1
    else:
        global_step = 1

    logger.info("Preparing model for training...")
    moe_model, optimizer, lr_scheduler, train_dataloader, eval_dataloader = accelerator.prepare(moe_model, optimizer, lr_scheduler, train_dataloader, eval_dataloader)

    progress_bar = tqdm(range(train_steps_num))
    progress_bar.set_description("Steps")
    
    for _ in range(config.num_epochs):
        moe_model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            output = moe_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                output_router_logits=True)
            loss = output.loss
            
            accelerator.backward(loss)
            total_norm = None
            if accelerator.sync_gradients:
                total_norm = accelerator.clip_grad_norm_(moe_model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            
            logs = {}
            logs["train_loss"] = output.loss.detach().item()
            logs['train_aux_loss'] = output.aux_loss.detach().item()
            progress_bar.set_postfix(**logs)
            logs["lr"] = lr_scheduler.get_last_lr()[0]
            if total_norm is not None:
                logs["total_norm"] = total_norm.item()
            progress_bar.update(1)

            accelerator.log(logs, step=global_step)

            global_step += 1
            if global_step % config.eval_steps == 0:
                accelerator.wait_for_everyone()
                val_loss, val_aux_loss, val_perplexity = evaluate(moe_model, accelerator, eval_dataloader)
                logs = {
                    "val_loss": val_loss,
                    "val_aux_loss": val_aux_loss,
                    "val_perplexity": val_perplexity,
                }
                accelerator.log(logs, step=global_step)
                accelerator.wait_for_everyone()
                accelerator.save_model(moe_model, config.checkpoint_folder)
    accelerator.end_training()

def get_qlora_bnb_config() -> BitsAndBytesConfig:

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    return bnb_config

