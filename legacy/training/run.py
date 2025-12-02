# -*- coding: utf-8 -*-

from datasets import load_from_disk
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          Trainer)
from transformers.integrations import WandbCallback
import torch

import fla  # noqa
from flame.data import DataCollatorForLanguageModeling
from flame.logging import LogCallback, LossCorrectionCallback, get_logger
from flame.parser import get_train_args
from flame.freeze_callback import FreezeLazyParamsCallback, MonitorLazyParamsCallback, DynamicLazyLRCallback

logger = get_logger(__name__)


class LazyParamTrainer(Trainer):
    """
    Custom Trainer that uses higher learning rate for bias/tau parameters.
    This helps overcome bf16 precision loss in optimizer updates.
    """

    def __init__(self, lazy_lr_multiplier=100.0, **kwargs):
        self.lazy_lr_multiplier = lazy_lr_multiplier
        super().__init__(**kwargs)

    def create_optimizer(self):
        """Create optimizer with parameter-specific learning rates."""
        if self.optimizer is not None:
            return self.optimizer

        # Separate parameters into lazy params and others
        lazy_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'learnable_bias_diagonals' in name or '.tau' in name:
                lazy_params.append(param)
                logger.info(f"Lazy param (higher lr): {name}")
            else:
                other_params.append(param)

        base_lr = self.args.learning_rate
        lazy_lr = base_lr * self.lazy_lr_multiplier

        logger.info(f"Base LR: {base_lr}, Lazy params LR: {lazy_lr} ({self.lazy_lr_multiplier}x)")

        optimizer_grouped_parameters = [
            {"params": other_params, "lr": base_lr},
            {"params": lazy_params, "lr": lazy_lr},
        ]

        # Use the same optimizer class as default
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)

        # Remove lr from kwargs since we set it per group
        optimizer_kwargs.pop("lr", None)

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


def main():
    args = get_train_args()
    logger.info(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=True,
        add_bos_token=True,
        add_eos_token=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))
    if args.from_config:
        logger.info("All model params are randomly initialized for from-scratch training.")
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(args.model_name_or_path))
    else:
        logger.info(f"Loading pretrained checkpoint {args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.train()

    trainable_params, all_param = model.num_parameters(only_trainable=True), model.num_parameters()
    logger.info(f"% of trainable params: {trainable_params:d} / {all_param:d} = {trainable_params / all_param:.2%}")
    logger.info(f"{tokenizer}\n{model}\n{model.config}")

    logger.info(f"Loading the `{args.split}` split directly from the cache {args.cache_dir}...")
    dataset = load_from_disk(args.cache_dir)
    logger.info(f"{dataset}")
    logger.info(f"Shuffling the dataset with seed {args.seed}")
    dataset = dataset.shuffle(seed=args.seed)
    logger.info("Creating the data collator")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, varlen=args.varlen)
    logger.info(f"{data_collator}")

    if args.lr_scheduler_type == 'cosine_with_min_lr':
        args.lr_scheduler_kwargs = {'min_lr_rate': 0.1}
    if args.lr_scheduler_type == 'warmup_stable_decay':
        args.lr_scheduler_kwargs = {
            'num_stable_steps': args.max_steps * 0.9 - args.warmup_steps,
            'num_decay_steps': args.max_steps * 0.1
        }

    # Callbacks
    # LossCorrectionCallback MUST be first to correct loss before WandbCallback sees it
    callbacks = [LossCorrectionCallback(), LogCallback()]

    # Monitor lazy attention parameters (bias, tau) to observe convergence
    monitor_lazy_steps = getattr(args, 'monitor_lazy_params_every', None)
    if monitor_lazy_steps is not None and monitor_lazy_steps > 0:
        logger.info(f"Will monitor bias/tau parameters every {monitor_lazy_steps} steps")
        callbacks.append(MonitorLazyParamsCallback(
            log_every_n_steps=monitor_lazy_steps,
            use_wandb=getattr(args, 'report_to', None) == 'wandb',
        ))

    # For SWAT models: freeze bias/tau after N steps to speed up training
    # This avoids slow atomic_add operations in Triton backward pass
    freeze_after_steps = getattr(args, 'freeze_lazy_params_after', None)
    if freeze_after_steps is not None and freeze_after_steps > 0:
        logger.info(f"Will freeze bias/tau parameters after {freeze_after_steps} steps")
        callbacks.append(FreezeLazyParamsCallback(freeze_after_steps=freeze_after_steps))

    # Dynamic lazy LR: multiply base_lr by loss * 10
    # lazy_lr = base_lr * loss * 10
    dynamic_lazy_lr = getattr(args, 'dynamic_lazy_lr', False)
    if dynamic_lazy_lr:
        min_mult = getattr(args, 'dynamic_lazy_lr_min_mult', 1.0)
        logger.info(f"Dynamic lazy LR enabled: lazy_lr = base_lr * loss * 10 (min_mult={min_mult}x)")
        callbacks.append(DynamicLazyLRCallback(
            min_mult=min_mult,
        ))

    # Get lazy_lr_multiplier from args, default 100x to overcome bf16 precision loss
    lazy_lr_multiplier = getattr(args, 'lazy_lr_multiplier', 100.0)

    trainer = LazyParamTrainer(
        lazy_lr_multiplier=lazy_lr_multiplier,
        model=model,
        args=args,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        train_dataset=dataset
    )

    # Reorder callbacks: LossCorrectionCallback must run BEFORE WandbCallback
    # HuggingFace Trainer adds default callbacks (including WandbCallback) before user callbacks
    # So we need to manually reorder them
    #
    # Find our LossCorrectionCallback instance from the callbacks list
    loss_correction_cb = None
    wandb_callback = None
    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, LossCorrectionCallback):
            loss_correction_cb = cb
        if isinstance(cb, WandbCallback):
            wandb_callback = cb

    if wandb_callback is not None and loss_correction_cb is not None:
        # Remove both, then re-add in correct order: LossCorrectionCallback -> WandbCallback
        trainer.remove_callback(LossCorrectionCallback)
        trainer.remove_callback(WandbCallback)
        trainer.add_callback(loss_correction_cb)
        trainer.add_callback(wandb_callback)
        logger.info("Reordered callbacks: LossCorrectionCallback -> WandbCallback for correct loss logging")

    results = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(trainer.args.output_dir)

    trainer.log_metrics("train", results.metrics)
    trainer.save_metrics("train", results.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
