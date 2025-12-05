# -*- coding: utf-8 -*-
"""
Training arguments for SWAT/Lazy Attention models.

Lazy Attention Parameters (bias/tau) Training:
==============================================

These parameters control how bias and tau are trained:

1. --lazy_lr_multiplier (default: 10.0)
   Peak LR for lazy params = base_lr * this
   Example: base_lr=1e-4, multiplier=10 -> lazy peak LR = 1e-3

2. --lazy_warmup_multiplier (default: 2.0)
   lazy_warmup_steps = base_warmup_steps * this
   Example: warmup=512, multiplier=2 -> lazy warmup = 1024 steps

3. --lazy_total_steps_multiplier (default: 2.0)
   lazy_total_steps = lazy_warmup_steps * this
   After this step, bias/tau are FROZEN (no more training)
   Example: lazy_warmup=1024, multiplier=2 -> freeze at step 2048

   Special values:
   - Set to 0: Freeze immediately from step 0 (no training at all)

4. --freeze_lazy_from_checkpoint (default: None)
   Load bias/tau from a pretrained checkpoint and freeze immediately.
   Useful for reusing learned lazy params without further training.
   Example: --freeze_lazy_from_checkpoint /path/to/checkpoint-2048

5. --monitor_lazy_params_every (default: None)
   Print bias/tau statistics every N steps to monitor convergence.
   Example: --monitor_lazy_params_every 100

6. --lazy_delayed_start_multiplier (default: 0)
   Delayed start mode: freeze lazy params initially, start training at warmup * this.
   When > 0, enables delayed constant mode (instead of cosine mode):
   - Steps 0 to (warmup * this - 1): params FROZEN (fast backward, skips atomic_add)
   - Steps (warmup * this) onwards: UNFREEZE, constant LR = base_lr * lr_multiplier
   - No auto-freeze, train until end
   Example: --lazy_delayed_start_multiplier 4 (with warmup=512 -> freeze 0-2047, train 2048+)

Example Commands:
-----------------

# Normal training (lazy params trained for 4x warmup steps, then frozen)
python run.py --warmup_steps 512

# No lazy param training (freeze immediately)
python run.py --lazy_total_steps_multiplier 0

# Load pretrained lazy params and freeze
python run.py --freeze_lazy_from_checkpoint /path/to/checkpoint-2048

# Custom schedule: slower warmup (3x), longer training (3x warmup = 9x base warmup)
python run.py --lazy_warmup_multiplier 3.0 --lazy_total_steps_multiplier 3.0

# Delayed start: freeze first 4x warmup steps, then constant LR until end
# (optimizes early training speed via fast backward)
python run.py --lazy_delayed_start_multiplier 4.0 --lazy_lr_multiplier 10.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import HfArgumentParser, TrainingArguments

from flame.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingArguments(TrainingArguments):
    """Extended TrainingArguments with lazy attention parameter controls."""

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    tokenizer: str = field(
        default="fla-hub/gla-1.3B-100B",
        metadata={"help": "Name of the tokenizer to use."}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    from_config: bool = field(
        default=True,
        metadata={"help": "Whether to initialize models from scratch."},
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The dataset(s) to use. Use commas to separate multiple datasets."},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of provided dataset(s) to use."},
    )
    cache_dir: str = field(
        default=None,
        metadata={"help": "Path to the cached tokenized dataset."},
    )
    split: str = field(
        default="train",
        metadata={"help": "Which dataset split to use for training and evaluation."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    buffer_size: int = field(
        default=2048,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."},
    )
    context_length: int = field(
        default=2048,
        metadata={"help": "The context length of the tokenized inputs in the dataset."},
    )
    varlen: bool = field(
        default=False,
        metadata={"help": "Enable training with variable length inputs."},
    )
    freeze_lazy_params_after: Optional[int] = field(
        default=None,
        metadata={
            "help": "Freeze bias/tau parameters in SWAT attention after N steps. "
                    "This speeds up Triton backward by avoiding atomic_add. "
                    "Set to None or 0 to disable."
        },
    )
    monitor_lazy_params_every: Optional[int] = field(
        default=None,
        metadata={
            "help": "Log bias/tau parameter statistics every N steps to monitor convergence. "
                    "Useful for determining when to freeze parameters. "
                    "Set to None or 0 to disable."
        },
    )
    lazy_lr_multiplier: float = field(
        default=10.0,
        metadata={
            "help": "Learning rate multiplier for bias/tau parameters. "
                    "Default 10x means lazy params use 10x higher max LR than other params."
        },
    )
    lazy_warmup_multiplier: float = field(
        default=2.0,
        metadata={
            "help": "Warmup multiplier for lazy params. "
                    "lazy_warmup = base_warmup * this. Default 2x means slower warmup."
        },
    )
    lazy_total_steps_multiplier: float = field(
        default=2.0,
        metadata={
            "help": "Total steps multiplier for lazy params (relative to lazy_warmup). "
                    "lazy_total = lazy_warmup * this. After this, params are frozen. "
                    "Set to 0 to freeze immediately from start (no training). "
                    "Example: warmup=512, warmup_mult=2, total_mult=2 -> freeze at step 2048."
        },
    )
    lazy_delayed_start_multiplier: float = field(
        default=0.0,
        metadata={
            "help": "Delayed start mode: freeze lazy params initially, start training at warmup * this. "
                    "Set to 0 for cosine mode (default). Set to 4.0 to start training at 4x warmup steps. "
                    "After start, uses constant max LR (lr_multiplier * base_lr) until training ends. "
                    "Early steps are optimized (frozen = fast backward, skips atomic_add). "
                    "Example: warmup=512, delayed=4 -> frozen 0-2047, train 2048+ with constant 10x LR."
        },
    )
    freeze_lazy_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to checkpoint to load bias/tau from, then freeze immediately. "
                    "Use this to reuse pretrained lazy params without further training."
        },
    )
    dynamic_lazy_lr: bool = field(
        default=False,
        metadata={
            "help": "Enable dynamic lazy LR based on loss. "
                    "Formula: lazy_lr = base_lr * loss * 10"
        },
    )
    dynamic_lazy_lr_scale: float = field(
        default=1.0,
        metadata={
            "help": "[DEPRECATED - ignored] Scale factor for dynamic lazy LR. "
                    "Now uses simple formula: lazy_lr = base_lr * loss."
        },
    )
    dynamic_lazy_lr_min_mult: float = field(
        default=1.0,
        metadata={
            "help": "Minimum LR multiplier floor for dynamic lazy LR. "
                    "Even when loss is very low, multiplier won't go below this."
        },
    )


def get_train_args():
    parser = HfArgumentParser(TrainingArguments)
    args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if unknown_args:
        print(parser.format_help())
        print("Got unknown args, potentially deprecated arguments: {}".format(unknown_args))
        raise ValueError("Some specified arguments are not used by the HfArgumentParser: {}".format(unknown_args))

    if args.should_log:
        transformers.utils.logging.set_verbosity(args.get_process_log_level())
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    # set seeds manually
    transformers.set_seed(args.seed)
    return args
