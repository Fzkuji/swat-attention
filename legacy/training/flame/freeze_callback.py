# -*- coding: utf-8 -*-
"""
Callbacks for lazy attention parameter management:
1. FreezeLazyParamsCallback - freeze bias/tau after N steps
2. MonitorLazyParamsCallback - track parameter convergence
3. load_and_freeze_lazy_params - load pretrained bias/tau and freeze
"""

import torch
import torch.distributed as dist
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from transformers.utils import logging

logger = logging.get_logger(__name__)


def load_and_freeze_lazy_params(model, source, verbose=True):
    """
    Load bias and tau from a pretrained source and freeze them.

    Args:
        model: Target model to load params into
        source: Can be:
            - str: Path to checkpoint directory or model file
            - dict: State dict containing bias/tau params
            - nn.Module: Another model to copy params from
        verbose: Print loading info

    Usage:
        # From checkpoint
        load_and_freeze_lazy_params(model, "/path/to/checkpoint-1000")

        # From another model
        load_and_freeze_lazy_params(model, pretrained_model)

        # From state dict
        state_dict = torch.load("model.pt")
        load_and_freeze_lazy_params(model, state_dict)
    """
    import os

    # Get source state dict
    if isinstance(source, str):
        # Path to checkpoint
        if os.path.isdir(source):
            # HuggingFace checkpoint directory
            model_path = os.path.join(source, "pytorch_model.bin")
            if not os.path.exists(model_path):
                model_path = os.path.join(source, "model.safetensors")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No model file found in {source}")
        else:
            model_path = source

        if model_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            source_state = load_file(model_path)
        else:
            source_state = torch.load(model_path, map_location="cpu")
    elif isinstance(source, dict):
        source_state = source
    elif hasattr(source, "state_dict"):
        source_state = source.state_dict()
    else:
        raise TypeError(f"Unknown source type: {type(source)}")

    # Find and load lazy params
    loaded_count = 0
    frozen_count = 0
    target_state = model.state_dict()

    for name, param in model.named_parameters():
        if "learnable_bias_diagonals" in name or "tau" in name:
            if name in source_state:
                # Load the parameter
                param.data.copy_(source_state[name])
                loaded_count += 1

            # Freeze it
            param.requires_grad = False
            frozen_count += 1

    if verbose:
        print(f"[load_and_freeze_lazy_params] Loaded {loaded_count} params, frozen {frozen_count} params")

    return loaded_count, frozen_count


def _is_main_process():
    """Check if this is the main process in distributed training."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _unwrap_model(model):
    """Unwrap model from DeepSpeed/FSDP/DDP wrappers."""
    # Try common wrapper attributes
    if hasattr(model, 'module'):
        return model.module
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


class MonitorLazyParamsCallback(TrainerCallback):
    """
    Monitor bias and tau parameters during training to determine when they converge.

    Logs:
    - Parameter norms (L2)
    - Gradient norms
    - Parameter change rate (delta from last checkpoint)

    Usage:
        trainer = Trainer(
            model=model,
            callbacks=[MonitorLazyParamsCallback(log_every_n_steps=100)],
            ...
        )

    The logs will show when parameters stabilize, helping you choose freeze_after_steps.
    """

    def __init__(
        self,
        log_every_n_steps: int = 100,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
    ):
        self.log_every_n_steps = log_every_n_steps
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard

        # Store previous parameter values to compute change rate
        self.prev_bias_params = {}  # layer_name -> tensor
        self.prev_tau_params = {}

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        if state.global_step % self.log_every_n_steps != 0:
            return

        # Unwrap model for distributed training
        unwrapped_model = _unwrap_model(model)
        metrics = self._collect_metrics(unwrapped_model, state.global_step)

        # Only log on main process
        if _is_main_process():
            # Log to console
            self._log_to_console(state.global_step, metrics)

            # Log to wandb if available
            if self.use_wandb:
                self._log_to_wandb(state.global_step, metrics)

            # Log to tensorboard
            if self.use_tensorboard and hasattr(state, 'log_history'):
                # Trainer will pick up metrics from state
                pass

    def _collect_metrics(self, model, step):
        """Collect bias/tau metrics from all layers."""
        metrics = {
            'bias_norm': [],
            'bias_grad_norm': [],
            'bias_change_rate': [],
            'bias_stats': [],  # per-layer bias statistics (min, max, mean, std)
            'tau_norm': [],
            'tau_grad_norm': [],
            'tau_change_rate': [],
            'tau_values': [],  # actual tau values (small tensor)
        }

        for name, module in model.named_modules():
            # Check for bias parameter
            if hasattr(module, 'learnable_bias_diagonals'):
                param = module.learnable_bias_diagonals
                metrics['bias_norm'].append(param.data.norm().item())

                if param.grad is not None:
                    metrics['bias_grad_norm'].append(param.grad.norm().item())

                # Compute change rate
                if name in self.prev_bias_params:
                    delta = (param.data - self.prev_bias_params[name]).norm().item()
                    prev_norm = self.prev_bias_params[name].norm().item()
                    if prev_norm > 0:
                        metrics['bias_change_rate'].append(delta / prev_norm)

                self.prev_bias_params[name] = param.data.clone()

                # Collect per-layer bias statistics
                # bias shape: [num_heads, max_bias_length]
                bias_data = param.data.cpu().float()
                # Compute stats per head, then aggregate
                head_means = bias_data.mean(dim=1)  # [num_heads]
                head_stds = bias_data.std(dim=1)    # [num_heads]
                # Also compute position-wise stats (how bias varies with distance)
                pos_means = bias_data.mean(dim=0)   # [max_bias_length]
                metrics['bias_stats'].append({
                    'head_mean': head_means.tolist(),
                    'head_std': head_stds.tolist(),
                    'overall_min': bias_data.min().item(),
                    'overall_max': bias_data.max().item(),
                    'overall_mean': bias_data.mean().item(),
                    'overall_std': bias_data.std().item(),
                    # Sample positions: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
                    'pos_samples': [pos_means[min(i, len(pos_means)-1)].item()
                                   for i in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]],
                })

            # Check for tau parameter
            if hasattr(module, 'tau'):
                param = module.tau
                metrics['tau_norm'].append(param.data.norm().item())
                # tau stores actual values directly
                actual_tau = param.data.cpu()
                metrics['tau_values'].append(actual_tau.tolist())

                if param.grad is not None:
                    metrics['tau_grad_norm'].append(param.grad.norm().item())

                if name in self.prev_tau_params:
                    delta = (param.data - self.prev_tau_params[name]).norm().item()
                    prev_norm = self.prev_tau_params[name].norm().item()
                    if prev_norm > 0:
                        metrics['tau_change_rate'].append(delta / prev_norm)

                self.prev_tau_params[name] = param.data.clone()

        # Aggregate metrics
        result = {}
        for key in ['bias_norm', 'bias_grad_norm', 'bias_change_rate',
                    'tau_norm', 'tau_grad_norm', 'tau_change_rate']:
            if metrics[key]:
                result[f'lazy/{key}_mean'] = sum(metrics[key]) / len(metrics[key])
                result[f'lazy/{key}_max'] = max(metrics[key])

        # Store first layer's tau values for reference
        if metrics['tau_values']:
            result['lazy/tau_layer0'] = metrics['tau_values'][0]
            # Store all layers' tau values for full printout
            result['lazy/all_tau_values'] = metrics['tau_values']

        # Store bias stats for all layers
        if metrics['bias_stats']:
            result['lazy/all_bias_stats'] = metrics['bias_stats']

        return result

    def _log_to_console(self, step, metrics):
        """Print metrics to console with full tau values for all layers."""
        bias_norm = metrics.get('lazy/bias_norm_mean', 0)
        bias_grad = metrics.get('lazy/bias_grad_norm_mean', 0)
        bias_change = metrics.get('lazy/bias_change_rate_mean', 0)
        tau_norm = metrics.get('lazy/tau_norm_mean', 0)
        tau_grad = metrics.get('lazy/tau_grad_norm_mean', 0)
        tau_change = metrics.get('lazy/tau_change_rate_mean', 0)

        # Get first layer's tau values for summary line
        tau_values = metrics.get('lazy/tau_layer0', [])
        tau_str = str([round(t, 2) for t in tau_values[:4]]) if tau_values else "N/A"

        # Summary line
        print(
            f"\n[Step {step}] Lazy params ({len(self.prev_bias_params)} layers): "
            f"bias(norm={bias_norm:.4f}, grad={bias_grad:.6f}, change={bias_change:.6f}) "
            f"tau(norm={tau_norm:.4f}, grad={tau_grad:.6f}, change={tau_change:.6f}) "
            f"layer0_tau={tau_str}",
            flush=True
        )

        # Print full tau values for all layers
        all_tau_values = metrics.get('lazy/all_tau_values', [])
        if all_tau_values:
            print(f"  Full tau values per layer:", flush=True)
            for layer_idx, tau_list in enumerate(all_tau_values):
                tau_formatted = [f"{t:.2f}" for t in tau_list]
                tau_min, tau_max, tau_mean = min(tau_list), max(tau_list), sum(tau_list)/len(tau_list)
                print(f"    L{layer_idx:02d}: [{', '.join(tau_formatted)}] (min={tau_min:.2f}, max={tau_max:.2f}, mean={tau_mean:.2f})", flush=True)

        # Print bias statistics per layer
        all_bias_stats = metrics.get('lazy/all_bias_stats', [])
        if all_bias_stats:
            print(f"  Bias statistics per layer (positions 0,1,2,4,8,16,32,64,128,256,512):", flush=True)
            for layer_idx, stats in enumerate(all_bias_stats):
                pos_samples = stats['pos_samples']
                pos_formatted = [f"{p:.3f}" for p in pos_samples]
                print(
                    f"    L{layer_idx:02d}: pos=[{', '.join(pos_formatted)}] "
                    f"(min={stats['overall_min']:.3f}, max={stats['overall_max']:.3f}, "
                    f"mean={stats['overall_mean']:.3f}, std={stats['overall_std']:.3f})",
                    flush=True
                )

        # Print convergence hint
        if bias_change < 0.001 and tau_change < 0.001 and step > 100:
            print(
                f"  -> Parameters appear stable! Consider freezing at step {step}",
                flush=True
            )

    def _log_to_wandb(self, step, metrics):
        """Log to Weights & Biases."""
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, step=step)
        except ImportError:
            pass


class FreezeLazyParamsCallback(TrainerCallback):
    """
    Freezes the learnable bias and tau parameters in SWAttention layers
    after a specified number of training steps.

    This is useful because:
    1. Triton backward for bias/tau uses atomic_add which is very slow
    2. These parameters typically converge quickly
    3. After freezing, backward becomes ~50x faster

    Usage:
        trainer = Trainer(
            model=model,
            args=args,
            callbacks=[FreezeLazyParamsCallback(freeze_after_steps=1000)],
            ...
        )
    """

    def __init__(self, freeze_after_steps: int = 1000, verbose: bool = True):
        """
        Args:
            freeze_after_steps: Number of steps after which to freeze bias/tau
            verbose: Whether to log when freezing happens
        """
        self.freeze_after_steps = freeze_after_steps
        self.verbose = verbose
        self.frozen = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        if self.frozen:
            return

        if state.global_step >= self.freeze_after_steps:
            # Unwrap model for distributed training
            unwrapped_model = _unwrap_model(model)
            self._freeze_lazy_params(unwrapped_model)
            self.frozen = True

            if self.verbose and _is_main_process():
                print(
                    f"\n[FreezeLazyParamsCallback] Froze bias/tau at step {state.global_step}. "
                    f"Backward will be much faster now.",
                    flush=True
                )

    def _freeze_lazy_params(self, model):
        """Freeze learnable_bias_diagonals and tau in all SWAttention layers."""
        frozen_count = 0

        for name, module in model.named_modules():
            # Check for SWAttention layers
            if hasattr(module, 'learnable_bias_diagonals') and hasattr(module, 'tau'):
                # Freeze bias
                if hasattr(module.learnable_bias_diagonals, 'requires_grad'):
                    module.learnable_bias_diagonals.requires_grad = False
                    frozen_count += 1

                # Freeze tau
                if hasattr(module.tau, 'requires_grad'):
                    module.tau.requires_grad = False
                    frozen_count += 1

        if self.verbose and _is_main_process():
            print(f"[FreezeLazyParamsCallback] Froze {frozen_count} parameters", flush=True)


class TwoStageTrainingCallback(TrainerCallback):
    """
    Two-stage training:
    - Stage 1: Use PyTorch naive (slower but correct gradients for bias/tau)
    - Stage 2: Switch to Triton with frozen bias/tau (fast)

    This requires the model to have a `set_use_triton()` method.
    """

    def __init__(
        self,
        switch_after_steps: int = 1000,
        verbose: bool = True
    ):
        self.switch_after_steps = switch_after_steps
        self.verbose = verbose
        self.switched = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        if self.switched:
            return

        if state.global_step >= self.switch_after_steps:
            unwrapped_model = _unwrap_model(model)
            self._switch_to_triton_frozen(unwrapped_model)
            self.switched = True

    def _switch_to_triton_frozen(self, model):
        """Switch to Triton mode and freeze bias/tau."""
        for name, module in model.named_modules():
            # Freeze params
            if hasattr(module, 'learnable_bias_diagonals'):
                module.learnable_bias_diagonals.requires_grad = False
            if hasattr(module, 'tau'):
                module.tau.requires_grad = False

            # Switch to Triton if method exists
            if hasattr(module, 'use_triton'):
                module.use_triton = True

        if self.verbose and _is_main_process():
            print(
                f"[TwoStageTrainingCallback] Switched to Triton mode with frozen bias/tau",
                flush=True
            )


class DynamicLazyLRCallback(TrainerCallback):
    """
    Dynamically adjust learning rate for lazy params (bias/tau) based on loss.

    Formula: lazy_lr = base_lr * loss * 10

    Examples:
        - loss=10: lazy_lr = base_lr * 100 = 100x base_lr
        - loss=5:  lazy_lr = base_lr * 50  = 50x base_lr
        - loss=3:  lazy_lr = base_lr * 30  = 30x base_lr
        - loss=1:  lazy_lr = base_lr * 10  = 10x base_lr

    Usage:
        trainer = Trainer(
            model=model,
            callbacks=[DynamicLazyLRCallback()],
            ...
        )
    """

    def __init__(
        self,
        min_mult: float = 1.0,         # Minimum LR multiplier (floor)
        update_every_n_steps: int = 10,  # How often to update
        verbose: bool = True,
        # Legacy params for compatibility (ignored)
        scale: float = None,
        high_loss: float = None,
        max_mult: float = None,
    ):
        self.min_mult = min_mult
        self.update_every_n_steps = update_every_n_steps
        self.verbose = verbose
        self.current_mult = 10.0  # Initial estimate (typical initial loss)
        self.lazy_param_group_idx = None  # Will be set on first call

    def _find_lazy_param_group(self, optimizer):
        """Find the parameter group index for lazy params."""
        # LazyParamTrainer puts lazy params in the second group (index 1)
        # with higher learning rate
        if len(optimizer.param_groups) >= 2:
            return 1  # Lazy params are in group 1
        return None

    def _compute_multiplier(self, loss: float) -> float:
        """Compute LR multiplier: loss * 10."""
        return max(loss * 10, self.min_mult)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        model=None,
        **kwargs
    ):
        """Update LR when loss is logged."""
        if logs is None or 'loss' not in logs:
            return

        if state.global_step % self.update_every_n_steps != 0:
            return

        loss = logs['loss']
        new_mult = self._compute_multiplier(loss)

        # Get optimizer from trainer (passed in kwargs or state)
        optimizer = kwargs.get('optimizer', None)
        if optimizer is None:
            return

        # Find lazy param group on first call
        if self.lazy_param_group_idx is None:
            self.lazy_param_group_idx = self._find_lazy_param_group(optimizer)

        if self.lazy_param_group_idx is None:
            return

        # Get base LR from first param group
        base_lr = optimizer.param_groups[0]['lr']
        new_lazy_lr = base_lr * new_mult

        # Update lazy param group LR
        old_lazy_lr = optimizer.param_groups[self.lazy_param_group_idx]['lr']
        optimizer.param_groups[self.lazy_param_group_idx]['lr'] = new_lazy_lr

        if self.verbose and _is_main_process() and abs(new_mult - self.current_mult) > 0.5:
            print(
                f"\n[DynamicLazyLR] Step {state.global_step}: loss={loss:.2f} -> "
                f"lazy_mult={new_mult:.1f}x (lazy_lr = base_lr * loss * 10), "
                f"lazy_lr={new_lazy_lr:.2e}",
                flush=True
            )

        self.current_mult = new_mult


class FastCosineSchedulerCallback(TrainerCallback):
    """
    Apply a custom LR schedule to lazy params (bias/tau).

    Supports two modes:

    Mode 1: Fast Cosine with Auto-Freeze (delayed_start_multiplier=0)
    ---------------------------------------------------------------
    Lazy params follow a compressed schedule based on warmup steps:
    - Warmup: base_warmup * warmup_multiplier (default 2x)
    - Total: lazy_warmup * total_steps_multiplier (default 2x, so 4x base_warmup)
    - After total steps: auto-freeze params (no more training)

    Mode 2: Delayed Constant LR (delayed_start_multiplier > 0)
    ----------------------------------------------------------
    - Before start: params FROZEN (requires_grad=False) for fast backward
    - Start step: base_warmup * delayed_start_multiplier
    - After start: UNFREEZE and use constant max LR until training ends
    - No auto-freeze, train until end

    Example Mode 2 with base warmup=512, delayed_start_multiplier=4, lr_multiplier=10:
        - Steps 0-2047: bias/tau FROZEN (fast backward, skips atomic_add)
        - Steps 2048+: UNFREEZE, constant LR = base_lr * 10

    This mode is useful when:
    1. Early training of lazy params hurts overall convergence
    2. You want to optimize compute in early steps (frozen = fast backward)
    3. You want constant LR without decay for lazy params
    """

    def __init__(
        self,
        lr_multiplier: float = 10.0,  # Peak LR multiplier for lazy params
        min_lr_ratio: float = 0.1,    # min_lr = max_lr * min_lr_ratio (only for cosine mode)
        warmup_multiplier: float = 2.0,  # Lazy warmup = base warmup * this (cosine mode)
        total_steps_multiplier: float = 2.0,  # Lazy total = lazy_warmup * this (cosine mode)
        auto_freeze: bool = True,  # Auto-freeze params after total_steps (cosine mode)
        delayed_start_multiplier: float = 0.0,  # Start training at base_warmup * this (0 = cosine mode)
        verbose: bool = True,
    ):
        self.lr_multiplier = lr_multiplier
        self.min_lr_ratio = min_lr_ratio
        self.warmup_multiplier = warmup_multiplier
        self.total_steps_multiplier = total_steps_multiplier
        self.auto_freeze = auto_freeze
        self.delayed_start_multiplier = delayed_start_multiplier
        self.verbose = verbose
        self.lazy_param_group_idx = None
        self.initialized = False
        self.frozen = False  # For cosine mode auto-freeze
        self.started = False  # For delayed mode - whether training has started
        self.initially_frozen = False  # For delayed mode - whether we froze at init
        self.last_logged_step = -100

    def _find_lazy_param_group(self, optimizer):
        """Find the parameter group index for lazy params."""
        if len(optimizer.param_groups) >= 2:
            return 1  # Lazy params are in group 1
        return None

    def _cosine_schedule(self, step, warmup_steps, total_steps, max_lr, min_lr):
        """Compute LR using cosine schedule with warmup."""
        import math

        if step < warmup_steps:
            # Linear warmup
            return max_lr * step / max(warmup_steps, 1)
        elif step >= total_steps:
            # After schedule completes, stay at min_lr
            return min_lr
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    def _freeze_lazy_params(self, model):
        """Freeze learnable_bias_diagonals and tau in all layers."""
        frozen_count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'learnable_bias_diagonals'):
                module.learnable_bias_diagonals.requires_grad = False
                frozen_count += 1
            if hasattr(module, 'tau'):
                module.tau.requires_grad = False
                frozen_count += 1
        return frozen_count

    def _unfreeze_lazy_params(self, model):
        """Unfreeze learnable_bias_diagonals and tau in all layers."""
        unfrozen_count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'learnable_bias_diagonals'):
                module.learnable_bias_diagonals.requires_grad = True
                unfrozen_count += 1
            if hasattr(module, 'tau'):
                module.tau.requires_grad = True
                unfrozen_count += 1
        return unfrozen_count

    def on_step_begin(
        self,
        args,
        state,
        control,
        model=None,
        **kwargs
    ):
        """Update lazy LR at each step according to schedule mode."""
        optimizer = kwargs.get('optimizer', None)
        if optimizer is None:
            return

        # Find lazy param group on first call
        if self.lazy_param_group_idx is None:
            self.lazy_param_group_idx = self._find_lazy_param_group(optimizer)

        if self.lazy_param_group_idx is None:
            return

        base_lr = args.learning_rate
        lazy_max_lr = base_lr * self.lr_multiplier

        # ============================================================
        # Mode 2: Delayed Constant LR
        # ============================================================
        if self.delayed_start_multiplier > 0:
            start_step = int(args.warmup_steps * self.delayed_start_multiplier)

            # Initialize: freeze params at the very beginning for fast backward
            if not self.initialized:
                if model is not None:
                    unwrapped = _unwrap_model(model)
                    frozen_count = self._freeze_lazy_params(unwrapped)
                    self.initially_frozen = True
                    if self.verbose and _is_main_process():
                        print(
                            f"\n[FastCosineScheduler] DELAYED MODE:"
                            f"\n  - Lazy params FROZEN for steps 0-{start_step-1} (fast backward)"
                            f"\n  - At step {start_step}: UNFREEZE, constant LR = {lazy_max_lr:.2e} ({self.lr_multiplier}x base)"
                            f"\n  - Train until end (no auto-freeze)",
                            flush=True
                        )
                # Set LR to 0 while frozen
                optimizer.param_groups[self.lazy_param_group_idx]['lr'] = 0.0
                self.initialized = True

            # Check if it's time to unfreeze and start training
            if not self.started and state.global_step >= start_step:
                if model is not None:
                    unwrapped = _unwrap_model(model)
                    unfrozen_count = self._unfreeze_lazy_params(unwrapped)
                    self.started = True
                    if self.verbose and _is_main_process():
                        print(
                            f"\n[FastCosineScheduler] Step {state.global_step}: "
                            f"UNFROZE {unfrozen_count} lazy params (bias/tau). "
                            f"Starting training with constant LR = {lazy_max_lr:.2e}",
                            flush=True
                        )

            # Set LR based on whether training has started
            if self.started:
                # Constant max LR after start
                optimizer.param_groups[self.lazy_param_group_idx]['lr'] = lazy_max_lr
            else:
                # LR = 0 before start (params are frozen anyway)
                optimizer.param_groups[self.lazy_param_group_idx]['lr'] = 0.0

            # Log occasionally
            if self.verbose and _is_main_process() and self.started:
                if state.global_step - self.last_logged_step >= 500:
                    base_lr_current = optimizer.param_groups[0]['lr']
                    print(
                        f"\n[FastCosineScheduler] Step {state.global_step}: "
                        f"base_lr={base_lr_current:.2e}, lazy_lr={lazy_max_lr:.2e} (constant)",
                        flush=True
                    )
                    self.last_logged_step = state.global_step
            return

        # ============================================================
        # Mode 1: Fast Cosine with Auto-Freeze (original behavior)
        # ============================================================
        # Skip if already frozen
        if self.frozen:
            return

        lazy_min_lr = lazy_max_lr * self.min_lr_ratio
        lazy_warmup_steps = int(args.warmup_steps * self.warmup_multiplier)
        lazy_total_steps = int(lazy_warmup_steps * self.total_steps_multiplier)

        # Auto-freeze after total_steps
        if self.auto_freeze and state.global_step >= lazy_total_steps:
            if model is not None:
                unwrapped = _unwrap_model(model)
                frozen_count = self._freeze_lazy_params(unwrapped)
                self.frozen = True
                # Set LR to 0 for lazy param group
                optimizer.param_groups[self.lazy_param_group_idx]['lr'] = 0.0
                if self.verbose and _is_main_process():
                    print(
                        f"\n[FastCosineScheduler] Step {state.global_step}: "
                        f"AUTO-FROZEN {frozen_count} lazy params (bias/tau). "
                        f"No more gradient updates for these params.",
                        flush=True
                    )
            return

        # Compute lazy LR using faster cosine schedule
        lazy_lr = self._cosine_schedule(
            state.global_step,
            lazy_warmup_steps,
            lazy_total_steps,
            lazy_max_lr,
            lazy_min_lr
        )

        # Update lazy param group LR
        optimizer.param_groups[self.lazy_param_group_idx]['lr'] = lazy_lr

        # Log occasionally
        if self.verbose and _is_main_process():
            if not self.initialized:
                print(
                    f"\n[FastCosineScheduler] COSINE MODE:"
                    f"\n  - Peak LR: {lazy_max_lr:.2e} ({self.lr_multiplier}x base)"
                    f"\n  - Min LR: {lazy_min_lr:.2e}"
                    f"\n  - Warmup: {lazy_warmup_steps} steps ({self.warmup_multiplier}x base {args.warmup_steps})"
                    f"\n  - Total: {lazy_total_steps} steps ({self.total_steps_multiplier}x warmup, then FREEZE)",
                    flush=True
                )
                self.initialized = True

            # Log at key milestones
            if (state.global_step in [1, lazy_warmup_steps, lazy_total_steps] or
                state.global_step - self.last_logged_step >= 500):
                base_lr_current = optimizer.param_groups[0]['lr']
                print(
                    f"\n[FastCosineScheduler] Step {state.global_step}: "
                    f"base_lr={base_lr_current:.2e}, lazy_lr={lazy_lr:.2e} "
                    f"({lazy_lr/base_lr_current:.1f}x)",
                    flush=True
                )
                self.last_logged_step = state.global_step
