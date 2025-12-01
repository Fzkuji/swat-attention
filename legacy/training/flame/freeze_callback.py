# -*- coding: utf-8 -*-
"""
Callbacks for lazy attention parameter management:
1. FreezeLazyParamsCallback - freeze bias/tau after N steps
2. MonitorLazyParamsCallback - track parameter convergence
"""

import torch
import torch.distributed as dist
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from transformers.utils import logging

logger = logging.get_logger(__name__)


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

            # Check for tau parameter
            if hasattr(module, 'tau'):
                param = module.tau
                metrics['tau_norm'].append(param.data.norm().item())
                # Get actual tau values (multiply by TAU_SCALE if using scaled-down representation)
                # tau_small * TAU_SCALE = actual_tau
                tau_scale = getattr(module, 'TAU_SCALE', 1.0)
                actual_tau = param.data.cpu() * tau_scale
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

        return result

    def _log_to_console(self, step, metrics):
        """Print metrics to console."""
        bias_norm = metrics.get('lazy/bias_norm_mean', 0)
        bias_grad = metrics.get('lazy/bias_grad_norm_mean', 0)
        bias_change = metrics.get('lazy/bias_change_rate_mean', 0)
        tau_norm = metrics.get('lazy/tau_norm_mean', 0)
        tau_grad = metrics.get('lazy/tau_grad_norm_mean', 0)
        tau_change = metrics.get('lazy/tau_change_rate_mean', 0)
        num_layers = len(metrics.get('lazy/bias_norm_mean', [])) if isinstance(metrics.get('lazy/bias_norm_mean'), list) else 0

        # Get first layer's tau values for debugging
        tau_values = metrics.get('lazy/tau_layer0', [])
        tau_str = str(tau_values[:4]) if tau_values else "N/A"  # Show first 4 heads

        # Use print for reliable output (logger.info may be filtered)
        print(
            f"\n[Step {step}] Lazy params (found {len(self.prev_bias_params)} layers): "
            f"bias(norm={bias_norm:.4f}, grad={bias_grad:.6f}, change={bias_change:.6f}) "
            f"tau(norm={tau_norm:.4f}, grad={tau_grad:.6f}, change={tau_change:.6f}) "
            f"tau_vals={tau_str}",
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
