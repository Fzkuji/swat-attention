# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.swat.configuration_swat import SWATConfig
from fla.models.swat.modeling_swat import (
    SWATForCausalLM, SWATModel)

AutoConfig.register(SWATConfig.model_type, SWATConfig)
AutoModel.register(SWATConfig, SWATModel)
AutoModelForCausalLM.register(SWATConfig, SWATForCausalLM)


__all__ = ['SWATConfig', 'SWATForCausalLM', 'SWATModel']
