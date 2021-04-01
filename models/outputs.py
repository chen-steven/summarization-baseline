from dataclasses import dataclass

from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutputWithPastAndCrossAttentions
import torch

@dataclass
class ExtractorAbstractorOutput(Seq2SeqLMOutput):
    gumbel_output: torch.FloatTensor = None
    extracted_attentions: torch.FloatTensor=None

@dataclass
class ExtractorModelOutput(BaseModelOutputWithPastAndCrossAttentions):
    input_ids: torch.FloatTensor = None
    masked_hidden_states: torch.FloatTensor = None
    new_attention_mask: torch.FloatTensor = None
    new_hidden_states: torch.FloatTensor = None