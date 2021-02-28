from dataclasses import dataclass

from transformers.modeling_outputs import Seq2SeqLMOutput
import torch

@dataclass
class ExtractorAbstractorOutput(Seq2SeqLMOutput):
    gumbel_output: torch.FloatTensor = None
    extracted_attentions: torch.FloatTensor=None
