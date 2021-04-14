from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from models.outputs import ExtractorAbstractorOutput
from models.t5_extractor_base import ExtractorBaseT5
from models.t5_extractor_base import T5ExtractorEncoder, ExtractorModelOutput
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

class EncoderWrapper(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
    def forward(input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            encoder_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):
#        print(input_ids.size())
#        print(attention_mask.size())
        outputs = self.encoder(
            input_ids=input_ids.view(-1, input_ids.size(-1)),
            attention_mask=attention_mask.view(-1, attention_mask.size(-1)),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            encoder_head_mask=encoder_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs

class MeanSumT5(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizers = [AutoTokenizer.from_pretrained('t5-small') for _ in range(4)]
#        self.encoder_wrapper = EncoderWrapper(config, self.encoder)

#    def get_encoder(self):
#        return self.encoder_wrapper

    def _reconstruction(self, reconstruction_input_ids, attention_mask, encoder_hidden_states):
        decoder_input_ids = self._shift_right(reconstruction_input_ids)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states.mean(1).unsqueeze(1),
#            encoder_attention_mask=attention_mask,
        )
        return decoder_outputs

    def greedy_decode(self, input_ids, encoder_hidden_states, encoder_attention_mask=None):
        input_ids = self._prepare_decoder_input_ids_for_generation(input_ids,
                                                                   decoder_start_token_id=self.config.decoder_start_token_id)

        cur_len = 0
        while cur_len < 100:
            decoder_outputs = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states.unsqueeze(1),

            )
            sequence_output = decoder_outputs[0]
            logits = self.lm_head(sequence_output)
            next_token_logits = logits[:, -1, :]
            gumbel = F.gumbel_softmax(next_token_logits, hard=True, dim=-1)
            indices = torch.arange(gumbel.size(1)).unsqueeze(0).cuda()

            new_tokens = (gumbel * indices).long().sum(-1)[:, None]
            # new_tokens = torch.argmax(next_token_logits, dim=-1)[:,None]
            input_ids = torch.cat((input_ids, new_tokens), dim=-1)
            cur_len += 1
#        print(self.tokenizers[0].batch_decode(input_ids))
        return input_ids

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        if self.training: labels=None
        flattened_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None

        flattened_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=flattened_input_ids,
                attention_mask=flattened_attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, ExtractorModelOutput):
            encoder_outputs = ExtractorModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        reshaped_hidden_states = hidden_states.view(4, 5, 512, -1) # (batch, num_docs, seq_len, d)

        pooled_hidden_states = reshaped_hidden_states.mean(2) # (batch, num_docs, d)
        pooled_document_hidden_states = pooled_hidden_states.mean(1) # (batch, d)

        if self.training:
            generated_summary = self.greedy_decode(input_ids, pooled_document_hidden_states)
            encoded_summary = self.encoder(generated_summary, attention_mask=(generated_summary != 0).long())
            reconstruction_labels = flattened_input_ids * flattened_attention_mask + (-100) * (1 - flattened_attention_mask)
            sequence_output = self._reconstruction(reconstruction_labels, flattened_attention_mask, hidden_states)[0]
        else:
            if labels is not None:
                decoder_input_ids = self._shift_right(labels)
            sequence_output = self.decoder(decoder_input_ids, encoder_hidden_states=pooled_document_hidden_states.unsqueeze(1))[0]
            
            

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if not self.training and labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)



        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            self.sentence_classifier = self.sentence_classifier.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), reconstruction_labels.view(-1) if self.training else labels.view(-1)) if self.training else torch.tensor(0.)
        if self.training:
            sim_loss_fct = nn.CosineSimilarity(2)
            loss -= sim_loss_fct(encoded_summary[0].mean(1).unsqueeze(1), pooled_hidden_states).mean()
        
        return ExtractorAbstractorOutput(
            loss=loss,
            logits=lm_logits,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, decoder_sentence_indicator=None, decoder_sentence_labels=None,
            past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # no need to pass input ids because encoder outputs is already computed from a prepare inputs for generation method
        res = super().prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask,
                                                    use_cache=use_cache, encoder_outputs=encoder_outputs, **kwargs)

        return res
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
            }
            input_ids = input_ids.view(-1, input_ids.size(-1))
            encoder_kwargs['attention_mask'] = encoder_kwargs['attention_mask'].view(-1, encoder_kwargs['attention_mask'].size(-1))
            model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids, return_dict=True, **encoder_kwargs)

#            del encoder_kwargs['attention_mask']
        return model_kwargs
