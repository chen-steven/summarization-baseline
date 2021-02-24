from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from models.outputs import ExtractorAbstractorOutput


class ExtractorAbstractorT5(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.sentence_classifier = nn.Linear(config.d_model, 2)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            sentence_indicator=None,
            sentence_labels=None,
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # extract salient sentences
        sentences = []
        for i in range(sentence_indicator.max()+1):
            mask = (sentence_indicator == i).long().cuda()
            sentences.append(
                torch.sum(hidden_states * mask.unsqueeze(-1), dim=1) / (mask.sum(dim=1).view(-1, 1) + 1e-12))

        sentences = torch.stack(sentences, dim=1)
        sentence_logits = self.sentence_classifier(sentences)

        sentence_label_one_hot = utils.convert_one_hot(sentence_labels, sentence_logits.size(1)).detach()

        if self.training:
            print(sentence_labels)
            print(sentence_logits.size())
            gumbel_output = utils.convert_one_hot(sentence_labels, sentence_logits.size(1))
        else:
            gumbel_output = utils.gumbel_softmax_topk(sentence_logits, 5, hard=True, dim=-1)
        gumbel_output = F.gumbel_softmax(sentence_logits, hard=True, dim=-1)[:, :, 1]

#        print(sentence_labels)
 #       print(gumbel_output)
        new_attention_mask = utils.convert_attention_mask(sentence_indicator, gumbel_output)
        masked_hidden_states = new_attention_mask.unsqueeze(-1) * hidden_states

        #create sentence mask using

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
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

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=masked_hidden_states,
            encoder_attention_mask=new_attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

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
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            sentence_loss_fct = nn.BCEWithLogitsLoss()
            #convert sentence_label to one_hot vector with the same size as the sentence logits
            #compute loss
            loss += sentence_loss_fct(sentence_logits, sentence_label_one_hot.float())

            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return ExtractorAbstractorOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            gumbel_output=gumbel_output
        )

    def prepare_inputs_for_generation(
            self, input_ids, decoder_sentence_indicator=None, decoder_sentence_labels=None, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # no need to pass input ids because encoder outputs is already computed from a prepare inputs for generation method
        res = super().prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, encoder_outputs=encoder_outputs, **kwargs)
        res['sentence_indicator'] = decoder_sentence_indicator
        res['sentence_labels'] = decoder_sentence_labels
        return res
