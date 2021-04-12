from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from models.outputs import ExtractorAbstractorOutput
from models.t5_extractor_base import ExtractorBaseT5
from models.t5_extractor_base import T5ExtractorEncoder, ExtractorModelOutput

class UnsupervisedDenoiseT5(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.sentence_classifier = nn.Linear(config.d_model + 1 if config.use_pmi else config.d_model, 1)
        self.attention_dropout = utils.NonInvertedDropout(0.6)
        self.tokenizers = [AutoTokenizer.from_pretrained('t5-small') for _ in range(4)]
        self.encoder_wrapper = T5ExtractorEncoder(config, self.encoder, self.sentence_classifier)

    def get_encoder(self):
        return self.encoder_wrapper

    def selection_step(self, cur_sum, cur_len, sentence_sums, sentence_lens, sentence_mask, sentence_label=None):
        combined_sentence_embeddings = cur_sum.unsqueeze(1) + sentence_sums
        combined_len = cur_len.unsqueeze(1) + sentence_lens

        pooled_embeddings = combined_sentence_embeddings / combined_len
        sentence_logits = self.sentence_classifier(pooled_embeddings).squeeze(-1)
        sentence_logits = utils.mask_tensor(sentence_logits, sentence_mask.detach())

        num_sentences = combined_sentence_embeddings.size(1)
        if self.training:
            if self.config.teacher_forcing and sentence_label is not None:
                one_hot = utils.convert_single_one_hot(sentence_label, num_sentences)
            else:
                one_hot = F.gumbel_softmax(sentence_logits, hard=True)
        else:
            one_hot = torch.argmax(sentence_logits, -1)
            one_hot = utils.convert_single_one_hot(one_hot, num_sentences)

        sentence_mask = (1 - one_hot) * sentence_mask
        one_hot = one_hot.unsqueeze(-1)

        new_embedding = (one_hot * combined_sentence_embeddings).sum(dim=1)
        new_len = (one_hot * combined_len).sum(dim=1)

        return sentence_logits, new_embedding, new_len, sentence_mask, one_hot.squeeze(-1)

    def selection_loop(self, hidden_states, sentence_indicator, sentence_labels, pmi_features=None):
        all_sentence_logits = []
        sentences = []
        sentence_lens = []
        for i in range(sentence_indicator.max() + 1):
            mask = (sentence_indicator == i).long().cuda()

            sentence_embedding = torch.sum(hidden_states * mask.unsqueeze(-1), dim=1)
            sentence_len = mask.sum(dim=1).view(-1, 1)
            sentences.append(sentence_embedding)
            sentence_lens.append(sentence_len)

        sentences = torch.stack(sentences, dim=1)
        sentence_lens = torch.stack(sentence_lens, dim=1)
        sentence_lens = sentence_lens.clamp(min=1)

        if self.config.use_pmi and pmi_features is not None:
            pmi_features = pmi_features[:, :sentence_indicator.max()+1].unsqueeze(-1)

            sentences = torch.cat((sentences, pmi_features), dim=-1)
        #        zero_len_mask = sentence_lens == 0
        #        sentence_lens = sentence_lens + zero_len_mask.float()

        cur_embedding = torch.zeros(sentences.size(0), sentences.size(-1)).cuda()
        cur_len = torch.zeros(sentence_lens.size(0), sentence_lens.size(-1)).cuda()

        selected_one_hot = torch.zeros(sentences.size(0), sentences.size(1)).cuda()
        selected_one_hot1 = torch.zeros(sentences.size(0), sentences.size(1)).cuda()
        sentence_mask = utils.get_sentence_mask(sentence_indicator, sentences.size(1)).float()

        for i in range(self.config.extraction_k):
            sentence_logits, cur_embedding, cur_len, sentence_mask, one_hot = self.selection_step(cur_embedding,
                                                                                                  cur_len,
                                                                                                  sentences,
                                                                                                  sentence_lens,
                                                                                                  sentence_mask,
                                                                                                  sentence_labels[:,
                                                                                                  i] if sentence_labels is not None else None)

            if i < 3:
                selected_one_hot1 = selected_one_hot1 + one_hot
            selected_one_hot = selected_one_hot + one_hot
            all_sentence_logits.append(sentence_logits)
        selected_one_hot = selected_one_hot.clamp(max=1)
        return selected_one_hot, selected_one_hot1, all_sentence_logits

    def single_extraction(self, hidden_states, sentence_indicator, sentence_labels):
        # extract salient sentences
        sentences = []
        for i in range(sentence_indicator.max() + 1):
            mask = (sentence_indicator == i).long().cuda()
            sentences.append(
                torch.sum(hidden_states * mask.unsqueeze(-1), dim=1) / (mask.sum(dim=1).view(-1, 1) + 1e-12))

        sentences = torch.stack(sentences, dim=1)

        sentence_logits = self.sentence_classifier(sentences)
        sentence_logits = utils.mask_sentences(sentence_logits, sentence_indicator)

        if self.training:
            if self.config.teacher_forcing:
                gumbel_output = utils.convert_one_hot(sentence_labels, sentence_logits.size(1))
            else:
                gumbel_output = utils.gumbel_softmax_topk(sentence_logits.squeeze(-1), self.config.extraction_k)
        else:
            # gumbel_output = utils.gumbel_softmax_topk(sentence_logits, 5, hard=True, dim=-1)
            #            gumbel_output = F.gumbel_softmax(sentence_logits, hard=True, dim=-1)[:, :, 1]
            # gumbel_output = utils.convert_one_hot(sentence_labels, sentence_logits.size(1))
            #            gumbel_output = torch.argmax(sentence_logits, -1)
            #            gumbel_output = (torch.sigmoid(sentence_logits) > 0.5).float().squeeze(-1)
            gumbel_output = torch.topk(sentence_logits.squeeze(-1), self.config.extraction_k, dim=-1)[1]
            gumbel_output = utils.convert_one_hot(gumbel_output, sentence_logits.size(1))

        return gumbel_output, sentence_logits

    def _get_extractive_summary(self, reference_input_ids, reference_sentence_indicator, gumbel_output):
        tokenizer = self.tokenizers[reference_input_ids.device.index]
        ref_max = reference_sentence_indicator.max()
        if ref_max >= gumbel_output.size(1):
            pad = torch.zeros(reference_sentence_indicator.size(0), ref_max + 1 - gumbel_output.size(1)).cuda()
            gumbel_output = torch.cat((gumbel_output, pad), -1)

        attention_mask = utils.convert_attention_mask(reference_sentence_indicator, gumbel_output).long().detach()
        extractive_summary_ids = reference_input_ids*attention_mask + (1-attention_mask)*tokenizer.pad_token_id

        extractive_summary = tokenizer.batch_decode(extractive_summary_ids, skip_special_tokens=True)
#        print('CLEAN:', extractive_summary)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(extractive_summary, max_length=200, padding="max_length", truncation=True)
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        return torch.tensor(labels['input_ids']).cuda(), torch.tensor(labels['attention_mask']).cuda()

    def forward(
            self,
            input_ids=None,
            reference_input_ids=None,
            shuffled_input_ids=None,
            pmi_features=None,
            attention_mask=None,
            sentence_indicator=None,
            reference_sentence_indicator=None,
            shuffled_sentence_indicator=None,
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
        elif return_dict and not isinstance(encoder_outputs, ExtractorModelOutput):
            encoder_outputs = ExtractorModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if self.training or not isinstance(encoder_outputs, ExtractorModelOutput):
            hidden_states = encoder_outputs[0]
            #hidden_states_non_pad = attention_mask.unsqueeze(-1)*hidden_states
            tokenizer = self.tokenizers[hidden_states.device.index]

            # extract salient sentences
            if self.config.sequential_extraction:
                gumbel_output, gumbel_output1, all_sentence_logits = self.selection_loop(hidden_states, sentence_indicator, sentence_labels, pmi_features)
            else:
                gumbel_output, sentence_logits = self.single_extraction(hidden_states, sentence_indicator, sentence_labels)

            new_attention_mask = utils.convert_attention_mask(shuffled_sentence_indicator if self.training else sentence_indicator, gumbel_output)
            original_selected_attention_mask = utils.convert_attention_mask(sentence_indicator, gumbel_output)
#            masked_hidden_states = new_attention_mask.unsqueeze(-1) * detached_hidden_states
            masked_hidden_states = original_selected_attention_mask.unsqueeze(-1) * hidden_states
            non_masked_hidden_states = (1-original_selected_attention_mask).unsqueeze(-1)*hidden_states

            
            selected_input_ids = input_ids * original_selected_attention_mask + (1-original_selected_attention_mask)*tokenizer.pad_token_id

#            encoded_hidden_states = self.encoder(selected_input_ids.long(), attention_mask=original_selected_attention_mask.long())[0]
            new_attention_mask = new_attention_mask.long()
            new_input_ids = (shuffled_input_ids if self.training else input_ids) * new_attention_mask + tokenizer.pad_token_id * (1 - new_attention_mask)
            
#            print("Shuffled", tokenizer.batch_decode(new_input_ids, skip_special_tokens=True))
            new_hidden_states = self.encoder(new_input_ids, attention_mask=new_attention_mask)[0]
        else:
            new_attention_mask = encoder_outputs.new_attention_mask
            new_hidden_states = encoder_outputs.new_hidden_states
            masked_hidden_states = encoder_outputs.masked_hidden_states
            gumbel_output = encoder_outputs.gumbel_output

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if self.training:
            labels, label_attention_mask = self._get_extractive_summary(reference_input_ids, reference_sentence_indicator, gumbel_output1)

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
            encoder_hidden_states=new_hidden_states,#masked_hidden_states,
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
            # if self.training:
            #     gumbel = F.gumbel_softmax(lm_logits, hard=True, dim=-1)
            #     indices = torch.arange(gumbel.size(-1)).view(1, 1, -1).expand(gumbel.size(0), gumbel.size(1), -1).cuda()
            #     summary = (gumbel*indices).long().sum(-1)
            #
            #     encoded_summary = self.get_encoder()(summary, attention_mask=label_attention_mask)
                        
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            sim_loss_fct = nn.CosineSimilarity()
            pooled_hidden_states = hidden_states.mean(1) #detach()?
            pooled_encoded_summary = masked_hidden_states.mean(1)
#            pooled_encoded_summary = encoded_hidden_states.mean(1)
            pooled_non_masked_hidden_states = non_masked_hidden_states.mean(1)
#            pooled_encoded_summary = new_hidden_states.mean(1)
            #pooled_encoded_summary = encoded_summary[0].mean(1) if self.training else masked_hidden_states.mean(1)
            if self.config.use_max_margin_sim_loss:
                sim_loss = self.config.max_margin - sim_loss_fct(pooled_hidden_states, pooled_encoded_summary) + sim_loss_fct(pooled_hidden_states, pooled_non_masked_hidden_states)
                loss += sim_loss.mean()
            else:
                loss -= (sim_loss_fct(pooled_hidden_states, pooled_encoded_summary)).mean()

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
            extracted_attentions=new_attention_mask,
            gumbel_output=None if self.training else gumbel_output
        )

    def prepare_inputs_for_generation(
            self, input_ids, decoder_real_input_ids=None, decoder_sentence_indicator=None, decoder_sentence_labels=None, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # no need to pass input ids because encoder outputs is already computed from a prepare inputs for generation method
        res = super().prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, encoder_outputs=encoder_outputs, **kwargs)
        #res['real_input_ids'] = decoder_real_input_ids

        res['sentence_indicator'] = decoder_sentence_indicator
        res['sentence_labels'] = decoder_sentence_labels
        return res

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids: torch.LongTensor, model_kwargs):
        m_k = super()._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)
        #m_k['real_input_ids'] = model_kwargs["decoder_real_input_ids"]
        return m_k
