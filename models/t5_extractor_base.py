from transformers import T5ForConditionalGeneration, T5EncoderModel
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import copy
from models.outputs import ExtractorAbstractorOutput
from dataclasses import dataclass

@dataclass
class ExtractorModelOutput(BaseModelOutputWithPastAndCrossAttentions):
    input_ids: torch.FloatTensor = None
    masked_hidden_states: torch.FloatTensor = None
    new_attention_mask: torch.FloatTensor = None
    new_hidden_states: torch.FloatTensor = None
    gumbel_output: torch.FloatTensor = None


class T5ExtractorEncoder(nn.Module):
    def __init__(self, config, encoder, sentence_classifier):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.sentence_classifier = sentence_classifier

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

    def selection_loop(self, hidden_states, sentence_indicator, sentence_labels, two_selections=False):
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
            selected_one_hot = selected_one_hot + one_hot
            if i < 3:
                selected_one_hot1 = selected_one_hot1 + one_hot
            all_sentence_logits.append(sentence_logits)
        selected_one_hot = selected_one_hot.clamp(max=1)
        selectd_one_hot1 = selected_one_hot1.clamp(max=1)
        if two_selections:
            return selected_one_hot, selected_one_hot1, all_sentence_logits
        return selected_one_hot, all_sentence_logits

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
            gumbel_output = torch.topk(sentence_logits.squeeze(-1), self.config.extraction_k, dim=-1)[1]
            gumbel_output = utils.convert_one_hot(gumbel_output, sentence_logits.size(1))

        return gumbel_output, sentence_logits

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            sentence_indicator=None,
            sentence_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            encoder_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
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

        hidden_states = outputs[0]

        if self.config.sequential_extraction:
            gumbel_output, all_sentence_logits = self.selection_loop(hidden_states, sentence_indicator, sentence_labels)
        else:
            gumbel_output, sentence_logits = self.single_extraction(hidden_states, sentence_indicator, sentence_labels)

        new_attention_mask = utils.convert_attention_mask(sentence_indicator, gumbel_output).long()
        new_input_ids = input_ids * new_attention_mask + self.config.pad_token_id * (1-new_attention_mask)

        new_hidden_states = self.encoder(new_input_ids, attention_mask=new_attention_mask)[0]
        masked_hidden_states = new_attention_mask.unsqueeze(-1) * hidden_states

        return ExtractorModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
            input_ids=input_ids,
            new_attention_mask=new_attention_mask,
            masked_hidden_states=masked_hidden_states,
            new_hidden_states=new_hidden_states,
            gumbel_output = gumbel_output
        )


class ExtractorBaseT5(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.sentence_classifier = nn.Linear(config.d_model, 1)

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

        sentence_mask = (1-one_hot)*sentence_mask
        one_hot = one_hot.unsqueeze(-1)

        new_embedding = (one_hot*combined_sentence_embeddings).sum(dim=1)
        new_len = (one_hot*combined_len).sum(dim=1)

        return sentence_logits, new_embedding, new_len, sentence_mask, one_hot.squeeze(-1)

    def selection_loop(self, hidden_states, sentence_indicator, sentence_labels, two_selections=False):
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
                                                                                                  sentence_labels[:, i] if sentence_labels is not None else None)
            selected_one_hot = selected_one_hot + one_hot
            if i < 3:
                selected_one_hot1 = selected_one_hot1 + one_hot
            all_sentence_logits.append(sentence_logits)
        selected_one_hot = selected_one_hot.clamp(max=1)
        selectd_one_hot1 = selected_one_hot1.clamp(max=1)
        if two_selections:
            return selected_one_hot, selected_one_hot1, all_sentence_logits
        return selected_one_hot, all_sentence_logits

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
                #gumbel_output = utils.gumbel_softmax_topk(sentence_logits, 5, hard=True, dim=-1)
    #            gumbel_output = F.gumbel_softmax(sentence_logits, hard=True, dim=-1)[:, :, 1]
                #gumbel_output = utils.convert_one_hot(sentence_labels, sentence_logits.size(1))
    #            gumbel_output = torch.argmax(sentence_logits, -1)
    #            gumbel_output = (torch.sigmoid(sentence_logits) > 0.5).float().squeeze(-1)
                gumbel_output = torch.topk(sentence_logits.squeeze(-1), self.config.extraction_k, dim=-1)[1]
                gumbel_output = utils.convert_one_hot(gumbel_output, sentence_logits.size(1))

        return gumbel_output, sentence_logits
