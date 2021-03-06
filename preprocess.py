from dataclasses import dataclass
from datasets import load_dataset
from nltk import sent_tokenize
from rouge_score import rouge_scorer
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel, AutoConfig
import json
import pickle
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import multiprocess as mp
from transformers.tokenization_utils_base import PaddingStrategy
import os
import numpy as np

@dataclass
class DataCollatorForExtractorAbstractor:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`
            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    sentence_label_pad_id: int = -1


    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        
        sentence_labels = [feature["sentence_labels"] for feature in features] if "sentence_labels" in features[0].keys() else None
        sentence_indicators = [feature['sentence_indicator'] for feature in features]
        reference_sentence_indicators = [feature['reference_sentence_indicator'] for feature in features] if 'reference_sentence_indicator' in features[0].keys() else None
        pmi_features = [feature['pmi_features'] for feature in features] if 'pmi_features' in features[0].keys() else None

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        max_sentence_indicator_length = max(len(l) for l in sentence_indicators)
        max_sentences = max(max(l) for l in sentence_indicators)+1
        for feature in features:
            remainder = [feature['sentence_indicator'][-1]]*(max_sentence_indicator_length - len(feature['sentence_indicator']))
            feature['sentence_indicator'] = feature['sentence_indicator'] + remainder

        if reference_sentence_indicators is not None:
            max_ref_sentence_indicator_length = max(len(l) for l in reference_sentence_indicators)
            for feature in features:
                remainder = [feature['reference_sentence_indicator'][-1]]*(max_ref_sentence_indicator_length - len(feature['reference_sentence_indicator']))
                feature['reference_sentence_indicator'] = feature['reference_sentence_indicator'] + remainder
#        max_sentence_label_length = max(len(l) for l in sentence_labels)
#        max_sentence_label_length = max(max_sentence_label_length, 5)
#        sentence_indicators = [feature['sentence_indicator'] for feature in features]
#        sentence_pad_id = max(l[-1] for l in sentence_indicators)
        if sentence_labels is not None:
            max_sentence_label_length = max(len(l) for l in sentence_labels)
            max_sentence_label_length = max(max_sentence_label_length, 5)
            for feature in features:
                remainder = [self.sentence_label_pad_id] * (max_sentence_label_length - len(feature['sentence_labels']))
                feature['sentence_labels'] = feature['sentence_labels'] + remainder

        if pmi_features is not None:
            for feature in features:
                feature['pmi_features'] = feature['pmi_features'][:max_sentences]
                remainder = [0]*(max_sentences - len(feature['pmi_features']))
                feature['pmi_features'] = feature['pmi_features'] + remainder

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

def _one_example(source, target, scorer, metric, tokenizer, max_length=200):
    # source = source.replace("|||", ' ')
    sents = sent_tokenize(source)
    summary_tokens, summary = [], [(-1, '')]
    indices = []
    max_rouge2 = -1
    while len(summary_tokens) < 200:
        max_sent_index, max_sent = -1, None
        indexs = [j for j, _ in summary]
        for i, sent in enumerate(sents):
            if i in indexs:
                continue
            tmp_summary = summary + [(i, sent)]
            tmp_summary = ' '.join([s for _, s in sorted(tmp_summary, key=lambda x: x[0])])
            if "rouge" in metric:
                res = scorer.score(target=target, prediction=tmp_summary)
                res = {metric: res[metric].fmeasure * 100}
            else:
                print("wrong metric!")
                return
            rouge2 = res[metric]
            if rouge2 > max_rouge2:
                max_rouge2 = rouge2
                max_sent_index = i
                max_sent = sent
        if max_sent_index == -1:
            break
        summary_tokens.extend(tokenizer.tokenize(max_sent))
        summary.append((max_sent_index, max_sent))
        indices.append(max_sent_index)

    # summary = ' '.join([s for _, s in sorted(summary, key=lambda x: x[0])])
    # summary_tokens = tokenizer.tokenize(summary)[:max_length]
    # summary = tokenizer.convert_tokens_to_string(summary_tokens)

    return sorted(indices)

def transform_dataset(examples, tokenizer):
    sentence_labels = {}
    inputs = examples['article']
    targets = examples['highlights']
    ids = examples['id']


    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    sentence_indices = []
    for _id, inp, target in tqdm(zip(ids, inputs,targets)):
        indices = _one_example(inp, target, scorer, 'rouge1', tokenizer)
        sentence_labels[_id] = indices

    print(len(sentence_labels))
    return sentence_labels


def preprocess_cnn(args):
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    dataset = load_dataset('cnn_dailymail', "3.0.0")

    train_data = dataset['train']
    val_data = dataset['validation']
    test_data = dataset['test']

    examples = test_data

    num_splits = 4
    split = len(test_data)//num_splits
    #data = [examples[:split], examples[split:2*split], examples[2*split:3*split], examples[3*split:4*split]]

    pool = mp.Pool(processes=4)
    results = [pool.apply_async(transform_dataset, args=(examples[i*split:(i+1)*split], tokenizer)) for i in range(num_splits)]
    outputs = [p.get() for p in results]
    d = {}
    for x in outputs:
        d = {**d, **x}
    json.dump(d, open('test_sentence_labels.json', 'w'))


    #transform_dataset(args, test_data, tokenizer)


# def oracle(source, target, output, max_length=200):
#
#     with open(source, 'r') as fs, open(target, 'r') as ft:
#         source = fs.readlines()
#         target = ft.readlines()
#     examples = [(s, t) for s, t in zip(source, target)]
#     best = {}
#     for metric in ["rouge1"]:
#         scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
#         res = []
#         for s, t in tqdm(examples):
#             res.append(_one_example(s, t, scorer, metric))
#         if "rouge" in metric:
#             score, score_str = rouge(target, res, score_keys=[metric])
#             best.update(score)
#         else:
#             score, score_str = bertScore(target, res)
#             best.update(score)
#         with open(f"{output}_{metric}_{max_length}.res", 'w') as f:
#             f.write('\n'.join(res))
#     print(best)

def _create_sentence_indicator(input_ids, tokenizer, sep_token_id=1, sentence_labels=None, index_map=None):
    sentence_indicators = []
    for idx, cur_input_id in enumerate(input_ids):
        sent_count = 0
        cur_indicator = [0]*len(cur_input_id)
        for i, ids in enumerate(cur_input_id):
            if ids == sep_token_id:
                sent_count += 1
            cur_indicator[i] = sent_count
            
        if sentence_labels is not None:
            sentence_labels[idx] = [x for x in sentence_labels[idx] if x < sent_count]

        sep_ids = [j for j, x in enumerate(cur_input_id) if x == sep_token_id][:-1]

        # remove temp sep tokens
        for i in sep_ids[::-1]:
            del cur_input_id[i]
            del cur_indicator[i]
            cur_input_id.append(tokenizer.pad_token_id)
            cur_indicator.append(sent_count)

        if index_map is not None:
            i_map = index_map[idx]
            mapped_cur_indicator = [i_map[x] if x in i_map else x for x in cur_indicator]
            sentence_indicators.append(mapped_cur_indicator)
        else:
            sentence_indicators.append(cur_indicator)

    return sentence_indicators

def _preprocess_denoise_train(examples, tokenizer, max_length, article_column):
    ids = examples['id']
    articles = examples[article_column]
    article_map = {ids[i]: articles[i] for i in range(len(ids))}
#    l = [key for key in article_map]

#    noisy_text = pickle.load(open('train_noise_data.pkl', 'rb'))
    paraphrased_text = pickle.load(open('data_augmentation/ppdb_paraphrase.pkl', 'rb'))
    processed_pmi_features = torch.load('data/train_pmi.pt')
    noisy_text_sentences = []
    clean_text_sentences = []
    pmi_features = []

    for _id in ids:
        #noisy_sentences = noisy_text[_id] if _id in noisy_text else sent_tokenize(article_map[_id])
        #noisy_sentences = sent_tokenize(article_map[_id])
        clean_sentences = sent_tokenize(article_map[_id])
        noisy_sentences = paraphrased_text[_id] if _id in paraphrased_text else sent_tokenize(article_map[_id])
       
        clean_text_sentences.append(clean_sentences)
        noisy_text_sentences.append(noisy_sentences)
        pmi_features.append(processed_pmi_features[_id])

    sep_token = "</s>"
    sep_token_id = 1
    clean_input = [f" {sep_token} ".join(s) for s in clean_text_sentences]
    noised_input = [f" {sep_token} ".join(s) for s in noisy_text_sentences]

    clean_model_input = tokenizer(clean_input, max_length=max_length, padding="max_length", truncation=True)
    noised_model_input = tokenizer(noised_input, max_length=max_length, padding="max_length", truncation=True)

    sent_lens = []
    for ii in noised_model_input['input_ids']:
        sent_lens.append(sum(1 for x in ii if x == sep_token_id))

    index_maps = []
    shuffled_noisy_sentences = []
    for idx, sent in enumerate(noisy_text_sentences):
        shuffle_to = sent_lens[idx]-1
        untruncated_sentences = sent[:shuffle_to]
        tmp_sentences = [(s, i) for i, s in enumerate(untruncated_sentences)]

        np.random.shuffle(tmp_sentences)
#        idx_map = {tup[1]: i for i, tup in enumerate(tmp_sentences)}
        idx_map = {i: tup[1] for i, tup in enumerate(tmp_sentences)}
        index_maps.append(idx_map)
        
        shuffled_noisy_sentences.append([tup[0] for tup in tmp_sentences] + sent[shuffle_to:])

    shuffled_noised_input = [f" {sep_token} ".join(s) for s in shuffled_noisy_sentences]
    shuffled_noised_model_input = tokenizer(shuffled_noised_input, max_length=max_length, padding="max_length", truncation=True)

    sentence_indicator_clean = _create_sentence_indicator(clean_model_input['input_ids'], tokenizer, sep_token_id)
                                                          #index_map=index_maps)
    sentence_indicator_noise = _create_sentence_indicator(noised_model_input['input_ids'], tokenizer, sep_token_id)

    shuffled_sentence_indicator_noise = _create_sentence_indicator(shuffled_noised_model_input['input_ids'], tokenizer, sep_token_id, index_map=index_maps)

    noised_model_input['sentence_indicator'] = sentence_indicator_noise
    noised_model_input['reference_input_ids'] = clean_model_input['input_ids']
    noised_model_input['shuffled_input_ids'] = shuffled_noised_model_input['input_ids']
    noised_model_input['shuffled_sentence_indicator'] = shuffled_sentence_indicator_noise
    noised_model_input['reference_sentence_indicator'] = sentence_indicator_clean
    noised_model_input['pmi_features'] = pmi_features
    targets = examples['highlights']
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=200, padding="max_length", truncation=True)#

    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    noised_model_input['labels'] = labels['input_ids']

    return noised_model_input


def _preprocess_denoise_eval(examples, tokenizer, max_length, max_target_length, article_column, summary_column):
    inputs = examples[article_column]
    targets = examples[summary_column]
    ids = examples['id']

    sep_token = "</s>"
    sep_token_id = 1
    new_input = [f" {sep_token} ".join(sent_tokenize(inp)) for inp in inputs]
    model_inputs = tokenizer(new_input, max_length=max_length, padding="max_length", truncation=True)

    sentence_label_map = json.load(open(os.path.join("data", 'val_sentence_labels.json'), 'r'))
    sentence_labels = [sentence_label_map[i] for i in ids]

    processed_pmi_features = torch.load('data/val_pmi.pt')
    pmi_features = [processed_pmi_features[i] for i in ids]

    #creates sentence indicator AND update sentence_labels inplace (ensures labels are within sentence count)
    sentence_indicator = _create_sentence_indicator(model_inputs['input_ids'], tokenizer, sep_token_id, sentence_labels) 

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding="max_length", truncation=True)
    
    labels["input_ids"] = [
         [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    model_inputs['pmi_features'] = pmi_features
    model_inputs["real_input_ids"] = model_inputs["input_ids"]
    model_inputs['sentence_indicator'] = sentence_indicator
    model_inputs['sentence_labels'] = sentence_labels
    return model_inputs


def preprocess_denoise(examples, tokenizer, max_length, max_target_length, split, article_column, summary_column):
    if split == "train":
        return _preprocess_denoise_train(examples, tokenizer, max_length, article_column)
    else:
        return _preprocess_denoise_eval(examples, tokenizer, max_length, max_target_length, article_column, summary_column)


if __name__ == '__main__':
    import torch
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    train_examples = dataset['train'][:10]
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    max_length=1024
    max_target_length=200
    from models import UnsupervisedDenoiseT5
    config = AutoConfig.from_pretrained('t5-small')
    config.sequential_extraction=True
    config.teacher_forcing=False
    config.extraction_k=5
    res = preprocess_denoise(train_examples, tokenizer, max_length, max_target_length, 'train')
    for key in res: print(key)
    model = UnsupervisedDenoiseT5.from_pretrained('t5-small', config=config)
    inputs = {key: torch.tensor(val) for key, val in res.items()}
    model(**inputs)


    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--split', type=int, default=1)
    # args = parser.parse_args()
    # preprocess_cnn(args)
