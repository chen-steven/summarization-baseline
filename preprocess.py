from dataclasses import dataclass

from datasets import load_dataset
from nltk import sent_tokenize
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
import json
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import multiprocess as mp
from transformers.tokenization_utils_base import PaddingStrategy


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
        sentence_labels = [feature["sentence_labels"] for feature in features]
        sentence_indicators = [feature['sentence_indicator'] for feature in features]
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
        for feature in features:
            remainder = [feature['sentence_indicator'][-1]]*(max_sentence_indicator_length - len(feature['sentence_indicator']))
            feature['sentence_indicator'] = feature['sentence_indicator'] + remainder
        max_sentence_label_length = max(len(l) for l in sentence_labels)
        max_sentence_label_length = max(max_sentence_label_length, 5)
#        sentence_indicators = [feature['sentence_indicator'] for feature in features]
#        sentence_pad_id = max(l[-1] for l in sentence_indicators)
        for feature in features:
            remainder = [self.sentence_label_pad_id] * (max_sentence_label_length - len(feature['sentence_labels']))
            feature['sentence_labels'] = feature['sentence_labels'] + remainder

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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=int, default=1)
    args = parser.parse_args()
    preprocess_cnn(args)
