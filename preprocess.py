from datasets import load_dataset
from nltk import sent_tokenize
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoTokenizer
import json

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

    json.dump(sentence_labels, open('test_sentence_labels.json', 'w'))





def preprocess_cnn():
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    dataset = load_dataset('cnn_dailymail', "3.0.0")

    train_data = dataset['train']
    val_data = dataset['validation']
    test_data = dataset['test']

    transform_dataset(test_data, tokenizer)


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
    preprocess_cnn()
