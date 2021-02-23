from datasets import load_dataset
from nltk import sent_tokenize
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoTokenizer
import json
import multiprocess as mp

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
