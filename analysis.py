from transformers import AutoTokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import torch
import nltk
import json
from tqdm import tqdm


def preprocess(examples, tokenizer):
    inputs = examples['article']
    ids = examples['id']

    examples = []
    example_sentences = [nltk.sent_tokenize(inp) for inp in inputs]
    for idx, sentences in enumerate(example_sentences):
        ex = []
        lab = []

        for i in range(len(sentences)):
            inp = " ".join(sentences[:i]) + " <extra_id_0> " + " ".join(sentences[i+1:])
            label = "<extra_id_0> " + sentences[i] + " <extra_id_1>"
            ex.append(inp)
            lab.append(label)

        inp_dict = tokenizer(ex, padding=True, truncation=True, return_tensors="pt")
        inp_dict['labels'] = tokenizer(lab, padding=True, truncation=True, return_tensors="pt").input_ids
        examples.append((inp_dict, ids[idx]))

    return examples

def get_probs(model, inputs):
    print(inputs)
    logits = model(**inputs).logits
    log_probs = torch.log_softmax(logits, -1)
    labels = inputs['labels']
    selected_probs = torch.gather(log_probs, 1, labels.unsqueeze(-1)).squeeze(-1)
    sentence_prob = torch.sum(selected_probs, -1)

    return sentence_prob


def main(args):
    dataset = load_dataset('cnn_dailymail', "3.0.0")
    train_dataset = dataset['train'].select(range(args.max_train_samples))

    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    examples = preprocess(train_dataset, tokenizer)

    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    # model.to(args.device)
    model.eval()

    sentence_ranks = {}

    for inputs, ids in tqdm(examples):
        # for key in inputs:
        #     inputs[key] = inputs[key].to(args.device)
        sentence_probs = get_probs(model, inputs)
        _, idxs = sentence_probs.sort()
        sentence_ranks[ids] = idxs.tolist()

    json.dump(sentence_ranks, open('sentence_prob_ranks.json', 'w'))




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_train_samples', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()
    main(args)