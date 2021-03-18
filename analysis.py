from transformers import AutoTokenizer, T5ForConditionalGeneration
from datasets import load_dataset, load_metric
import torch
import nltk
import json
from tqdm import tqdm
from models.metrics import ExtractionScorer


def preprocess(examples, tokenizer):
    inputs = examples['article']
    ids = examples['id']

    examples = []
    example_sentences = [nltk.sent_tokenize(inp) for inp in inputs]
    for idx, sentences in tqdm(enumerate(example_sentences)):
        ex = []
        lab = []

        for i in range(len(sentences)):
            inp = " ".join(sentences[:i]) + " <extra_id_0> " + " ".join(sentences[i+1:])
            label = "<extra_id_0> " + sentences[i] + " <extra_id_1>"
            ex.append(inp)
            lab.append(label)

        inp_dict = tokenizer(ex, padding=True, truncation=True, return_tensors="pt")
        labels = tokenizer(lab, padding=True, truncation=True, return_tensors="pt")
        inp_dict['labels'] = labels.input_ids
        examples.append((inp_dict, labels.attention_mask, ids[idx]))

    return examples

def get_probs(model, inputs, label_att_mask):
    logits = model(**inputs).logits
    log_probs = torch.log_softmax(logits, -1)
    labels = inputs['labels']

    selected_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
    sentence_prob = torch.sum(label_att_mask*selected_probs, -1)
    
    return sentence_prob

class Model(torch.nn.Module):
    def __init__(self, t5):
        super().__init__()
        self.t5 = t5
    def forward(self, label_att_mask, input_ids=None, attention_mask=None, labels=None):
        logits = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
        log_probs = torch.log_softmax(logits, -1)
        selected_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        sentence_probs = torch.sum(label_att_mask*selected_probs, -1)
        
        return sentence_probs
    
def main(args):
    dataset = load_dataset('cnn_dailymail', "3.0.0")
    train_dataset = dataset['train'].select(range(args.max_train_samples))

    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    examples = preprocess(train_dataset, tokenizer)

    t5 = T5ForConditionalGeneration.from_pretrained('t5-small')
    model = Model(t5)
    model = model.cuda()
    model = torch.nn.DataParallel(model)

#    model.to(args.device)
    model.eval()

    sentence_ranks = {}

    for inputs, label_att_mask, ids in tqdm(examples):
        for key in inputs:
             inputs[key] = inputs[key].to(args.device)
#        sentence_probs = get_probs(model, inputs, label_att_mask)
        sentence_probs = model(label_att_mask, **inputs)
        _, idxs = sentence_probs.sort()
        sentence_ranks[ids] = idxs.tolist()

    json.dump(sentence_ranks, open('sentence_prob_ranks2.json', 'w'))

def compute_extraction_metrics():
    scorer = ExtractionScorer()
    metric = load_metric('rouge')
    labels = json.load(open('train_sentence_labels.json', 'r'))
    preds = json.load(open('sentence_prob_ranks.json', 'r'))
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    train_dataset = dataset['train']
    inputs = train_dataset['article']
    targets = train_dataset['highlights']
    

    ids = train_dataset['id']
    data_dict = {uid: (inputs[i], targets[i]) for i, uid in enumerate(ids)}

    pred_list = []
    label_list = []
    pred_sum = []
    target_sum = []
    for key in preds:
        lab = labels[key]
        pred = preds[key][:len(lab)]
        pred_list.append(pred)
        label_list.append(labels[key])
        inp, tar = data_dict[key]
        sentences = nltk.sent_tokenize(inp)
        pred_sum.append("\n".join([sentences[i] for i in pred]))
        target_sum.append("\n".join(sent for sent in nltk.sent_tokenize(tar)))

    result = metric.compute(predictions=pred_sum, references=target_sum, use_stemmer=True)
    
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    print(result)
    res = scorer.compute_metric(pred_list, label_list, postprocess=False)
    print(res)

def lead_baseline(num_lead=3):
    scorer = ExtractionScorer()
    metric = load_metric('rouge')
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    train_dataset = dataset['validation']
    inputs = train_dataset['article']
    targets = train_dataset['highlights']
    labels = json.load(open('data/val_sentence_labels.json', 'r'))

    print("Tokenizing sentences...")
    sentences = [nltk.sent_tokenize(inp) for inp in tqdm(inputs)]

    print("Creating predictions...")
    preds = ['\n'.join(s[i] for i in range(min(num_lead, len(s)))) for s in sentences]
    print("Creating targets...")
    tar = ['\n'.join(nltk.sent_tokenize(s)) for s in tqdm(targets)]

    pred_evidence = [[i for i in range(num_lead)] for _ in range(len(labels))]
    gt_evidence = [labels[key] for key in labels]

    result = metric.compute(predictions=preds, references=tar, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    print(result)
    res = scorer.compute_metric(pred_evidence, gt_evidence, postprocess=False)
    print(res)

def compute_rouge_score(pred, targets):
    metric = load_metric('rouge')

    preds = ['\n'.join(nltk.sent_tokenize(s)) for s in tqdm(pred)]
    targets = ['\n'.join(nltk.sent_tokenize(s)) for s in tqdm(targets)]
    result = metric.compute(predictions=preds, references=targets, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return result

def t5_zeroshot():
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    val_dataset = dataset['validation']
    inputs = val_dataset['article']
    targets = val_dataset['highlights']
    device = torch.device('cuda:1')
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained('t5-small')

    predictions = []
    for article in tqdm(inputs):
        inp = tokenizer("summarize: "+article, return_tensors="pt", padding="max_length", max_length=512).to(device)
        summary_ids = model.generate(inp.input_ids, num_beams=1, no_repeat_ngram_size=3, min_length=10, max_length=128, length_penalty=2.0)
        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        predictions.append(output)

    print(compute_rouge_score(predictions, targets))
    
        
    

    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_train_samples', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()
#    main(args)
#    compute_extraction_metrics()
#    lead_baseline(5)
    t5_zeroshot()
