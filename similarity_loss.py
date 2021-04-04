from transformers import T5EncoderModel, AutoTokenizer
from models import UnsupervisedDenoiseT5
import torch
from datasets import load_dataset, load_metric
from preprocess import _create_sentence_indicator
from tqdm import tqdm
import nltk
import os
import pickle
import json
import utils
import numpy as np
from models.metrics import ExtractionScorer

def _get_val_article_sentences():
    path = 'data/cnn_val_sentences.p'
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    else:
        dataset = load_dataset('cnn_dailymail', '3.0.0')
        val_dataset = dataset['validation']
        ids = val_dataset['id']
        articles = val_dataset['article']
        d = {}
        for i in tqdm(range(len(ids))):
            d[ids[i]] = nltk.sent_tokenize(articles[i])
        pickle.dump(d, open(path, 'wb'))
        return d

def _create_sentence_embeddings(model, ids, model_input, sentence_indicators):
    d = {}
    sim = torch.nn.CosineSimilarity(-1)
    for idx in tqdm(range(len(ids))):
        inputs = {'input_ids': torch.tensor([model_input['input_ids'][idx]]).cuda(),
                  'attention_mask': torch.tensor([model_input['attention_mask'][idx]]).cuda()}
        sentence_indicator = torch.tensor([sentence_indicators[idx]]).cuda()
        output = model(**inputs)
        hidden_states = output[0]
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
        pooled_embedding = (hidden_states*inputs['attention_mask'].unsqueeze(-1)).sum(1).unsqueeze(1)

        sentence_mask = utils.get_sentence_mask(sentence_indicator, sentences.size(1)).float()

        cur = torch.zeros(sentences.size(0), sentences.size(-1)).cuda()
        cur_len = torch.zeros(sentence_lens.size(0), sentence_lens.size(-1)).cuda()
        l = []
        for i in range(3):
            candidates = cur.unsqueeze(1) + sentences
            candidate_lens = cur_len.unsqueeze(1) + sentence_lens
            cur_embedding = candidates / candidate_lens
            scores = sim(cur_embedding, pooled_embedding)
            
            scores = utils.mask_tensor(scores, sentence_mask)
            index = torch.argmax(scores)
            cur = candidates[range(1), index]
            cur_len = candidates[range(1), index]
#            pooled_embedding -= sentences[range(1),index]
            sentence_mask[range(1), index] = 0
            l.append(index.item())

        d[ids[idx]] = l

    pickle.dump(d, open('sim_oracle5.p', 'wb'))
    return d


def similarity_oracle():
    torch.cuda.set_device(1)
    model = T5EncoderModel.from_pretrained('t5-small').cuda()
#    model = UnsupervisedDenoiseT5.from_pretrained('output_dirs/uns_denoise_debug6').encoder.cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    val_dataset = dataset['validation']
    ids = val_dataset['id']
    articles = val_dataset['article']
    sentences = _get_val_article_sentences()
    for i in ids:
        np.random.shuffle(sentences[i])
    sep_token = "</s>"
    sep_token_id = 1
    inputs = [f" {sep_token} ".join(sentences[i]) for i in ids]

    model_input = tokenizer(inputs, max_length=1024, padding="max_length", truncation=True)
    sentence_indicator = _create_sentence_indicator(model_input['input_ids'], tokenizer, sep_token_id)

    d = _create_sentence_embeddings(model, ids, model_input, sentence_indicator)
    torch.save(d, 'val_sentence_embeddings.pt')

def compute_metrics():
    scorer = ExtractionScorer()
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    val_dataset = dataset['validation']
    inputs = val_dataset['article']
    targets = val_dataset['highlights']
    ids = val_dataset['id']
    metric = load_metric('rouge')
    labels = json.load(open('data/val_sentence_labels.json', 'r'))

    pred_indices = pickle.load(open('sim_oracle5.p', 'rb'))
    sentences = [nltk.sent_tokenize(inp) for inp in tqdm(inputs)]

    preds = []
    for i in range(len(sentences)):
        p = pred_indices[ids[i]]
        preds.append('\n'.join(sentences[i][j] for j in p))

    tar = ['\n'.join(nltk.sent_tokenize(s)) for s in tqdm(targets)]

    pred_evidence = [sorted(pred_indices[i]) for i in ids]
    gt_evidence = [labels[i] for i in ids]

    result = metric.compute(predictions=preds, references=tar, use_stemmer=True)
    result = {key : value.mid.fmeasure*100 for key, value in result.items()}
    print(result)
    res = scorer.compute_metric(pred_evidence, gt_evidence, postprocess=False)
    print(res)

if __name__ == "__main__":
    #similarity_oracle()
    compute_metrics()
