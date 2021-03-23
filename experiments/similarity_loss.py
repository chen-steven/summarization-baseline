from transformers import T5EncoderModel, AutoTokenizer
import torch
from datasets import load_dataset
from ..preprocess import _create_sentence_indicator
import nltk
import os
import pickle

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
        for i in range(len(ids)):
            d[ids[i]] = nltk.sent_tokenize(articles)
        pickle.dump(d, open(path, 'wb'))
        return d

def _create_sentence_embeddings(model, ids, model_input, sentence_indicators):
    d = {}
    for idx in range(len(ids)):
        inputs = {'input_ids': torch.tensor(model_input['input_ids'][idx]),
                  'attention_mask': torch.tensor(model_input['attention_mask'][idx])}
        sentence_indicator = torch.tensor(sentence_indicators[i])
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
        d[ids[idx]] = (sentences, sentence_len)
    return d


def similarity_oracle():
    model = T5EncoderModel.from_pretrained('t5-small')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    val_dataset = dataset['validation']
    ids = val_dataset['id']
    articles = val_dataset['article']
    sentences = _get_val_article_sentences()
    sep_token = "<sep>"
    sep_token_id = 1
    input = [f" {sep_token} ".join(s) for s in sentences]

    model_input = tokenizer(input, max_length=1024, padding="max_length", truncation=True)
    sentence_indicator = _create_sentence_indicator(model_input['input_ids'], tokenizer, sep_token_id)
    d = _create_sentence_embeddings(model, ids, model_input, sentence_indicator)
    torch.save(d, 'val_sentence_embeddings.pt')

if __name__ == "__main__":
    similarity_oracle()