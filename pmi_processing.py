from transformers import GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from nltk import sent_tokenize
from tqdm import tqdm
import pickle
import json
import os
from analysis import compute_extraction_performance

GPT_MODEL = "gpt2"
EOS_TOKEN = "<|endoftext|>"
EOS_TOKEN_ID = 50256


class SentenceProbability(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(GPT_MODEL)
    def forward(self,
                input_ids=None,
                attention_mask=None,
                sentence_mask=None):
        attention_mask = (input_ids != EOS_TOKEN_ID).long()
        logits = self.gpt2(input_ids, attention_mask=attention_mask)[0]
        batch_size, seq_len = logits.size(0), logits.size(1)
        log_probs = torch.log_softmax(logits, -1)
        flattened_probs = log_probs.view(-1, log_probs.size(-1))
        input_probs = flattened_probs[range(flattened_probs.size(0)), input_ids.view(-1)].view(batch_size, seq_len)
        log_prob = (sentence_mask * input_probs).sum(-1)
        doc_log_prob = ((1-sentence_mask)*(attention_mask*input_probs)).sum(-1)
        pmi = log_prob - doc_log_prob
        return pmi


def create_dataset(model, split="train"):
    path = f'data/pmi_{split}_features1.json'

    tokenizer = AutoTokenizer.from_pretrained(GPT_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset('cnn_dailymail', "3.0.0")
    cur_data = dataset[split][:100000]
    articles = cur_data['article']
    ids = cur_data['id']
#    if os.path.exists(path):
#        return json.load(open(path, 'r')), cur_data['id']

    all_sentences = [sent_tokenize(inp) for inp in articles]
    all_sentences = [x[:60] for x in all_sentences]
    
    tensor_map = {}
    print("Start")
    for i, sents in enumerate(tqdm(all_sentences)):
        suffix = f" {EOS_TOKEN} {articles[i]}"
        tmp = [x + suffix for x in sents]
        model_input = tokenizer(tmp, padding="longest", max_length=512, truncation=True)
        sentence_mask = []
        for input_id in model_input['input_ids']:
            s_mask = [1] * len(input_id)
            for idx, ii in enumerate(input_id):
                s_mask[idx] = 0
                if ii == EOS_TOKEN_ID:
                    del input_id[idx]
                    del s_mask[idx]
                    s_mask.append(1)
                    input_id.append(EOS_TOKEN_ID)
                    break

            sentence_mask.append(s_mask)
        model_input['sentence_mask'] = sentence_mask
        inp = {key: torch.tensor(val) for key, val in model_input.items()}
        with torch.no_grad():
            tensor_map[ids[i]] = model(**inp).tolist()
    torch.save(tensor_map, 'data/train_pmi.pt')
#    json.dump(data, open(path, 'w'))


def extraction_performance():
    preds = torch.load('data/val_pmi.pt')
    topk = {}
    for key in preds:
        num_select = min(3, len(preds[key]))
        topk[key] = torch.tensor(preds[key]).topk(k=num_select)[1].long().tolist()
        topk[key] = sorted(topk[key])

    res, ex_res = compute_extraction_performance(topk, split="validation")
    print(res)
    print(ex_res)
        
        
        

    
    

def main():
    model = SentenceProbability().cuda()
    model = torch.nn.DataParallel(model)
    model.eval()
    print('Training:',model.training)

    #tensor_map = {}
    create_dataset(model, 'train')
   # for i, x in enumerate(tqdm(dataset)):
   #     inp = {key: torch.tensor(val).cuda() for key, val in x.items()}
   #     with torch.no_grad():
  #          tensor_map[ids[i]] = model(**inp).tolist()

    #torch.save(tensor_map, 'data/train_pmi.pt')


    #print(dataset[0])
    # create sentence indicator and compute PMI for every sentence to the document
    # store this in a tensor or np array and load it in as a feature


if __name__ == "__main__":
    main()
#    extraction_performance()












