from transformers import GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from nltk import sent_tokenize
from tqdm import tqdm

GPT_MODEL = "gpt2"
EOS_TOKEN = "<|endoftext|>"
EOS_TOKEN_ID = 50256


class SentenceProbability(torch.nn.Module):
    def __init__(self):
        self.gpt2 = GPT2LMHeadModel.from_pretrained(GPT_MODEL)
    def forward(self,
                input_ids=None,
                attention_mask=None,
                sentence_mask=None):
        logits = self.gpt2(input_ids, attention_mask=attention_mask)
        batch_size, seq_len = logits.size(0), logits.size(1)
        log_probs = torch.log_softmax(logits, -1)
        flattened_probs = log_probs.view(-1, log_probs.size(-1))
        input_probs = flattened_probs[range(flattened_probs.size(0)), input_ids.view(-1)].view(batch_size, seq_len)
        log_prob = (sentence_mask * input_probs).sum(-1)
        return log_prob

def create_dataset(split="train"):
    tokenizer = AutoTokenizer.from_pretrained(GPT_MODEL)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    dataset = load_dataset('cnn_dailymail', "3.0.0")
    cur_data = dataset[split]
    articles = cur_data['article'][:10]

    all_sentences = [sent_tokenize(inp) for inp in articles]
    data = []

    for i, sents in enumerate(tqdm(all_sentences)):
        suffix = f" {EOS_TOKEN} {articles[i]}"
        tmp = [x + suffix for x in sents]
        model_input = tokenizer(tmp, padding=True, truncation=True)
        sentence_mask = []
        for input_id in model_input['input_ids']:
            s_mask = [1] * len(input_id)
            for idx, ii in enumerate(input_id):
                s_mask[idx] = 0
                if ii == EOS_TOKEN_ID: break

            sentence_mask.append(s_mask)
        model_input['sentence_mask'] = sentence_mask
        data.append(model_input)

    return data

def main():
    model = SentenceProbability()
    dataset = create_dataset()
    for x in dataset:
        inp = {key: torch.tensor(val) for key, val in x.items()}
    print(model(**inp))

    #print(dataset[0])
    # create sentence indicator and compute PMI for every sentence to the document
    # store this in a tensor or np array and load it in as a feature


if __name__ == "__main__":
    main()












