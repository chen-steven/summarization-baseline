from transformers import MarianTokenizer, MarianMTModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

en_to_fr = 'Helsinki-NLP/opus-mt-en-fr'
fr_to_en = 'Helsinki-NLP/opus-mt-fr-en'


def main():
    dataset = load_dataset('cnn_dailymail', "3.0.0")

    train_data = dataset['train']
    inputs = train_data['article']
    targets = train_data['highlights']
    ids = train_data['id']

    tok_en_fr = MarianTokenizer.from_pretrained(en_to_fr)
    tok_fr_en = MarianTokenizer.from_pretrained(fr_to_en)
    model_en_fr = MarianMTModel.from_pretrained(en_to_fr)
    model_fr_en = MarianMTModel.from_pretrained(fr_to_en)

    d = {}

    dataset = [(ids[i], inputs[i]) for i in range(len(ids))]
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    for batch in tqdm(dataloader):
        _ids, en_text = batch
        en_input = tok_en_fr.prepare_seq2seq_batch(en_text, return_tensors="pt")
        fr_translation = tok_en_fr.batch_decode(model_en_fr.generate(**en_input), skip_special_tokens=True)
        fr_input = tok_fr_en.prepare_seq2seq_batch(fr_translation, return_tensors="pt")
        en_backtranslation = tok_fr_en.batch_decode(model_fr_en.generate(**fr_input), skip_special_tokens=True)

        for i in range(len(_ids)):
            d[_ids[i]] = en_backtranslation[i]
        break

    json.dump(d, open('train_backtranslation.json', 'w'))


if __name__ == "__main__":
    main()