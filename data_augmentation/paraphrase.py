import nlpaug.augmenter.word as naw
from datasets import load_dataset
import nltk
import json
from tqdm import tqdm

def main():
    aug = naw.SynonymAug(aug_p=0.4)
    dataset = load_dataset('cnn_dailymail', "3.0.0")

    train_data = dataset['train']
    inputs = train_data['article']
    targets = train_data['highlights']
    ids = train_data['id']

    articles = json.load(open('articles_sentences.json', 'r'))#[nltk.sent_tokenize(inp) for inp in tqdm(inputs)]
    #json.dump(articles, open('articles_sentences.json', 'w'))
    d = {}
    for i, art in enumerate(tqdm(articles)):
        augmented = aug.augment(art)
        d[ids[i]] = ' '.join(augmented)


    json.dump(d, open('paraphrase.json', 'w'))


if __name__ == "__main__":
    main()
