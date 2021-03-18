]1;95;0cimport nlpaug.augmenter.word as naw
from datasets import load_dataset
import nltk
import json
import pickle
from tqdm import tqdm
import multiprocessing as mp

def paraphrase(aug, data, ids):
    d =  {'a': 'test'}
    print('test')
    for _id, sentences in tqdm(zip(ids, data)):
        _id, sentences = tup
        augmented = aug.augment(sentences)
        d[_id] = ' '.join(augmented)
    return d
        
    
def main():
    aug = naw.SynonymAug(aug_p=0.4)
    dataset = load_dataset('cnn_dailymail', "3.0.0")

    train_data = dataset['train']
    inputs = train_data['article']
    targets = train_data['highlights']
    ids = train_data['id']
    print(len(train_data))
#    articles = [nltk.sent_tokenize(inp) for inp in tqdm(inputs)]
#    pickle.dump(articles, open('articles_sentences.json', 'wb'))
    articles = pickle.load(open('articles_sentences.json', 'rb'))[80000:]

    d = {}
    for i, art in enumerate(tqdm(articles)):
        augmented = aug.augment(art)
        d[ids[i]] = ' '.join(augmented)

#    num_splits = 10
#    split = len(articles)//num_splits
#    pool = mp.Pool(processes=num_splits)
#    results = []
#    for i in range(num_splits):
#        data = articles[i*split:(i+1)*split] if i < num_splits-1 else articles[i*split:]
#        ids = ids[i*split:(i+1)*split] if i < num_splits - 1 else ids[i*split:]
        
#        results.append(pool.apply_async(paraphrase, args=(aug, data, ids)))
#    results = [pool.apply_async(paraphrase, args=(aug, articles[i*split:(i+1)*split] if i < num_splits-1 else articles[i*split:], ids[i*split:(i+1)*split] if i < num_splits-1 else ids[i*split:])) for i in range(num_splits)]

 #   outputs = [p.get() for p in results]
 #   for x in outputs:
 #       d = {**d, **x}
    pickle.dump(d, open('paraphrase1.pkl', 'wb'))

def fix_data():
    dataset = load_dataset('cnn_dailymail', "3.0.0")
    train_data = dataset['train']
    data1 = pickle.load(open('paraphrase.pkl', 'rb'))
    data2 = pickle.load(open('paraphrase1.pkl', 'rb'))
    
    ids = train_data['id']
    d = {**data1}
    for i in range(80000, len(ids)):
        d[ids[i]] = data2[ids[i-80000]]
    print(len(d))
    pickle.dump(d, open('fixed_paraphrase.pkl', 'wb'))
    
if __name__ == "__main__":
#    main()
    fix_data()
