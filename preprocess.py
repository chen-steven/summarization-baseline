from datasets import load_dataset

def transform_dataset(examples, tokenizer):
    inputs = examples['article']
    targets = examples['highlights']



def preprocess_cnn():
    dataset = load_dataset('cnn_dailymail ', "3.0.0")
    train_data = dataset['train']
    val_data = dataset['val']
    test_data = dataset['test']


