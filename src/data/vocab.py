from collections import Counter

def build_vocab(df, min_freq=1):
    counter = Counter()
    for idx, row in df.iterrows():
        text = str(row['Title']) + " " + str(row['Description'])
        counter.update(text.split())
    
    # <pad>: 0, <unk>: 1
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
            
    print(f"Vocab 크기: {len(vocab)}")
    return vocab