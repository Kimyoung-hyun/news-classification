from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import re
from .data_processing import basic_english_normalize

class CustomDataset(Dataset):
    def __init__(self, df, vocab, is_train=True):
        self.vocab = vocab
        self.is_train = is_train
        
        self.data = []
        self.labels = []
        
        for _, row in df.iterrows():
            # 전처리?
            text = str(row['Title']) + " " + str(row['Description'])
            text = basic_english_normalize(text)
            token_ids = []
            for word in text.split():
                token_ids.append(self.vocab.get(word, 1))
            
            self.data.append(torch.tensor(token_ids, dtype=torch.long))
            # 걍 다들 하길래 함... ㅎㅎ
            self.labels.append(int(row['Class Index']) - 1)
        
    #  데이터 전체 길이  
    def __len__(self):
        return len(self.data)
    
    # 어떻게 반환?
    def __getitem__(self, idx):
        if self.is_train:
            return self.data[idx], self.labels[idx] # train 만 class index 값 넘기기
        else:
            return self.data[idx]
        
def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        text_list.append(_text)
        label_list.append(_label)
        
    inputs_padded = pad_sequence(text_list, batch_first=True, padding_value=0)
    labels_stacked = torch.tensor(label_list, dtype=torch.long)
    
    return inputs_padded, labels_stacked