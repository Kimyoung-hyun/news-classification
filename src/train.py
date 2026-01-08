import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import random

from .data.data_processing import data_processing
from .data.vocab import build_vocab
from .data.custom_dataset import CustomDataset, collate_batch
from .LSTM_classifier import LSTMClassifier

def main():
    MIN_FREQ = 2
    BATCH_SIZE = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # random seed
    SEED = 42
    set_seed(SEED)
    
    train_df = pd.read_csv("/data/ephemeral/test/news-classification/data/test.csv", header=0, names=['Class Index', 'Title', 'Description'])
    train_df = data_processing(train_df)

    test_df = pd.read_csv("/data/ephemeral/test/news-classification/data/test.csv", header=0, names=['Class Index', 'Title', 'Description'])

    vocab = build_vocab(train_df, min_freq=MIN_FREQ)

    train_dataset = CustomDataset(train_df, vocab, is_train=True)
    test_dataset = CustomDataset(test_df, vocab, is_train=True) # test 도 라벨있는디..

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    
    INPUT_DIM = len(vocab)
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    OUTPUT_DIM = 4
    DROPOUT = 0.5
    
    model = LSTMClassifier(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, n_layers=2, dropout=DROPOUT)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 6. 실제 학습 실행
    EPOCHS = 10
    print(f"학습 시작 device: {device}")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, test_loader, criterion, device)
        
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {valid_loss:.3f} | Val Acc: {valid_acc*100:.2f}%')

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)
        
        # 이전 배치 초기화
        optimizer.zero_grad()
        # 예...측?
        output = model(data)
        # loss function
        loss = criterion(output, label)
        acc = (output.argmax(1) == label).float().mean()
        # 역전파 - 기울기 계산
        loss.backward()
        # 가중치 업데이트 
        optimizer.step()
        # 손실값 누적?
        epoch_loss += loss.item()
        epoch_acc += acc.item()   
        
    return epoch_loss / len(train_loader), epoch_acc / len(train_loader)    

# 검증? inference?
def evaluate(model, test_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            
            output = model(data)
            loss = criterion(output, label)
            acc = (output.argmax(1) == label).float().mean()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    return epoch_loss / len(test_loader), epoch_acc / len(test_loader)
    
def set_seed(seed_value):
    random.seed(seed_value)  # 파이썬 난수 생성기
    np.random.seed(seed_value)  # Numpy 난수 생성기
    torch.manual_seed(seed_value)  # PyTorch 난수 생성기

    # CUDA 환경에 대한 시드 설정 (GPU 사용 시)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    main()