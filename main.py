import pandas as pd
import torch
import os
from models import LanguageClassifierModel
from data import tokenize, preprocessing, get_dataloader
from trainer import Trainer
from utils import load_state_dict

# Hyperparameters
EPOCHS = 10
LR = 5
STEP_SIZE = 1.0
GAMMA = 0.1
BATCH_SIZE = 64
INPUT_DIM = 8
EMBED_DIM = 32
NUM_CLASSES = 17

def train():
    df = pd.read_csv("Language_Detection.csv")
    df, vocab = preprocessing(df, indput_dim=INPUT_DIM)
    model = LanguageClassifierModel(vocab_size=len(vocab.keys()), embed_dim=EMBED_DIM, num_class=NUM_CLASSES) 
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    if not os.path.exists("model.pt"):
        train, valid, test = get_dataloader(df, batch_size=BATCH_SIZE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA) 
        trainer = Trainer(model, train, valid, test, optimizer, scheduler)
        trainer.train()
    model, optimizer, data = load_state_dict(model, optimizer, load_path="model.pt")
    return model, optimizer, vocab, data


def test(model, test_loader):
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for sentence, language in test_loader:
            predicted_label = model(sentence)
            total_acc += (predicted_label.argmax(1) == language).sum().item()
            total_count += language.size(0)
    return total_acc/total_count


def predict(model, vocab, sentence):
    model.eval()
    input = tokenize([sentence], vocab, INPUT_DIM)
    predicted_labels = model(input)
    label = torch.argmax(predicted_labels, dim=1)
    return label.item()


model, optimizer, vocab, data = train()
test_accuracy = test(model, data["test"])
label = predict(model, vocab, "questa è una frase in italiano!")

print(test_accuracy)
print(label)