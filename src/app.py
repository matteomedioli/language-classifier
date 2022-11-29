import pandas as pd
import torch
import os
from models import LanguageClassifierModel
from data import tokenize, preprocessing, get_dataloader
from trainer import Trainer
from utils import load_model, load_state_dict
from flask import Flask, request
from data import LANG_DICT

app = Flask(__name__)

EPOCHS = 10
LR = 5
STEP_SIZE = 1.0
GAMMA = 0.1
BATCH_SIZE = 64
INPUT_DIM = 4
EMBED_DIM = 32
NUM_CLASSES = 17
LANG_LOOKUP = {v:k for k,v in LANG_DICT.items()}


@app.route('/train', methods=['GET', 'POST'])
def train():
    os.system("del vocab.pt")
    os.system("del model.pt")
    df = pd.read_csv("Language_Detection.csv")
    df, vocab = preprocessing(df, input_dim=INPUT_DIM)
    model = LanguageClassifierModel(vocab_size=len(vocab.keys()), embed_dim=EMBED_DIM, num_class=NUM_CLASSES) 
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    train, valid, test = get_dataloader(df, batch_size=BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA) 
    trainer = Trainer(model=model, 
                train_loader=train, 
                valid_loader=valid, 
                test_loader=test, 
                optimizer=optimizer, 
                scheduler=scheduler,
                num_epochs=EPOCHS)
    accuracy = trainer.train()
    return {"accuracy": accuracy}


@app.route('/test')
def test():
    vocab = torch.load("vocab.pt")
    model = LanguageClassifierModel(vocab_size=len(vocab.keys()), embed_dim=EMBED_DIM, num_class=NUM_CLASSES) 
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    model, optimizer, data = load_state_dict(model, optimizer, load_path="model.pt")
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for sentence, language in data["test"]:
            predicted_label = model(sentence)
            total_acc += (predicted_label.argmax(1) == language).sum().item()
            total_count += language.size(0)
    return {"test_accuracy":total_acc/total_count}


@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.get_json()["text"]
    vocab = torch.load("vocab.pt")
    model = LanguageClassifierModel(vocab_size=len(vocab.keys()), embed_dim=EMBED_DIM, num_class=NUM_CLASSES) 
    model = load_model(model, load_path="model.pt")
    model.eval()
    input = tokenize([sentence], vocab, INPUT_DIM)
    predicted_labels = model(input)
    label = torch.argmax(predicted_labels, dim=1)
    response = 1 if LANG_LOOKUP[label.item()] == "Italian" else 0
    return {"class": response}


@app.route('/tensorboard', methods=['GET', 'POST'])
def tensorboard():
    return {"response": os.popen('tensorboard --logdir=runs &').read()}


@app.route('/run_tests', methods=['GET', 'POST'])
def run_tests():
    return {"response": os.popen('pytest tests.py').read()}


if __name__ == '__main__':
    app.run(debug=True)