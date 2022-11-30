import pandas as pd
import torch
import os
from tqdm import tqdm
from models import LanguageClassifierModel
from data import tokenize, preprocessing, get_dataloader
from utils import rollback_training
from trainer import Trainer, TrainingParams
from flask import Flask, request
from torch.utils.tensorboard import SummaryWriter
from data import LANG_LOOKUP
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route('/train', methods=['POST'])
def train():
    try:
        print(request.get_json())
        params = TrainingParams.from_dict(request.get_json())
        df = pd.read_csv("./data/Language_Detection.csv")
        df, vocab = preprocessing(df, input_dim=params.input_dim)
        model = LanguageClassifierModel(vocab_size=len(vocab.keys()), embed_dim=params.embed_dim, num_class=params.num_classes) 
        optimizer = optimizer = torch.optim.SGD(model.parameters(), lr=params.lr)
        train, valid, test = get_dataloader(df, batch_size=params.batch_size)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=params.gamma)
        writer = SummaryWriter(comment=f"B_{params.batch_size}_ED_{params.embed_dim}_IN_{params.input_dim}_LR_{params.lr}")
        trainer = Trainer (
            model=model, 
            train_loader=train, 
            valid_loader=valid, 
            test_loader=test, 
            optimizer=optimizer, 
            scheduler=scheduler,
            epochs=params.epochs,
            batch_size=params.batch_size,
            eval_every=params.eval_every,
            writer=writer,
            model_path=params.model_path
        )
        accuracy = trainer.train()
        return {"accuracy": accuracy}
    except Exception as ex:
        rollback_training()
        logger.exception(ex)
        return {"error": str(ex)}


@app.route('/test', methods=['GET'])
def test():
    try:
        state_dict = torch.load("model.pt")
        kwargs = state_dict["kwargs"]
        model = LanguageClassifierModel(**kwargs) 
        model.load_state_dict(state_dict["model"])
        data = state_dict["data"]
        model.eval()
        total_acc, total_count = 0, 0
        with torch.no_grad():
            for sentence, language, offsets in tqdm(data["test"], desc="Testing..."):
                predicted_label = model(sentence, offsets)
                total_acc += (predicted_label.argmax(1) == language).sum().item()
                total_count += language.size(0)
        return {"test_accuracy":total_acc/total_count}
    except Exception as ex:
        logger.exception(ex)
        return {"error": str(ex)}


@app.route('/predict', methods=['POST'])
def predict():
    try:
        sentence = request.get_json()["text"]
        vocab = torch.load("vocab.pt")
        state_dict = torch.load("model.pt")
        kwargs = state_dict["kwargs"]
        model = LanguageClassifierModel(**kwargs) 
        model.load_state_dict(state_dict["model"])
        model.eval()
        input = tokenize([sentence], vocab)
        predicted_labels = model(input, None)
        label = torch.argmax(predicted_labels, dim=1)
        response = 1 if LANG_LOOKUP[label.item()] == "Italian" else 0
        return {"class": response}
    except Exception as ex:
        logger.exception(ex)
        return {"error": str(ex)}

if __name__ == '__main__':
    app.run(debug=True)