import torch
import os
import torch.nn as nn
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from utils import save_state_dict
import logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

@dataclass_json
@dataclass
class TrainingParams:
    epochs: int = 10
    lr: int = 0.1
    step_size: float = 1.0
    gamma: float = 0.1
    batch_size: int = 64
    input_dim: int = 8
    embed_dim: int = 32
    num_classes: int = 17
    eval_every: int = 100
    model_path: str = "model.pt"


@dataclass
class Trainer:
    model: nn.Module
    train_loader: object
    valid_loader: object
    test_loader: object
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler
    writer: object
    criterion: nn.Module = nn.CrossEntropyLoss()
    epochs: int =  10
    batch_size: int = 64
    eval_every: int = 100
    device: torch.device = torch.device("cpu")
    global_step: int = 0
    logger: object = logging.getLogger(__name__)
    model_path: str = "model.pt"

    def train(self):
        accuracy = None
        for epoch in range(1, self.epochs + 1):
            self.logger.info("-"*21+f"EPOCH {epoch}"+"-"*21)
            for i,(sentence, language, offsets) in enumerate(self.train_loader):
                self.model.train()
                self.optimizer.zero_grad()
                predicted_label = self.model(sentence, offsets)
                loss = self.criterion(predicted_label, language)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
                self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]["lr"], self.global_step)
                self.writer.add_scalar('Training Loss', loss.item(), self.global_step)

                if i % self.eval_every == 0 and i > 0:
                    accu_val = self.evaluate()
                    self.logger.info(f"\tstep: {self.global_step}\t|\ttraining loss: {loss.item()}\t|\taccuracy: {accu_val}")
                    self.writer.add_scalar('Model Accuracy', accu_val, self.global_step)
                self.global_step+=1

            accu_val = self.evaluate()
            if accuracy is not None and accuracy > accu_val:
                self.scheduler.step()
            else:
                accuracy = accu_val
            self.logger.info(f"\tstep: {self.global_step}\t|\ttraining loss: {loss.item()}\t|\taccuracy: {accu_val}")
            self.writer.add_scalar('Model Accuracy', accu_val, self.global_step)


        save_state_dict("model.pt", self.model, self.optimizer, self.train_loader, self.valid_loader, self.test_loader)
        if self.model_path != "model.pt":
            save_state_dict(self.model_path, self.model, self.optimizer, self.train_loader, self.valid_loader, self.test_loader)
        self.logger.info(f"\tTraining Completed! Model accuracy: {accuracy}")
        return accuracy

    def evaluate(self):
        self.model.eval()
        total_acc, total_count = 0, 0
        with torch.no_grad():
            for sentence, language, offsets in self.valid_loader:
                predicted_label = self.model(sentence, offsets)
                val_loss = self.criterion(predicted_label, language)
                self.writer.add_scalar('Validation Loss', val_loss, self.global_step)
                total_acc += (predicted_label.argmax(1) == language).sum().item()
                total_count += language.size(0)
        return total_acc/total_count

