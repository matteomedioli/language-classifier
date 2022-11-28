import torch
import torch.nn as nn
from math import inf
from dataclasses import dataclass
from utils import save_state_dict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

@dataclass
class Trainer:
    model: nn.Module
    train_loader: object
    valid_loader: object
    test_loader: object
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler
    criterion: nn.Module = nn.CrossEntropyLoss()
    num_epochs: int =  10
    device: torch.device = torch.device("cpu")
    global_step: int = 0
    writer: object = SummaryWriter()  

    def train(self):
        total_accu = None
        for _ in range(1, self.num_epochs + 1):
            
                total_acc, total_count = 0, 0
                eval_everty = 250
                for i,(sentence, language) in tqdm(enumerate(self.train_loader), desc="Training"):
                    self.model.train()
                    self.optimizer.zero_grad()
                    predicted_label = self.model(sentence)
                    loss = self.criterion(predicted_label, language)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                    self.optimizer.step()
                    total_acc += (predicted_label.argmax(1) == language).sum().item()
                    total_count += language.size(0)
                    self.writer.add_scalar('Training Loss', loss.item(), self.global_step)

                    if i % eval_everty == 0 and i > 0:
                        accu_val = self.evaluate()
                        self.writer.add_scalar('Validation Accuracy', accu_val, self.global_step)
                        total_acc, total_count = 0, 0
                    self.global_step+=1

                accu_val = self.evaluate()
                if total_accu is not None and total_accu > accu_val:
                    self.scheduler.step()
                else:
                    total_accu = accu_val
                
                self.writer.add_scalar('Validation Accuracy', accu_val, self.global_step)

        save_state_dict("model.pt", self.model, self.optimizer, self.train_loader, self.valid_loader, self.test_loader)


    def evaluate(self):
        self.model.eval()
        total_acc, total_count = 0, 0
        with torch.no_grad():
            for sentence, language in self.train_loader:
                predicted_label = self.model(sentence)
                val_loss = self.criterion(predicted_label, language)
                self.writer.add_scalar('Validation Loss', val_loss, self.global_step)
                total_acc += (predicted_label.argmax(1) == language).sum().item()
                total_count += language.size(0)
        return total_acc/total_count

