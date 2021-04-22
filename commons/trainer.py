import torch
from tqdm import tqdm
from metrics import iou as IoU


class Trainer:
    def __init__(self, model, data_loader, optimizer, criterion, metric, device):
        super(Trainer, self).__init__()
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric
        self.device = device

    def run_epoch(self, use_tqdm=True):
        loop = tqdm(self.data_loader) if use_tqdm else self.data_loader
        if self.device.type == 'cuda':
            self.model.cuda()
        self.model.train()
        epoch_loss = 0.0
        for idx, batch in enumerate(loop):
            data = batch[0].to(self.device)
            target = batch[1].long().to(self.device)
            target = torch.argmax(target, 1)
            pred = self.model(data)
            loss = self.criterion(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            iou = IoU.iou(pred, target)
            print(iou)
        return epoch_loss / len(self.data_loader)
