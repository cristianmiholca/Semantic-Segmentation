import torch
from tqdm import tqdm

from commons.arguments import get_arguments
import models.utils as model_utils
from commons.trainer import Trainer
from commons.tester import Tester
import torch.nn as nn
import torch.optim as optim

args = get_arguments()
device = torch.device(args.device)


def train(train_loader, val_loader, class_encoding):
    num_classes = len(class_encoding)
    model = model_utils.get_model(num_classes, pretrained=True)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    trainer = Trainer(model=model, data_loader=train_loader, optimizer=optimizer, criterion=criterion, metric=None,
                      device=device)
    val = Tester(model=model, data_loader=val_loader, criterion=criterion, metric=None, device=device)
    for epoch in tqdm(range(0, args.epochs)):
        print("[Epoch: {0:d}] Training".format(epoch))
        epoch_loss = trainer.run_epoch()
        print("[Epoch: {0:d} Avg loss: {1:.4f}".format(epoch, epoch_loss))
    return model


