import torch
from tqdm import tqdm

from commons.arguments import get_arguments
import models.utils as model_utils
from commons.trainer import Trainer
from commons.tester import Tester
from metrics.iou import IoU
import torch.nn as nn
import torch.optim as optim

args = get_arguments()
device = torch.device(args.device)

best_result = {
    'iou': [],
    'miou': 0.0,
    'epoch': 0
}


def train(train_loader, val_loader, class_encoding):
    num_classes = len(class_encoding)
    model = model_utils.get_model(num_classes, pretrained=True)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    metric = IoU(num_classes=num_classes, ignore_index=None)
    trainer = Trainer(model=model, data_loader=train_loader, optimizer=optimizer, criterion=criterion, metric=metric,
                      device=device)
    val = Tester(model=model, data_loader=val_loader, criterion=criterion, metric=metric, device=device)
    for epoch in tqdm(range(0, args.epochs)):
        print("[Epoch: {0:d}] Training".format(epoch))
        epoch_loss, (iou, miou) = trainer.run_epoch()
        print("[Epoch: {0:d} mIoU: {1:.4f}".format(epoch, miou))
        if miou > best_result['miou']:
            best_result['epoch'] = epoch
            best_result['iou'] = iou
        print("[Epoch: {0:d} Avg loss: {1:.4f}".format(epoch, epoch_loss))
    return model


def test(model, test_loader, class_encoding):
    num_classes = len(class_encoding)
    criterion = nn.CrossEntropyLoss()
    tester = Tester(model=model, data_loader=test_loader, criterion=criterion, metric=None, device=device)
    loss = tester.run_epoch()
    print("[Test: loss: {0:.4f}".format(loss))




