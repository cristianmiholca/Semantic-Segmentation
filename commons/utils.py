import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm

import models.utils as model_utils
import transforms as ext_transforms
from commons.arguments import get_arguments
from commons.tester import Tester
from commons.trainer import Trainer
from metrics.iou import IoU

args = get_arguments()
device = torch.device(args.device)

best_result = {
    'iou': [],
    'miou': 0.0,
    'epoch': 0
}


def train(train_loader, val_loader, class_encoding):
    print("\nTraining\n")
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
        loss, (iou, miou) = trainer.run_epoch()
        print("[Epoch: {0:d}] Avg Loss:{1:.4f} MIoU: {2:.4f}".format(epoch, loss, miou))
        print(iou)
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print("[Epoch: {0:d}] Validation".format(epoch))
            loss, (iou, miou) = val.run_epoch()
            print("[Epoch: {0:d}] Avg loss: {1:.4f} MIoU: {2:.4f}".format(epoch, loss, miou))
            if miou > best_result['miou']:
                best_result['miou'] = miou
                best_result['epoch'] = epoch
                best_result['iou'] = iou
                print(best_result)
    return model


# TODO complete test method
def test(model, test_loader, class_encoding):
    print("\nTesting\n")
    num_classes = len(class_encoding)
    criterion = nn.CrossEntropyLoss()
    metric = IoU(num_classes=num_classes, ignore_index=None)
    tester = Tester(model=model, data_loader=test_loader, criterion=criterion, metric=metric, device=device)
    loss, (iou, miou) = tester.run_epoch()
    print("[Test] loss: {0:.4f}".format(loss))
    # TODO show results and add plt.show() in main (if not works by default)
    data, targets = iter(test_loader).__next__()
    # imshow_batch(data[0], targets[0])
    predict(model, data, targets, class_encoding)


def predict(model, images, targets, class_encoding):
    images = images.to(device)

    # Make predictions!
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    _, predictions = torch.max(predictions.data, 1)
    label_to_rgb = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    color_predictions = batch_transform(predictions.cpu(), label_to_rgb)
    imshow_batch(images.detach().cpu(), color_predictions)


def batch_transform(batch, transform):
    transf_slices = [transform(tensor) for tensor in torch.unbind(batch)]
    return torch.stack(transf_slices)


def imshow_batch(images, labels):
    images = torchvision.utils.make_grid(images).numpy()
    labels = torchvision.utils.make_grid(labels).numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
    ax1.imshow(np.transpose(images, (1, 2, 0)))
    ax2.imshow(np.transpose(labels, (1, 2, 0)))

    plt.show()

# def show_results(model: nn.Module, data, targets):
#     data = data.to(device)
#     model.eval()
#     with torch.no_grad():
#         pred = model(data)
#     print(pred.shape)
#     print(pred)
#
#
#
#
# def imshow_batch(images, targets):
#     images = torchvision.utils.make_grid(images).numpy()
#     targets = torchvision.utils.make_grid(targets).numpy()
#
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
#     ax1.imshow(np.transpose(images, (1, 2, 0)))
#     ax2.imshow(np.transpose(targets, (1, 2, 0)))
#     plt.show()
