import torch
import torch.nn as nn
import torchvision


def get_encoder(name: str, pretrained=True):
    encoder, feature_names, backbone_output = None, None, None
    if name == 'resnet18':
        encoder = torchvision.models.resnet18(pretrained)
    elif name == 'resnet34':
        encoder = torchvision.models.resnet34(pretrained)
    elif name == 'vgg16':
        encoder = torchvision.models.vgg16_bn(pretrained)
    elif name == 'vgg19':
        encoder = torchvision.models.vgg19_bn(pretrained)
    # if name.startswith('resnet'):
    #     feature_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
    #     backbone_output = 'layer_4'

    return encoder


class UnetWithEncoder(nn.Module):
    def __init__(self):
        super().__init__()


def test():
    encoder = get_encoder('vgg19')
    features = encoder.features.children()
    print(features)
    # for i in range(0, len(features)):
    #     print("(" + str(i + 1) + ")" + str(features[i]))


if __name__ == "__main__":
    test()
