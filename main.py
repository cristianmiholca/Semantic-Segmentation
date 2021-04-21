import torch
import datasets.utils as dataset_utils
import commons.utils as commons_utils

from commons.arguments import get_arguments


args = get_arguments()


if __name__ == '__main__':
    print(torch.__version__)
    if args.dataset.lower() == 'camvid':
        from datasets.camvid import CamVid as dataset
    else:
        raise RuntimeError('\"{0}\" is not a supported dataset.'.format(args.dataset))
    data_loaders, class_encoding = dataset_utils.load_dataset(dataset)
    train_loader, val_loader, test_loader = data_loaders
    model = commons_utils.train(train_loader, val_loader, class_encoding)