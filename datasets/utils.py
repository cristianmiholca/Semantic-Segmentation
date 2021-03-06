import os

import torch.utils.data
import torch.utils.data as data
import torchvision.transforms as TF

from commons.arguments import get_arguments

args = get_arguments()


def name_cond(name_filter: str):
    if name_filter is None:
        return lambda filename: True
    else:
        return lambda filename: name_filter in filename


def ext_cond(ext_filter: str):
    if ext_filter is None:
        return lambda filename: True
    else:
        return lambda filename: filename.endswith(ext_filter)


def get_files(dir_path, name_filter=None, extension_filter=None):
    if not os.path.isdir(dir_path):
        raise RuntimeError("\"{0}\" is not a directory.".format(dir_path))
    result = []
    for path, _, files in os.walk(dir_path):
        files.sort()
        for f in files:
            full_path = os.path.join(path, f)
            result.append(full_path)
    return result


def load_dataset(dataset):
    train_set = get_dataset(dataset, 'train')
    val_set = get_dataset(dataset, 'val')
    test_set = get_dataset(dataset, 'test')
    train_loader = get_dataloader(train_set)
    val_loader = get_dataloader(val_set)
    test_loader = get_dataloader(test_set)
    class_encoding = train_set.class_encoding
    return (train_loader, val_loader, test_loader), class_encoding


def get_dataset(dataset, mode):
    image_transform = TF.Compose([
        TF.Resize((args.width, args.height)),
        TF.ToTensor()
    ])
    # TODO POSSIBLE PROBLEM: PILToTensor transform
    target_transform = TF.Compose([
        TF.Resize((args.width, args.height), TF.InterpolationMode.NEAREST),
        TF.PILToTensor()
    ])
    return dataset(
        root_dir=args.dataset_dir,
        mode=mode,
        data_transform=image_transform,
        label_transform=target_transform
    )


def get_dataloader(dataset, shuffle=True):
    return data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        drop_last=True
    )


def get_target_mask(target, class_encoding):
    colors = class_encoding.values()
    mapping = {tuple(c): t for c, t in zip(colors, range(len(colors)))}
    target_size = list(target.size())
    mask = torch.zeros(target_size[1], target_size[2], dtype=torch.long)
    for k in mapping:
        idx = (target == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
        validx = (idx.sum(0) == 3)
        mask[validx] = torch.tensor(mapping[k], dtype=torch.long)
    return mask
