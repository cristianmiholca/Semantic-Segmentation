from argparse import ArgumentParser


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        '--device',
        default='cuda',
        help='Device on which the network will be trained. Default: cuda')
    parser.add_argument(
        '--width',
        type=int,
        default=512,
        help='The width of the image. Default: 512'),
    parser.add_argument(
        '--height',
        type=int,
        default=512,
        help='The width of the image. Default: 512'),
    parser.add_argument(
        '--epochs',
        '-e',
        type=int,
        default=100,
        help='The number of epochs. Default: 100')
    parser.add_argument(
        '--batch-size',
        '-bs',
        type=int,
        default=8,
        help='The batch size. Default: 8')
    parser.add_argument(
        '--learning-rate',
        '-lr',
        type=float,
        default=5e-4,
        help='The learning rate. Default: 5e-4')
    parser.add_argument(
        '--workers',
        default=2,
        help='Number of workers. Default: 2')
    parser.add_argument(
        '--dataset',
        choices=['camvid'],
        default='camvid',
        help='Dataset to use. Default: camvid')
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='data/camvid',
        help='Path to the selected dataset. Default: data/camvid')
    parser.add_argument(
        '--model',
        choices=['pspnet', 'unet', 'segnet'],
        default='unet',
        help='The model to use. Default: unet')
    parser.add_argument(
        '--tqdm',
        choices=['True', 'False'],
        default='True',
        help='True if should use tqdm. Default: True')

    # Arguments for saving the model
    parser.add_argument(
        '--name',
        type=str,
        default='UNet',
        help="Name given to the model when saving. Default: UNet")
    parser.add_argument(
        "--save-dir",
        type=str,
        default='checkpoint',
        help="The directory where models are saved. Default: checkpoint")
    return parser.parse_args()
