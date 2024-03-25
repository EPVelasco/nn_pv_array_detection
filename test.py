import torch
import torch.nn as nn
from models import UnetPlusPlus, PSPNet, DeepLabV3Plus
from dataset import SegmentationDataset
from utils import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import time
import sys

#TODO se debe modificar para poder detectar de  un solo shot
# El sigueinte codigo no funciona, es un modelo base para modificar a las necesidades actuales

PATH = "/epvelasco/estancia/red_segmentacion/experiments/"
TEST_IMG_DIR = PATH
TEST_MASK_DIR = PATH
TEST_PREDS_DIR = PATH
IMAGE_HEIGHT = 2048
IMAGE_WIDTH = 128  # 240 for unet, 256 for unet smp with resnet18
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_WORKERS = os.cpu_count()
PIN_MEMORY = True


def test_single_shot():

    n_epoch = sys.argv[1]

    model = UnetPlusPlus("resnet18", "imagenet", in_channels=3, out_channels=1).to(DEVICE)

    load_checkpoint(torch.load(PATH + "epochs/checkpoint_epoch_" + str(n_epoch) + ".pth.tar"), model)

    test_transform = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0  # value you want to divide by the pixels
            ),
            ToTensorV2(),
        ]
    )

    id_val = sys.argv[2]

    image = np.array(Image.open(PATH + "train/merged_" + str(id_val) + ".png").convert("RGB"))

    #with torch.inference_mode():
    image = test_transform(image=image)
    image = image["image"]

    image = image.to(DEVICE).unsqueeze(0)
    pred = torch.sigmoid(model(image))
    pred = (pred > 0.5).float()

    torchvision.utils.save_image(
        pred, PATH + "out/merged_" + str(id_val) + "_pred.png"
      )
    torchvision.utils.save_image(
        image, PATH + "out/merged_" + str(id_val) + "_imag.png"
      )

if __name__ == '__main__':
    test_single_shot()