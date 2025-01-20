# it solves the problem for import modules
import sys
import pathlib


current_path = pathlib.Path(__file__).parent.resolve()
working_dir_path = pathlib.Path().resolve()

sys.path.append(str(working_dir_path))


import argparse
import logging
import os
from typing import Any, Tuple, List

import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.nn.modules.loss import _Loss
from torchvision import transforms
from PIL import Image
from pathlib import Path

from src.attention_model.averager import Averager
from src.attention_model.label_converting import Converter, AttnLabelConverter
from src.attention_model.model import Model
from src.attention_model.resize_normalization import AlignCollate
from src.dataset.attention_dataset import AttentionDataset
from src.dataset.utils import get_charset
from src.utils.logger import get_logger
from src.utils.metrics import string_accuracy, cer, wer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_image(image_path: str, opt: Any) -> torch.Tensor:
    """Обработка изображения для подачи в модель."""
    transform = transforms.Compose(
        [
            transforms.Resize((opt.img_h, opt.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    image = Image.open(image_path).convert("RGB" if opt.rgb else "L")
    return transform(image).unsqueeze(0).to(device)


def test(opt: Any, logger: logging.Logger) -> None:
    converter = AttnLabelConverter(opt.character)
    align_collate = AlignCollate(
        img_h=opt.img_h, img_w=opt.img_w, keep_ratio_with_pad=opt.pad
    )
    opt.num_class = len(converter.character)

    model = Model(opt, logger)
    model = torch.nn.DataParallel(model).to(device)
    logger.info(f"Loading pretrained model from {opt.saved_model}")
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    model.eval()

    P = Path(opt.images_path)
    for x in P.rglob("*"):
        if ".jpg" in str(x):
            image_tensors = process_image(x, opt)
        else:
            continue

        with torch.no_grad():
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(
                device
            )
            text_for_pred = (
                torch.LongTensor(batch_size, opt.batch_max_length + 1)
                .fill_(0)
                .to(device)
            )

            preds = model(image, text_for_pred, is_train=False)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

            pred = preds_str[0]
            pred_eos = pred.find("[s]")
            pred = pred[:pred_eos]
            logger.info(f"Распознанный текст: {pred}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir", type=str, help="Directory for saving log file", required=True
    )
    parser.add_argument(
        "--log_name", type=str, help="Name of the log file", required=True
    )
    # parser.add_argument('--data_dir', type=str, help='Path to dataset', required=True)
    # parser.add_argument('-n', '--label_files', nargs='+', required=True, help='Names of files with labels')
    parser.add_argument(
        "--saved_model",
        type=str,
        help="Path to attention_model to evaluate",
        required=True,
    )
    parser.add_argument(
        "--write_errors",
        action="store_true",
        help="Write attention_model's errors to the log file",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Input batch size")
    parser.add_argument(
        "--eval_stage", type=str, default="test", help="Name of test dataset stage"
    )
    parser.add_argument(
        "--images_path",
        type=str,
        help="Path to the image for recognition",
        required=True,
    )

    # Data processing
    parser.add_argument(
        "--batch_max_length", type=int, default=40, help="Maximum label length"
    )
    parser.add_argument(
        "--img_h", type=int, default=32, help="The height of the input image"
    )
    parser.add_argument(
        "--img_w", type=int, default=100, help="The width of the input image"
    )
    parser.add_argument("--rgb", action="store_true", help="Use rgb input")
    parser.add_argument(
        "--pad",
        action="store_true",
        help="Whether to keep ratio then pad for image resize",
    )

    opt = parser.parse_args()
    __charset = " !\"%'()+,-./0123456789:;=?R[]abcehinoprstuxy«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№"
    opt.character = __charset

    os.makedirs(opt.log_dir, exist_ok=True)
    logger = get_logger(out_file=os.path.join(opt.log_dir, opt.log_name))
    test(opt, logger)
