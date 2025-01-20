# it solves the problem for import modules
import sys
import pathlib


current_path = pathlib.Path(__file__).parent.resolve()
working_dir_path = pathlib.Path().resolve()

sys.path.append(str(working_dir_path))


import argparse
import logging
import os
import random
import time
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from torch.optim import Optimizer, AdamW
from tqdm.notebook import tqdm
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel

from src.attention_model.averager import Averager
from src.dataset.trocr_dataset import TrocrDataset
from src.dataset.transforms import transforms
from src.utils.logger import get_logger
from src.utils.metrics import compute_metrics, string_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")


def prepare_data(
    opt: Any,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_df_list, val_df_list = [], []

    for label_file in opt.label_files:
        data_df = pd.read_csv(
            os.path.join(opt.data_dir, label_file), sep=",", dtype={"text": str}
        )
        train_df_list.append(data_df[data_df.stage == "train"])
        val_df_list.append(data_df[data_df.stage == "val"])

    train_dataset = TrocrDataset(
        train_df_list, opt.data_dir, processor=processor, transforms=transforms
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True
    )

    val_dataset = TrocrDataset(
        val_df_list, opt.data_dir, processor=processor, transforms=transforms
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True
    )

    return train_loader, val_loader


def load_model(opt: Any, Logger: logging.Logger) -> torch.nn.DataParallel:
    # TODO Добавить выбор модели через sh скрипт
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-small-handwritten"
    )

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # data parallel for multi-GPU
    
    # Предполагаю, что эта строчка сбрасывает настройки config и beam search
    # model = torch.nn.DataParallel(model)

    # if opt.saved_model != "":
    #     logger.info(f"Loading pretrained trocr_model from {opt.saved_model}")
    #     state_dict = torch.load(opt.saved_model, map_location=device)
    #     model.load_state_dict(state_dict, strict=not opt.ft)

    model.train()
    model = model.to(device)
    return model


def get_training_utils(
    logger: logging.Logger, model: torch.nn.DataParallel, opt: Any
) -> Tuple[Optimizer]:
    pass


def train(opt: Any, logger: logging.Logger) -> None:
    train_loader, val_loader = prepare_data(opt)
    model = load_model(opt, logger)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_averager = Averager()

    start_iter = 0
    if opt.saved_model != "":
        try:
            start_iter = int(opt.saved_model.split("_")[-1].split(".")[0])
            logger.info(f"Continue to train, start_iter: {start_iter}")
        except:
            pass

    start_time, best_accuracy, best_cer, iteration, best_loss, loss_increase_num = (
        time.time(),
        -1,
        np.inf,
        start_iter,
        np.inf,
        0,
    )

    for epoch in range(opt.epochs):

        for batch in tqdm(train_loader):
            for k, v in batch.items():
                batch[k] = v.to(device)

            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            loss_averager.add(loss)
            optimizer.zero_grad()

        # validation part
        elapsed_time = int((time.time() - start_time) / 60.0)
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader):
                # run batch generation
                outputs = model.generate(batch["pixel_values"].to(device))
                # compute metrics
                accuracy_value, cer_value, wer_value = string_accuracy(
                    outputs, batch["labels"]
                ), compute_metrics(outputs, batch["labels"])
                current_cer += cer_value
                current_wer += wer_value
                current_accuracy += accuracy_value

        model.train()

        # training loss and validation loss
        train_loss = loss_averager.val()

        logger.info(
            f"[Epoch {epoch}/{opt.epochs}] Train loss: {train_loss:0.5f}, elapsed time: {elapsed_time} min"
        )
        loss_averager.reset()
        logger.info(
            f'{"Current accuracy":17s}: {current_accuracy:0.3f}, {"current CER":17s}: {current_cer:0.2f}, {"current WER":17s}: {current_wer:0.2f}'
        )
        best_accuracy = (
            current_accuracy if current_accuracy > best_accuracy else best_accuracy
        )
        # keep the best cer attention_model (on valid dataset)
        if current_cer < best_cer:
            best_cer = current_cer
            logger.info("Save attention_model with best CER")
            torch.save(model.state_dict(), os.path.join(opt.out_dir, "best_cer.pth"))
        logger.info(
            f'{"Best accuracy":17s}: {best_accuracy:0.3f}, {"Best CER":17s}: {best_cer:0.2f}'
        )

        # TODO write the code for showing predicted results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir", type=str, help="Directory for saving log file", required=True
    )
    parser.add_argument(
        "--log_name", type=str, help="Name of the log file", required=True
    )
    parser.add_argument("--out_dir", help="Where to store models", required=True)
    parser.add_argument(
        "--data_dir", type=str, help="Path to the dataset", required=True
    )
    parser.add_argument(
        "-n",
        "--label_files",
        nargs="+",
        required=True,
        help="Names of files with labels",
    )
    parser.add_argument(
        "--manual_seed", type=int, default=1111, help="For random seed setting"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Input batch size")
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--saved_model",
        type=str,
        default="",
        help="Path to attention_model to continue training",
    )
    parser.add_argument("--ft", action="store_true", help="Whether to do fine-tuning")

    # doesn't added
    # parser.add_argument('--patience', type=int, default=3, help='Patience for the early stopping')
    # parser.add_argument('--write_errors', action='store_true', help='Write attention_model\'s errors to the log file')

    # Data processing
    # parser.add_argument('--batch_max_length', type=int, default=40, help='Maximum label length')
    # parser.add_argument('--img_h', type=int, default=32, help='The height of the input image')
    # parser.add_argument('--img_w', type=int, default=100, help='The width of the input image')
    # parser.add_argument('--rgb', action='store_true', help='Use rgb input')
    # parser.add_argument('--pad', action='store_true', help='Whether to keep ratio then pad for image resize')

    opt = parser.parse_args()

    os.makedirs(opt.log_dir, exist_ok=True)
    logger = get_logger(out_file=os.path.join(opt.log_dir, opt.log_name))
    os.makedirs(opt.out_dir, exist_ok=True)

    # Seed and GPU setting
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

    cudnn.benchmark = True
    cudnn.deterministic = True

    start_time = time.time()
    train(opt, logger)
    elapsed_time = (time.time() - start_time) / 60.0
    logger.info(f"Overall elapsed time: {elapsed_time:.3f} min")
