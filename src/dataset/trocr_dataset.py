import torch
import pandas as pd
import os

from PIL import Image
from typing import Any, Tuple, List

from src.dataset.abstract_dataset import AbstractDataset


class TrocrDataset(AbstractDataset):
    def __init__(
        self,
        df_list: List[pd.DataFrame],
        data_dir: str,
        processor,
        max_target_length=128,
        transforms=None,
    ):
        super().__init__(df_list, data_dir)
        self.processor = processor
        self.max_target_length = max_target_length
        self.transforms = transforms

    def __getitem__(self, idx: int) -> dict:
        assert idx <= len(self), "index range error"

        df_idx = idx % len(self.texts)
        current_idx = self.current_idx_list[df_idx]
        self.current_idx_list[df_idx] = (self.current_idx_list[df_idx] + 1) % len(
            self.texts[df_idx]
        )

        # get file name and text
        img_path = os.path.join(self.data_dir, self.paths[df_idx][current_idx])
        text = str(self.texts[df_idx][current_idx])

        # prepare image (convert to grayscale + resize + normalize)
        image = Image.open(img_path).convert("RGB")

        # if self.transforms:
        #     image = self.transforms(image=image)["image"]

        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_target_length
        ).input_ids

        # important: make sure that PAD tokens are ignored by the loss function
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]

        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels),
        }
        return encoding
