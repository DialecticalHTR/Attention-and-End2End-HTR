# it solves the problem for import modules
import sys
import pathlib


current_path = pathlib.Path(__file__).parent.resolve()
working_dir_path = pathlib.Path().resolve()
repo_name = current_path.relative_to(working_dir_path).parts[0]
repo_path = working_dir_path / repo_name

sys.path.append(repo_path)

import argparse
import logging
import os
from typing import List

from abstract_dataset_processor import AbstractDatasetProcessor
from cyrillic_processor import CyrillicDatasetProcessor
from hkr_processor import HKRDatasetProcessor
from synthetic_processor import SyntheticDatasetProcessor
from utils.logger import get_logger


def get_processors_list(logger: logging.Logger) -> List[AbstractDatasetProcessor]:
    processors = [
        CyrillicDatasetProcessor(logger=logger),
        HKRDatasetProcessor(logger=logger),
        SyntheticDatasetProcessor(logger=logger),
    ]
    return processors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--datasets_list', nargs='+', required=True, help='List of datasets names, options: '
                                                                                'hkr, cyrillic, synthetic_[hkr|cyrillic], '
                                                                                'gan_[hkr|cyrillic], stackmix_[hkr|cyrillic]')
    parser.add_argument('--out_dir', type=str, help='Directory for saving dataset', required=True)
    parser.add_argument('--log_dir', type=str, help='Directory for saving log file', required=True)
    parser.add_argument('--log_name', type=str, help='Name of the log file', required=True)
    opt = parser.parse_args()

    os.makedirs(opt.out_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)
    logger = get_logger(out_file=os.path.join(opt.log_dir, opt.log_name))
    processors = get_processors_list(logger)

    for dataset_name in opt.datasets_list:
        for p in processors:
            if not p.can_process(dataset_name):
                continue

            img_dir = f"img_{dataset_name}"
            os.makedirs(os.path.join(opt.out_dir, img_dir), exist_ok=True)
            p.process_dataset(opt.out_dir, img_dir, f"gt_{dataset_name}.csv", dataset_name)
