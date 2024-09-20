import argparse
import json
import logging
import random
from functools import partial

import torch
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from datasets.utils.logging import set_verbosity_info
from numpy.random import default_rng


from clean_helpers import build_dedup_template, build_dedup_document, concatenate_lm_fr_ester
from clean_helpers.deduplication import document_batch_normalizer, url_host_and_path_batch_normalizer, \
    url_lm_es_pseudocrawl_filtered_341_es_cointelegraph_com, url_lm_en_pseudocrawl_filtered_619_www_qut_edu_au



set_verbosity_info()
logger = logging.getLogger(__name__)
torch.set_num_threads(1)

# Deduplication functions and boolean to save a sample of the modifications: function(ds: Dataset, num_proc: int, batch_size: int) -> Dataset
DEDUPS = {
    "dedup_template_soft": (build_dedup_template(
        min_template_line_size=15,
        min_template_line_occurence=10,
    ), True),
    "dedup_pseudocrawl_newspapers": (build_dedup_template(
        min_template_line_size=0,
        min_template_line_occurence=2,
    ), True),
    "dedup_document": (build_dedup_document(document_batch_normalizer), True),
    "dedup_document_on_url": (build_dedup_document(url_host_and_path_batch_normalizer), True),
    "dedup_document_on_url_lm_es_pseudocrawl-filtered_341_es_cointelegraph_com": (build_dedup_document(
        url_lm_es_pseudocrawl_filtered_341_es_cointelegraph_com
    ), True),
    "dedup_document_on_url_lm_en_pseudocrawl_filtered_619_www_qut_edu_au": (build_dedup_document(
        url_lm_en_pseudocrawl_filtered_619_www_qut_edu_au
    ), True),
    "concatenate_lm_fr_ester": (concatenate_lm_fr_ester, False)
}


DEDUPS_KEYS = set(DEDUPS.keys())

def get_size_per_example(texts: List[str]) -> Dict:
    size_values = [len(text.encode()) for text in texts]
    examples = {"bytes_len": size_values}
    return examples

def quick_size_estimation(
    ds: Dataset,
    num_proc: int,
    batch_size: int,
    content_key:str ="text"
) -> int:
    if len(ds) == 0:
        return 0
    rng = default_rng(1991)
    subset_size = min(10000, len(ds))
    indices = rng.choice(len(ds), size=subset_size, replace=False, shuffle=False)
    partial_ds = ds.select(indices)
    ratio = float(len(ds)) / float(subset_size)

    partial_ds = partial_ds.map(
        get_size_per_example,
        batched=True, 
        num_proc=num_proc,
        batch_size=batch_size,
        input_columns=[content_key],
        remove_columns=partial_ds.column_names,
    )
    len_bytes = sum(partial_ds["bytes_len"])
    return len_bytes * ratio




def filter_diff_text(examples, in_text_col, out_text_col):
    return [text_in != text_out for text_in, text_out in zip(examples[in_text_col], examples[out_text_col])]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Dataset path we load the dataset from.")
    parser.add_argument("--output_path", type=Path, required=True,
                        help="Path where we save resulting dataset after modifications.")
    parser.add_argument('--text_column', type=str)
    parser.add_argument("--cache", type=str, required=True, help="Cache Path.")
    parser.add_argument("--checks_save_path", type=Path, default=None,
                        help="Path where we save samples we've removed or changed throughout the modifications.")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--load_arrow_file", action="store_true",
                        help="Option to indicate how to load original dataset. By default we use `load_dataset`. "
                             "If the flag is use, we use `load_from_disk`")
    parser.add_argument("--sampling_size_map_checks", type=int, default=None,
                        help="Optional argument. Checked dataset, ie sample we've changed throughout the "
                             "modifications, are either save in whole or only a subset. If set to None, this flag "
                             "saves everything, otherwise it saves a subset with its size corresponding to this value.")
    parser.add_argument("--sampling_size_filter_checks", type=int, default=None,
                        help="Optional argument. Checked dataset, ie sample we've removed throughout the "
                             "modifications, are either save in whole or only a subset. If set to None, this flag "
                             "saves everything, otherwise it saves a subset with its size corresponding to this value.")
    parser.add_argument("--from_scratch", action="store_true", help="Resave all datasets on disk.")
    parser.add_argument("--save_to_json", default=True, help="Save output dataset in json format.")
    return parser.parse_args()

def log_stats(title: str, original_ds: Dataset, after_transformation_ds: Dataset, operation_type: str, args):
    original_length = len(original_ds)
    after_transformation_length = len(after_transformation_ds)
    original_bytes = quick_size_estimation(original_ds, batch_size=args.batch_size, num_proc=args.num_proc, content_key=args.text_column)
    after_transformation_btyes = quick_size_estimation(after_transformation_ds, batch_size=args.batch_size, num_proc=args.num_proc, content_key=args.text_column)
    logger.info(title)
    logger.info(f"     Initial number of samples: {original_length} samples")
    logger.info(f"     {operation_type} samples: {original_length - after_transformation_length} samples")
    logger.info(f"     {operation_type} percentage: {(original_length - after_transformation_length) / original_length * 100:.2f} %")
    logger.info(f"     Final number of samples: {after_transformation_length} samples")
    logger.info(f"     Initial size in bytes: {original_bytes * 1e-9:.4f} GB")
    logger.info(f"     {operation_type} bytes: {(original_bytes - after_transformation_btyes) * 1e-9:.4f} GB")
    logger.info(f"     {operation_type} percentage in bytes: {(original_bytes - after_transformation_btyes) / original_bytes * 100:.2f} %")
    logger.info(f"     Final size in bytes: {after_transformation_btyes * 1e-9:.4f} GB")



def get_modified_documents(
    ds: Dataset,
    mapped_ds: Dataset,
    num_proc: int,
    batch_size: int,
    sampling_size: Optional[int],
    text_column,
) -> Dataset:
    remove_columns = set(ds.column_names)
    remove_columns.remove(text_column)
    ds = ds.remove_columns(remove_columns)
    ds = ds.rename_column(text_column, f"old_text")

    assert len(mapped_ds) == len(ds), f"Mapping function are batched, but they should not alter the size of the batch."
    mapped_diff_ds = concatenate_datasets([mapped_ds.flatten_indices(), ds.flatten_indices()], axis=1).filter(
        partial(filter_diff_text, in_text_col="old_text", out_text_col=text_column),
        batched=True,
        num_proc=num_proc,
        batch_size=batch_size
    )

    logger.info("Examples of modified examples:")
    idx_samples = random.sample(range(len(mapped_diff_ds)), min(len(mapped_diff_ds), 10))
    for idx in idx_samples:
        logger.info(f"     Examples n°{idx} :\n{json.dumps(mapped_diff_ds[idx], indent=2)}")

    if sampling_size is not None:
        idx_samples = random.sample(range(len(mapped_diff_ds)), min(len(mapped_diff_ds), sampling_size))
        mapped_diff_ds = mapped_diff_ds.select(idx_samples)

    return mapped_diff_ds


def apply_function(function_name: str, ds: Dataset, args) -> Tuple[Dataset, Optional[Dataset]]:
    logger.info(f"Applying: {function_name}")
    if function_name in DEDUPS:
        dedup_function, dedup_check = DEDUPS[function_name]
        deduplicated_ds = dedup_function(ds, num_proc=args.num_proc, batch_size=args.batch_size)
        log_stats(f"Applied deduplication function: {function_name}",  ds,  deduplicated_ds,  operation_type="Deduplicated", args=args)

        # Some deduplication do not preserve the number of samples, so alignement is lost. For example "dedup_document"
        if args.checks_save_path is not None and dedup_check:
            deduped_diff_ds = get_modified_documents(ds, deduplicated_ds, args.num_proc, args.batch_size, args.sampling_size_map_checks, args.text_column)
            return deduplicated_ds, deduped_diff_ds
        else:
            return deduplicated_ds, None
    else:
        raise NotImplementedError(f"{function_name} has not matched any existing function names. Available names:\n"
                                  f"Dedup functions: {DEDUPS_KEYS}\n"
                                  )

def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = get_args()
    logger.info(f"** The job is runned with the following arguments: **\n{args}\n **** ")

    # Load dataset
    logger.info(f" ===== Loading {args.input_path} =====")
    if args.load_arrow_file:
        ds = load_from_disk(args.input_path)
    else:
        ds = load_dataset("json", data_files=args.input_path, split="train", cache_dir= args.cache)

    # Apply series of dedups
    logger.info(f" ===== Applying transformations =====")
    
    preprocessings = ["dedup_template_soft", "dedup_document"]
    for idx, preprocessing in enumerate(preprocessings):
        ds, ds_diff = apply_function(preprocessing, ds, args)
        if ds_diff is not None and len(ds_diff) != 0:
            saving_path = args.checks_save_path / f"{idx}_{preprocessing}_checks"
            if not args.from_scratch and saving_path.exists():
                continue
            tmp_save_path = Path(saving_path.parent, f"tmp-{saving_path.name}")
            logger.info(f" ===== Saving examples to check after {preprocessing}  =====")
            ds_diff.save_to_disk(tmp_save_path)
            tmp_save_path.rename(saving_path)


    # Save to disk
    if args.from_scratch or not args.output_path.exists():
        logger.info(f" ===== Saving dataset =====")
        logger.info(f"Saving to final dataset at {args.output_path}.")
        tmp_save_path = Path(args.output_path.parent, f"tmp-{args.output_path.name}")
        if len(ds) == 0:
            logger.info("Dataset was empty. Not saving anything.")
            return
        if args.save_to_json:
            ds.to_json(
                tmp_save_path,
                num_proc=args.num_proc,
                force_ascii=False
            )
        else:
            ds.save_to_disk(tmp_save_path)
        tmp_save_path.rename(args.output_path)
    else:
        logging.info(f"Dataset was already saved at {args.output_path}")


if __name__ == "__main__":
    main()
