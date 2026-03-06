import json
from typing import Any, Dict, Tuple

from datasets import DatasetDict, load_dataset
from PIL.Image import Image
from transformers import DonutProcessor


def _extract_cord_structure(raw_label: Any) -> Any:
    """
    Extract the structured annotation from a CORD-v2 label field.

    The exact schema of the Hugging Face CORD-v2 dataset may evolve, so this
    helper tries a few common patterns and falls back to returning the label
    as-is if it already looks like a JSON-serializable structure.
    """
    if isinstance(raw_label, dict):
        # Common pattern in Donut examples: label["gt_parse"] holds the JSON
        if "gt_parse" in raw_label and isinstance(raw_label["gt_parse"], dict):
            return raw_label["gt_parse"]
        return raw_label
    return raw_label


def _structure_to_target_text(
    structure: Any,
    task_start_token: str,
    task_end_token: str,
) -> str:
    """
    Convert a structured CORD annotation into a Donut target sequence.

    The model is trained to generate a JSON string between task-specific
    start and end tokens, e.g.:
        <s_cord-v2>{"store_name": "..."}</s_cord-v2>
    """
    json_str = json.dumps(structure, ensure_ascii=False)
    return f"{task_start_token}{json_str}{task_end_token}"


def _build_transform(
    processor: DonutProcessor,
    image_column: str,
    label_column: str,
    task_start_token: str,
    task_end_token: str,
    max_seq_length: int,
    ignore_pad_token_for_loss: bool,
):
    """
    Create a lazy transform to be applied with `dataset.with_transform`.

    This keeps the underlying HF dataset on disk and only converts images
    and labels to tensors when they are actually loaded by the Trainer.
    """

    pad_token_id = processor.tokenizer.pad_token_id

    def transform(example: Dict[str, Any]) -> Dict[str, Any]:
        image: Image = example[image_column]
        raw_label = example[label_column]

        structure = _extract_cord_structure(raw_label)
        target_sequence = _structure_to_target_text(
            structure=structure,
            task_start_token=task_start_token,
            task_end_token=task_end_token,
        )

        # Keep batch dim (1, C, H, W) so model always receives 4D input
        pixel_values = processor(
            image,
            return_tensors="pt",
        ).pixel_values

        # Tokenize the target sequence
        tokenized = processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].squeeze(0)

        labels = input_ids.clone()
        if ignore_pad_token_for_loss:
            labels[labels == pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

    return transform


def load_cord_datasets(
    processor: DonutProcessor,
    dataset_name: str,
    train_split: str,
    validation_split: str,
    image_column: str,
    label_column: str,
    task_start_token: str,
    task_end_token: str,
    max_seq_length: int,
    ignore_pad_token_for_loss: bool,
    max_train_samples: int = None,
    max_eval_samples: int = None,
) -> Tuple[DatasetDict, str, str]:
    """
    Load the CORD-v2 dataset from Hugging Face and attach lazy transforms.

    Returns:
        - a DatasetDict with "train" and "validation" entries, each having a
          transform that yields `pixel_values` and `labels`.
        - the name of the train split key
        - the name of the validation split key
    """
    raw_datasets = load_dataset(dataset_name)

    if train_split not in raw_datasets or validation_split not in raw_datasets:
        raise ValueError(
            f"Expected splits '{train_split}' and '{validation_split}' in dataset "
            f"'{dataset_name}', but got: {list(raw_datasets.keys())}"
        )

    train_dataset = raw_datasets[train_split]
    eval_dataset = raw_datasets[validation_split]

    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
    if max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))

    transform = _build_transform(
        processor=processor,
        image_column=image_column,
        label_column=label_column,
        task_start_token=task_start_token,
        task_end_token=task_end_token,
        max_seq_length=max_seq_length,
        ignore_pad_token_for_loss=ignore_pad_token_for_loss,
    )

    train_dataset = train_dataset.with_transform(transform)
    eval_dataset = eval_dataset.with_transform(transform)

    return DatasetDict({"train": train_dataset, "validation": eval_dataset}), "train", "validation"

