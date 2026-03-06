import argparse
import logging
import os
from typing import Any, Dict

import numpy as np
import torch
import yaml
from transformers import (
    DonutProcessor,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    set_seed,
)

from dataset import load_cord_datasets
from metrics import DonutMetrics


LOGGER = logging.getLogger(__name__)


def donut_collate_fn(features: list) -> Dict[str, torch.Tensor]:
    """Batch pixel_values into (B, C, H, W) and labels into (B, L)."""
    pixel_values_list = []
    labels_list = []

    for f in features:
        pv = f["pixel_values"]
        # Ensure a batch dimension is present: (C, H, W) -> (1, C, H, W)
        if pv.ndim == 3:
            pv = pv.unsqueeze(0)
        pixel_values_list.append(pv)

        labels_list.append(f["labels"])

    pixel_values = torch.cat(pixel_values_list, dim=0)
    labels = torch.stack(labels_list)

    return {"pixel_values": pixel_values, "labels": labels}


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Donut on CORD-v2.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    setup_logging()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config = load_config(args.config)

    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    set_seed(int(train_cfg.get("seed", 42)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Using device: %s", device)

    LOGGER.info("Loading processor and model: %s", model_cfg["pretrained_name"])
    processor = DonutProcessor.from_pretrained(model_cfg["pretrained_name"])
    model = VisionEncoderDecoderModel.from_pretrained(model_cfg["pretrained_name"])

    # Configure decoder start token for the CORD-v2 task
    task_start_token = model_cfg["task_start_token"]
    task_end_token = model_cfg["task_end_token"]

    decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(task_start_token)
    if decoder_start_token_id is None:
        raise ValueError(
            f"Task start token '{task_start_token}' is not part of the tokenizer vocabulary."
        )

    model.config.decoder_start_token_id = decoder_start_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    model.to(device)

    # Load datasets with lazy preprocessing
    LOGGER.info("Loading and preparing CORD-v2 datasets.")
    datasets, train_split_name, eval_split_name = load_cord_datasets(
        processor=processor,
        dataset_name=data_cfg["dataset_name"],
        train_split=data_cfg["train_split"],
        validation_split=data_cfg["validation_split"],
        image_column=data_cfg["image_column"],
        label_column=data_cfg["label_column"],
        task_start_token=task_start_token,
        task_end_token=task_end_token,
        max_seq_length=int(model_cfg["max_seq_length"]),
        ignore_pad_token_for_loss=bool(model_cfg["ignore_pad_token_for_loss"]),
        max_train_samples=(
            int(data_cfg["max_train_samples"]) if data_cfg.get("max_train_samples") else None
        ),
        max_eval_samples=(
            int(data_cfg["max_eval_samples"]) if data_cfg.get("max_eval_samples") else None
        ),
    )

    train_dataset = datasets[train_split_name]
    eval_dataset = datasets[eval_split_name]

    output_dir = train_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Dataset uses with_transform(); column_names stay as image/ground_truth but
    # each batch is transform output (pixel_values, labels). Disable column removal.
    log_dir = os.path.join(output_dir, "logs")
    os.environ["TENSORBOARD_LOGGING_DIR"] = log_dir

    batch_size = int(train_cfg["per_device_train_batch_size"])
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    num_epochs = float(train_cfg["num_train_epochs"])
    num_training_steps = int(
        (len(train_dataset) / (batch_size * grad_accum)) * num_epochs
    )
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))
    warmup_steps = int(num_training_steps * warmup_ratio)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps,
        logging_steps=int(train_cfg.get("logging_steps", 50)),
        save_steps=int(train_cfg.get("save_steps", 1000)),
        eval_steps=int(train_cfg.get("eval_steps", 1000)),
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        gradient_accumulation_steps=grad_accum,
        fp16=bool(train_cfg.get("fp16", False)),
        predict_with_generate=True,
        generation_max_length=int(train_cfg.get("generation_max_length", model_cfg["max_seq_length"])),
        generation_num_beams=int(train_cfg.get("generation_num_beams", 1)),
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        report_to=["none"],
        dataloader_num_workers=int(data_cfg.get("num_workers", 4)),
        remove_unused_columns=False,
    )

    metrics = DonutMetrics(processor=processor)

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=int(train_cfg.get("early_stopping_patience", 3)),
        early_stopping_threshold=0.0,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=donut_collate_fn,
        compute_metrics=metrics,
        callbacks=[early_stopping],
    )

    LOGGER.info("Starting training.")
    train_result = trainer.train()

    # Save final model and tokenizer
    LOGGER.info("Training finished. Saving model to %s", output_dir)
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # Save training metrics for inspection
    metrics_path = os.path.join(output_dir, "train_results.npy")
    np.save(metrics_path, train_result.metrics)
    LOGGER.info("Training metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
