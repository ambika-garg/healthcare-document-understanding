import argparse
import json
import logging
import os
from typing import Any

import torch
import yaml
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from metrics import _extract_json_substring


LOGGER = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def run_inference(
    image_path: str,
    model_dir: str,
    config_path: str,
) -> Any:
    """
    Load a fine-tuned Donut model and run inference on a single image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    task_start_token = model_cfg["task_start_token"]
    max_seq_length = int(model_cfg.get("max_seq_length", 512))
    num_beams = int(train_cfg.get("generation_num_beams", 4))

    LOGGER.info("Loading model and processor from %s", model_dir)
    processor = DonutProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Build decoder start prompt for the CORD-v2 task
    decoder_input_ids = processor.tokenizer(
        task_start_token,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=max_seq_length,
            num_beams=num_beams,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    sequence = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    json_str = _extract_json_substring(sequence)

    try:
        parsed = json.loads(json_str) if json_str else None
    except Exception:
        parsed = None

    return {
        "raw": sequence,
        "json": parsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned Donut model.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="outputs",
        help="Directory containing the fine-tuned model and processor.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    setup_logging()
    result = run_inference(
        image_path=args.image,
        model_dir=args.model_dir,
        config_path=args.config,
    )

    print("=== Raw model output ===")
    print(result["raw"])
    print("\n=== Parsed JSON (if available) ===")
    if result["json"] is not None:
        print(json.dumps(result["json"], ensure_ascii=False, indent=2))
    else:
        print("None (could not parse JSON)")


if __name__ == "__main__":
    main()

