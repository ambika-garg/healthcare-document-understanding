import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from transformers import DonutProcessor
from transformers.trainer_utils import EvalPrediction


def _extract_json_substring(text: str) -> str:
    """
    Heuristically extract the JSON object from a Donut output string.

    Donut is trained to emit task tokens around a JSON object. At inference
    time there can be extra tokens or whitespace, so we locate the first '{'
    and the last '}' and parse the substring between them.
    """
    if not text:
        return ""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def _safe_json_loads(text: str) -> Any:
    """
    Parse JSON from text, returning None on failure.
    """
    json_str = _extract_json_substring(text)
    if not json_str:
        return None
    try:
        return json.loads(json_str)
    except Exception:
        return None


def _flatten_dict(prefix: str, value: Any, out: Dict[str, str]) -> None:
    """
    Flatten nested dicts/lists into dot-separated keys for field-level metrics.
    """
    if isinstance(value, dict):
        for k, v in value.items():
            next_prefix = f"{prefix}.{k}" if prefix else str(k)
            _flatten_dict(next_prefix, v, out)
    elif isinstance(value, list):
        for idx, v in enumerate(value):
            next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            _flatten_dict(next_prefix, v, out)
    else:
        out[prefix] = "" if value is None else str(value)


def _compute_example_field_accuracy(
    pred_obj: Any,
    label_obj: Any,
) -> Tuple[int, int]:
    """
    Compute the number of correct fields and total fields for a single example.

    Only fields present in the ground truth are counted towards the total.
    """
    if not isinstance(label_obj, (dict, list)):
        return 0, 0

    gt_flat: Dict[str, str] = {}
    _flatten_dict("", label_obj, gt_flat)

    if not isinstance(pred_obj, (dict, list)):
        return 0, len(gt_flat)

    pred_flat: Dict[str, str] = {}
    _flatten_dict("", pred_obj, pred_flat)

    correct = 0
    for key, gt_value in gt_flat.items():
        if key in pred_flat and pred_flat[key] == gt_value:
            correct += 1
    return correct, len(gt_flat)


@dataclass
class DonutMetrics:
    """
    Callable metrics helper for Seq2SeqTrainer.

    It decodes predictions and labels using the DonutProcessor tokenizer,
    parses JSON, and returns:
        - exact_match: strict JSON equality
        - field_level_accuracy: micro-averaged field accuracy over all examples
    """

    processor: DonutProcessor

    def __call__(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        predictions, label_ids = eval_pred

        # Generated output ids when predict_with_generate=True
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Replace -100 in labels before decoding
        labels = np.where(label_ids != -100, label_ids, self.processor.tokenizer.pad_token_id)

        pred_texts: List[str] = self.processor.batch_decode(
            predictions, skip_special_tokens=True
        )
        label_texts: List[str] = self.processor.batch_decode(
            labels, skip_special_tokens=True
        )

        exact_matches = 0
        total_examples = len(pred_texts)

        total_fields = 0
        total_correct_fields = 0

        for pred_str, label_str in zip(pred_texts, label_texts):
            pred_obj = _safe_json_loads(pred_str)
            label_obj = _safe_json_loads(label_str)

            if pred_obj is not None and label_obj is not None and pred_obj == label_obj:
                exact_matches += 1

            correct, total = _compute_example_field_accuracy(pred_obj, label_obj)
            total_correct_fields += correct
            total_fields += total

        exact_match = exact_matches / total_examples if total_examples > 0 else 0.0
        field_accuracy = total_correct_fields / total_fields if total_fields > 0 else 0.0

        return {
            "exact_match": float(exact_match),
            "field_level_accuracy": float(field_accuracy),
        }

