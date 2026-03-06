### Donut CORD-v2 Information Extraction

This project provides a **complete, production-ready pipeline** to fine-tune the **DONUT** model (`naver-clova-ix/donut-base`) on the **CORD‑v2** receipt dataset (`naver-clova-ix/cord-v2`) for information extraction.

It uses:
- **PyTorch**
- **Hugging Face Transformers**
- **DonutProcessor** and **VisionEncoderDecoderModel**
- **Seq2SeqTrainer** with mixed precision, gradient accumulation, early stopping, and checkpointing

The model learns to generate structured **JSON** from receipt images, wrapped in task-specific tokens:

```text
<s_cord-v2>{"store_name": "...", "items": [...], ...}</s_cord-v2>
```

---

### Project Structure

- `train.py` — training entrypoint using `Seq2SeqTrainer`
- `dataset.py` — CORD‑v2 loading and preprocessing with `DonutProcessor`
- `metrics.py` — JSON parsing, exact match, and field-level accuracy
- `inference.py` — single-image inference script
- `config.yaml` — configuration for model, data, and training hyperparameters
- `requirements.txt` — Python dependencies
- `README.md` — project documentation

---

### Setup

**1. Create and activate a virtual environment (recommended)**

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**3. Verify GPU availability (optional but recommended)**

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

---

### Configuration

All main hyperparameters and dataset options live in `config.yaml`.

- **Model settings**
  - `pretrained_name`: `"naver-clova-ix/donut-base"`
  - `task_start_token`: `"<s_cord-v2>"`
  - `task_end_token`: `"</s_cord-v2>"`
  - `max_seq_length`: maximum generated sequence length
  - `ignore_pad_token_for_loss`: use `-100` for padded label tokens

- **Data settings**
  - `dataset_name`: `"naver-clova-ix/cord-v2"`
  - `train_split`, `validation_split`: HF split names
  - `image_column`: image feature column (`"image"`)
  - `label_column`: structured annotation column (`"ground_truth"`)
  - `max_train_samples`, `max_eval_samples`: optional subsampling for debugging

- **Training settings**
  - `output_dir`: directory for checkpoints and logs
  - `num_train_epochs`
  - `per_device_train_batch_size`, `per_device_eval_batch_size`
  - `learning_rate`, `weight_decay`, `warmup_ratio`
  - `logging_steps`, `save_steps`, `eval_steps`
  - `gradient_accumulation_steps`
  - `fp16`: enable mixed precision on GPU
  - `early_stopping_patience`
  - `generation_max_length`, `generation_num_beams`

You can adjust these values without touching the code.

---

### Training

Run training with:

```bash
python train.py --config config.yaml
```

What happens during training:
- Images are lazily loaded and converted to `pixel_values` using `DonutProcessor`.
- Ground-truth structures are converted to JSON, then wrapped in `<s_cord-v2> ... </s_cord-v2>`.
- Targets are tokenized, padded to a fixed length, and padded tokens are set to `-100` so they are **ignored in the loss**.
- `Seq2SeqTrainer` runs with:
  - **mixed precision** (`fp16`) when enabled
  - **gradient accumulation** for larger effective batch sizes
  - **early stopping** based on `exact_match`
  - **model checkpointing** in `output_dir`

The best checkpoint (by `exact_match`) is automatically restored at the end of training and saved to `output_dir` together with the processor.

---

### Evaluation and Metrics

During evaluation, the trainer:
- Uses `generate` to produce sequences.
- Decodes predictions and labels with `DonutProcessor`.
- Attempts to parse JSON from each sequence.

`metrics.py` computes:
- **exact_match** — fraction of examples where the predicted JSON equals the ground-truth JSON.
- **field_level_accuracy** — micro-averaged accuracy over flattened key–value pairs:
  - JSON objects/lists are flattened with dot and index notation (for example, `items[0].price`).
  - Only keys present in the ground truth count towards the denominator.

These metrics are logged during training and used for model selection.

---

### Inference

After training, you can run inference on a single image:

```bash
python inference.py \
  --image /path/to/receipt.png \
  --model_dir outputs \
  --config config.yaml
```

This will:
- Load the fine-tuned model and processor from `outputs`.
- Build a task prompt starting with `<s_cord-v2>`.
- Generate a sequence with `VisionEncoderDecoderModel.generate`.
- Print:
  - The **raw model output** string.
  - The **parsed JSON** object (if parsing succeeds).

You can point `--model_dir` to any compatible checkpoint directory created by `train.py`.

---

### GPU Requirements

The CORD‑v2 dataset contains **1,000 high-resolution receipt images**. Training DONUT on this dataset is GPU-intensive.

Practical guidelines:

- **Minimum** (for experimentation / small batch sizes)
  - 1 × GPU with **12 GB** VRAM (e.g. RTX 3060/3060 Ti)
  - Use:
    - Smaller `per_device_train_batch_size` (1–2)
    - Higher `gradient_accumulation_steps` to keep the effective batch size reasonable
    - Possibly reduce `max_seq_length` if memory is tight

- **Recommended** (for smoother training)
  - 1 × GPU with **16–24 GB** VRAM (e.g. RTX 4090, A5000)
  - Default `config.yaml` settings should work with minor tuning.

Training on CPU is technically possible but **not recommended** due to very long runtimes.

---

### Notes and Customization

- To adapt this project to another document IE dataset:
  - Update `dataset_name`, `image_column`, and `label_column` in `config.yaml`.
  - Adjust `_extract_cord_structure` in `dataset.py` to convert your dataset’s label format into a JSON-serializable structure.
  - Keep the task tokens but change them to something dataset-specific (for example, `<s_invoice>` / `</s_invoice>`).

- Logging and outputs:
  - Training logs and metrics are written under `outputs/`.
  - Model checkpoints (including the best checkpoint) are also stored in `outputs/`.

