# RAHF

## Project Overview

RAPD is a framework for detecting AI-generated text. Its core method is **PGR**. The framework fine-tunes a language model to distinguish human-written text from text generated or edited by large language models (LLMs).

The framework treats human-written text as dispreferred and machine-revised text as preferred. Through preference learning, the scoring model learns the stylistic differences between them and uses those differences to detect machine-written text.

## Method Overview

### PGR Training Workflow

```
Original human text (dispreferred) ──┐
                                     ├──▶ PGR Loss + KL Regularization ──▶ Scoring Model (LoRA Fine-Tuning)
Machine-revised text (preferred)  ───┘
```

1. **Scoring model** uses GPT-Neo 2.7B with parameter-efficient LoRA fine-tuning (r=8, alpha=32).
2. **Reference model** is a frozen pretrained language model that provides a baseline language distribution.
3. **Scoring function** combines sequence-level and local window discrepancies and uses KL divergence for regularization.
4. **PGR loss** uses preference learning, with β controlling the strength of the preference.

### Three Core PGR Components

The PGR method combines three components. The source files implement different combinations for ablation experiments.

| Component | Description | Source file |
|-----------|-------------|-------------|
| **PGR** | Applies bounded probability aggregation to improve scoring robustness | `PGR.py` |
| **DAC** | Corrects systematic bias in scoring | `DAC.py` |
| **MSDD** | Uses dynamic semantic windows to capture local features | `MSDD.py` |
| All components | Full method combining all three components | `RAHF_full.py` |
| Baseline (no components) | Comparison baseline with all three components removed | `baseline.py` |

### Scoring Function

- **Sequence-level discrepancy** measures the shift in the probability distribution over the full text.
- **Window-level discrepancy** captures fine-grained n-gram style features using weights from dynamic semantic windows.
- **KL divergence regularization** constrains the scoring model relative to the reference distribution.

## Project Structure

```
├── dpr/                          # Core model and training code
│   ├── RAHF.py                   # PGR model definition (ComputeScore class and scoring function)
│   ├── engine.py                 # Training and evaluation engine
│   ├── run_RAHF.py               # PGR training entry point
│   ├── dataset.py                # Dataset loading (CustomDataset classes)
│   ├── metrics.py                # Evaluation metrics (ROC AUC and PR AUC)
│   ├── inference.py              # Interactive inference demo
│   ├── evaluation.py             # Standalone evaluation script
│   ├── utils_RAHF.py             # PGR loss computation
│   ├── utils_visualization.py    # Training plots (gradient traps, NLL distributions, token signals)
│   ├── demo.py                   # Visualization demo with an example training loop
│   │
│   │ # Ablation files for different component combinations
│   ├── baseline.py               # Baseline without any components
│   ├── PGR.py                    # Bounded probability aggregation only
│   ├── DAC.py                    # Bias reduction only
│   ├── MSDD.py                   # Dynamic semantic windows only
│   ├── PGR_DAC.py                # Two-component combination
│   ├── PGR_MSDD.py               # Two-component combination
│   ├── DAC_MSDD.py               # Two-component combination
│   └── RAHF_full.py              # Full method combining all three components
│
├── tools/                        # Data construction and utility scripts
│   ├── data_builder.py           # LLM data generation (polish, rewrite, expand, generation)
│   ├── data_builder_gpts/        # Data generation notebooks for GPT models
│   ├── show_results.py           # Results display
│   ├── count_words.py            # Word counts
│   ├── truncation.py             # Text truncation experiments
│   └── solve_llama_output.py     # Llama output post-processing
│
├── scripts/                      # Shell scripts for training, evaluation, and downloads
│   ├── train_RAHF.sh             # PGR training
│   ├── eval_all.sh               # Full evaluation (polish, rewrite, generation, expand)
│   ├── eval_RAHF_*.sh            # Evaluation by task
│   ├── eval_ablation.sh          # Ablation experiments
│   ├── build_data.sh             # Data construction
│   └── download_*.sh             # Model downloads
│
├── data/                         # Datasets
│   ├── ai_detection_*_polish*.json   # Training datasets with 500 or 3,000 examples
│   ├── polish/                   # Polish data organized by model
│   ├── rewrite/                  # Rewrite data
│   ├── expand/                   # Expand data
│   ├── generation/               # Generation data
│   ├── truncation/               # Data for truncation length experiments
│   └── multilang/                # Multilingual data (Chinese, Spanish, and Portuguese)
│
├── results/                      # Experimental results
│   ├── main/                     # Main experiment results organized by task and model
│   └── ablation/                 # Ablation results comparing SFT, ORPO, and RLHF
│
├── evaluation/                   # ROC curves, scatter plots, and token highlight plots
├── logs/                         # TensorBoard training logs
└── local_infer_ref/              # Reference distributions for local inference
```

## Supported Detection Tasks

| Task | Description | Example |
|------|-------------|---------|
| **Polish** | An LLM polishes or revises human-written text | Improve the wording while preserving the meaning |
| **Rewrite** | An LLM paraphrases human-written text | Replace words and restructure sentences |
| **Expand** | An LLM expands human-written text | Add details and develop the argument |
| **Generation** | An LLM continues a given opening to produce a complete text | Generate a full passage from a prefix |

## Supported Models

### Base Scoring Model

- GPT-Neo 2.7B (default, fine-tuned with LoRA)

### LLMs Used to Generate Machine Text

- **Closed-source models** include GPT-3.5-Turbo and GPT-4o.
- **Open-source models** include Llama-3-8B, Mistral-7B, Qwen2-7B, and Deepseek-7b.

### Evaluation Datasets

- **XSum** contains news summaries.
- **Writing** contains creative writing.
- **PubMed** contains medical literature.
- **SQuAD** contains reading comprehension examples.

## Quick Start

### Dependencies

```bash
pip install torch transformers peft accelerate
pip install scikit-learn matplotlib tensorboard
pip install tqdm numpy
```

### 1. Build Data

Use an open-source LLM to generate machine text.

```bash
# Polish the XSum dataset with Llama-3-8B
python tools/data_builder.py \
    --model_name Llama-3-8B \
    --dataset xsum \
    --task polish \
    --dataset_file ./data/xsum \
    --result_dir ./data/polish/Llama-3-8B
```

### 2. Train PGR

```bash
# Train for two epochs on 500 Polish examples
python dpr/run_RAHF.py \
    --datanum 500 \
    --task_name ai_detection_polish_500 \
    --epochs 2 \
    --lr 1e-4 \
    --beta 0.05
```

- `--lr` sets the learning rate (default `1e-4`).
- `--beta` sets the PGR loss coefficient β (default `0.05`).
- `-a` sets the number of gradient accumulation steps (default `1`).
- `--datanum` sets the number of training examples (default `500`).

### 3. Evaluate

```bash
# Evaluate a single dataset
python dpr/evaluation.py \
    --eval_dataset ./data/polish/gpt-3.5-turbo/xsum_polish_gpt-3.5-turbo.raw_data.json

# Run the full evaluation
bash scripts/eval_all.sh
```

### 4. Run Interactive Inference

```bash
python dpr/inference.py --task polish --detail True
```

Enter a passage to see the model's estimated probability that it was machine-generated.

## Evaluation Metrics

- **ROC AUC** is the area under the receiver operating characteristic curve. It measures overall discrimination.
- **PR AUC** is the area under the precision-recall curve. It measures performance on imbalanced data.
- **Reward Accuracy** measures how often the PGR model orders the preferred and dispreferred examples correctly during training.
- **Reward Margin** measures the difference between the rewards for preferred and dispreferred examples.

## Technical Details

### LoRA Configuration

```python
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,              # LoRA rank
    lora_alpha=32,     # Scaling factor
    lora_dropout=0.1   # Dropout rate
)
```

### PGR Loss Function

```python
loss = -log_sigmoid(β * (r_preferred - r_dispreferred))
```

Here, `r` is the log probability under the scoring model minus the log probability under the reference model. The coefficient β controls the strength of the preference.

### KL Divergence Regularization

A KL divergence term augments the PGR loss to constrain the scoring model relative to the reference distribution.

```
total_loss = pgr_loss + λ * KL(q_score || p_ref)
```

## Visualization

1. **Sequence-level vs. window-level scatter plots** show the distribution of human and machine text across the two scoring dimensions.
2. **Token-level criterion highlights** mark tokens in machine text that the detector identifies as machine-like.
3. **Training visualizations** compare gradient trap distributions, NLL distributions, and token-level signals.
4. **TensorBoard logs** track training loss, ROC AUC, and PR AUC during training.

## Ablation Experiments

The project includes ablation experiments comparing several baseline methods.

**PGR component ablations** (source files in `dpr/`)

- Baseline with no components uses `baseline.py`.
- Single-component variants use `PGR.py`, `DAC.py`, and `MSDD.py`.
- Two-component variants use `PGR_DAC.py`, `PGR_MSDD.py`, and `DAC_MSDD.py`.
- The full method with all three components uses `RAHF_full.py`.

**Training strategy ablations** (`results/ablation/`)

- **SFT** uses standard supervised fine-tuning with 1,000, 2,000, or 3,000 examples.
- **ORPO** stands for Odds Ratio Preference Optimization.
- **RLHF** stands for reinforcement learning from human feedback.

## Citation

If you use this project's code or method, please cite the relevant paper.

## License

This project is for academic research use only.
