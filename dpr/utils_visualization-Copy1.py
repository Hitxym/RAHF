# Training metrics and semantic visualization utilities.
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
# Configure fonts and plot output for a headless environment.
plt.switch_backend('Agg')  # Use a noninteractive backend so tkinter is unnecessary.
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']  # Provide fallback fonts for plot labels.
plt.rcParams['axes.unicode_minus'] = False  # Render minus signs using the selected fonts.
plt.rcParams['figure.dpi'] = 150  # On-screen plot resolution.
plt.rcParams['savefig.dpi'] = 300 # Saved plot resolution.
sns.set_style("whitegrid")
COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E", "#577590"] # Plot palette.

# Training metric collector.
class TrainMetricCollector:
    def __init__(self):
        self.metrics = {
            "total_loss": [], "spo_loss": [], "kl_reg_loss": [],
            "reward_acc": [], "reward_margin": [],
            "human_kl_mean": [], "machine_kl_mean": [],
            "human_discrepancy": [], "machine_discrepancy": [],
            "seq_discrepancy": [], "local_discrepancy": [],
            "log_probs_mean": []
        }
    def update(self, model_output):
        # Collect metrics from the dictionary returned by RAHF.py.
        self.metrics["total_loss"].append(model_output["loss"].detach().cpu().item())
        self.metrics["spo_loss"].append(model_output["spo_loss"].detach().cpu().item())
        self.metrics["kl_reg_loss"].append(model_output["kl_reg_loss"].detach().cpu().item())
        self.metrics["reward_acc"].append(model_output["reward_metrics"]["reward_acc"])
        self.metrics["reward_margin"].append(model_output["reward_metrics"]["reward_margin"])
        self.metrics["human_kl_mean"].append(model_output["kl_stats"]["human_kl_mean"])
        self.metrics["machine_kl_mean"].append(model_output["kl_stats"]["machine_kl_mean"])
        self.metrics["human_discrepancy"].extend(model_output["crit"][1].detach().cpu().numpy())
        self.metrics["machine_discrepancy"].extend(model_output["crit"][3].detach().cpu().numpy())
        # Sequence and local discrepancies.
        seq_val = model_output["components"]["train_original"]["seq_discrepancy"].mean().detach().cpu().item()
        local_val = model_output["components"]["train_original"]["local_discrepancy"].mean().detach().cpu().item()
        self.metrics["seq_discrepancy"].append(seq_val)
        self.metrics["local_discrepancy"].append(local_val)
        # Mean log probabilities.
        log_prob_val = model_output["components"]["train_original"]["label_log_probs"].mean().detach().cpu().item()
        self.metrics["log_probs_mean"].append(log_prob_val)

    def get_data(self, key):
        return np.array(self.metrics[key])

# Generate plots for all training metrics.
def plot_all_visualizations(collector, save_root_dir, epoch, prefix=""):
    vis_dir = os.path.join(save_root_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    prefix = f"{prefix}epoch_{epoch+1:03d}"

    # Plot the loss curves.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    steps = np.arange(len(collector.get_data("total_loss")))
    axes[0].plot(steps, collector.get_data("total_loss"), label="total_loss", color=COLORS[0], linewidth=1.5)
    axes[0].plot(steps, collector.get_data("spo_loss"), label="spo_loss", color=COLORS[1], linewidth=1.5, linestyle="--")
    axes[0].plot(steps, collector.get_data("kl_reg_loss"), label="kl_reg_loss", color=COLORS[2], linewidth=1.5, linestyle=":")
    axes[0].set_title("Training Loss Curves", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Loss Value")
    axes[0].legend()
    axes[0].grid(True, ls='--', alpha=0.3)

    # Plot KL divergence.
    axes[1].plot(steps, collector.get_data("human_kl_mean"), label="Human KL", color=COLORS[3], linewidth=1.5)
    axes[1].plot(steps, collector.get_data("machine_kl_mean"), label="Machine KL", color=COLORS[4], linewidth=1.5, linestyle="--")
    axes[1].set_title("KL Divergence Curves", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("KL Mean Value")
    axes[1].legend()
    axes[1].grid(True, ls='--', alpha=0.3)

    # Plot the reward metrics.
    axes[2].plot(steps, collector.get_data("reward_acc"), label="Reward Acc", color=COLORS[5], linewidth=1.5)
    axes[2].plot(steps, collector.get_data("reward_margin"), label="Reward Margin", color=COLORS[0], linewidth=1.5, linestyle="--")
    axes[2].set_title("Reward Metrics Curves", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("Reward Value")
    axes[2].legend()
    axes[2].grid(True, ls='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{prefix}_loss_kl_reward.png"), dpi=200)
    plt.close()

    # Draw the box plots.
    human_data = collector.get_data("human_discrepancy")
    machine_data = collector.get_data("machine_discrepancy")
    plot_data = pd.DataFrame({
        "Discrepancy Value": np.concatenate([human_data, machine_data]),
        "Text Type": ["Human"]*len(human_data) + ["Machine"]*len(machine_data)
    })
    fig = plt.figure(figsize=(8, 6))
    sns.boxplot(x="Text Type", y="Discrepancy Value", hue="Text Type", data=plot_data, 
                palette=[COLORS[0], COLORS[1]], width=0.5, legend=False)
    sns.stripplot(x="Text Type", y="Discrepancy Value", data=plot_data, color="black", alpha=0.2, size=2)
    plt.text(0, np.mean(human_data), f"Mean: {np.mean(human_data):.3f}", ha="center", va="bottom", fontweight="bold")
    plt.text(1, np.mean(machine_data), f"Mean: {np.mean(machine_data):.3f}", ha="center", va="bottom", fontweight="bold")
    plt.title("Discrepancy Value Distribution (Human vs Machine)", fontsize=12, fontweight="bold")
    plt.grid(True, ls='--', alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{prefix}_discrepancy_boxplot.png"), dpi=200)
    plt.close()
    print(f"✅ All {prefix} visualization plots saved to: {vis_dir}")

# Plot the semantic importance heatmap.
def plot_semantic_heatmap(model, sample_text, save_root_dir, epoch, prefix=""):
    vis_dir = os.path.join(save_root_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    prefix = f"{prefix}epoch_{epoch+1:03d}"

    tokenizer = model.scoring_tokenizer
    tokenized = tokenizer(sample_text, return_tensors="pt", padding=True).to(model.device)
    input_ids = tokenized.input_ids[:, :-1]
    attention_mask = tokenized.attention_mask[:, :-1]
    with torch.no_grad():
        logits = model.scoring_model(input_ids, attention_mask=attention_mask).logits
        lprobs = torch.log_softmax(logits, dim=-1)
        label_log_probs = lprobs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    mean_ref = torch.zeros_like(label_log_probs)
    var_ref = torch.ones_like(label_log_probs)
    importance = semantic_importance_scoring(label_log_probs, mean_ref, var_ref)
    importance_np = importance.cpu().numpy()[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

    fig = plt.figure(figsize=(len(tokens)*0.5, 3))
    sns.heatmap(importance_np.reshape(1, -1), cmap="RdYlBu_r", annot=True, fmt=".3f",
                xticklabels=tokens, yticklabels=["Semantic Importance"], cbar=True)
    plt.title(f"Semantic Importance Heatmap - {prefix}", fontsize=10, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{prefix}_semantic_heatmap.png"), dpi=200)
    plt.close()
    print(f"✅ {prefix} Semantic heatmap saved to: {vis_dir}")
def semantic_importance_scoring(log_probs, mean_ref, var_ref):
    """
    Estimate semantic importance from log probability deviation and entropy.
    Preserve the input dtype in FP16 and FP32.
    Args:
        log_probs: Label log probabilities in FP16 or FP32 [batch_size, seq_len].
        mean_ref: Reference distribution means [batch_size, seq_len].
        var_ref: Reference distribution variances [batch_size, seq_len].
    Returns:
        importance: Scores between 0 and 1 [batch_size, seq_len] with the input dtype.
    """
    # Preserve the input dtype and device.
    dtype = log_probs.dtype
    device = log_probs.device
    
    # 1. Measure deviation from the reference distribution with numerical safeguards.
    prob_deviation = torch.abs(log_probs - mean_ref) / torch.sqrt(var_ref.clamp_min(1e-8))
    # 2. Use lower entropy as a signal of greater semantic importance.
    probs = torch.exp(log_probs.clamp(max=0))  # Prevent overflow in exp.
    entropy = -torch.sum(probs * log_probs.clamp(min=-100), dim=-1, keepdim=True)  # Clamp log probabilities for numerical stability.
    # 3. Normalize to the range [0, 1] while preserving the dtype.
    prob_deviation = F.normalize(prob_deviation, p=1, dim=1)
    entropy = F.normalize(entropy, p=1, dim=1)
    # 4. Combine higher deviation and lower entropy into a higher importance score.
    importance = (1 - entropy) * prob_deviation
    importance = F.normalize(importance, p=1, dim=1)  # Apply final normalization.
    
    # Match the output dtype to the input dtype.
    return importance.to(dtype=dtype, device=device)

# Plot per-token criterion values and highlight them by threshold during evaluation.
def plot_crit_token_align_visual(human_tokens, human_crits, llm_tokens, llm_crits, threshold, 
                                 save_dir, epoch, prefix="", crit_name="Token-Corresponding Crit"):
    """
    Match each displayed token to its criterion value and style it according to the threshold.
    Flatten nested arrays and clean invalid values before indexing.
    """
    # Flatten the data into a numeric vector before indexing.
    # Convert nested lists and arrays to a one-dimensional numeric list.
    def flatten_to_1d(arr):
        if isinstance(arr, (list, np.ndarray)):
            return np.array([float(x) for x in np.array(arr).flatten()])
        else:
            return np.array([float(arr)])
    # Flatten nested numeric values.
    human_crits = flatten_to_1d(human_crits)
    llm_crits = flatten_to_1d(llm_crits)
    human_tokens = np.array(human_tokens).flatten()
    llm_tokens = np.array(llm_tokens).flatten()

    # Filter NaN, infinite, and outlier values.
    human_crits = np.nan_to_num(human_crits, nan=0.0, posinf=threshold*2, neginf=0.0)
    llm_crits = np.nan_to_num(llm_crits, nan=0.0, posinf=threshold*2, neginf=0.0)
    
    # Filter invalid tokens to keep the labels readable.
    invalid_tokens = ['<pad>', '<bos>', '<eos>', '[CLS]', '[SEP]', '<unk>', '']
    def filter_valid_token_crit(tokens, crits):
        valid_t = []
        valid_c = []
        for t, c in zip(tokens, crits):
            if t not in invalid_tokens and len(str(t).strip())>0:
                valid_t.append(t)
                valid_c.append(c)
        return valid_t, np.array(valid_c)
    
    human_tokens, human_crits = filter_valid_token_crit(human_tokens, human_crits)
    llm_tokens, llm_crits = filter_valid_token_crit(llm_tokens, llm_crits)
    
    # Handle cases where filtering removes all tokens.
    if len(human_tokens) == 0: human_tokens, human_crits = ["[EMPTY]"], np.array([0.0])
    if len(llm_tokens) == 0: llm_tokens, llm_crits = ["[EMPTY]"], np.array([0.0])

    # Create the output directory and filename.
    vis_dir = os.path.join(save_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    save_name = f"{prefix}epoch_{epoch+1:03d}_token_crit_align_threshold.png"
    save_path = os.path.join(vis_dir, save_name)
    
    # Use side-by-side panels for human and machine text.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(len(max(human_tokens, llm_tokens, key=len))*0.6, 8), sharex=False)
    fig.suptitle(f'{prefix[:-1].upper()} {crit_name} (Token ↔ Crit Align | Threshold={threshold:.4f})', 
                 fontsize=12, fontweight='bold', y=0.95)
    
    # Style values above and below the threshold differently.
    style_below = {"color": COLORS[0], "marker": "^", "s": 80, "alpha": 0.9, "bg_color": "#E6F3FF"}
    style_above = {"color": COLORS[2], "marker": "o", "s": 80, "alpha": 0.9, "bg_color": "#FFE6E6"}

    # Left panel. Human text tokens and their criterion values.
    ax1.set_title("Human Text | Token ↔ Corresponding Crit Value", fontsize=10, fontweight='bold', pad=15)
    x1 = np.arange(len(human_tokens))
    # Use NumPy indices compatible with one-dimensional arrays.
    human_below = np.where(human_crits <= threshold)[0]  # Select indices at or below the threshold.
    human_above = np.where(human_crits > threshold)[0]
    # Plot points at or below the threshold.
    ax1.scatter(x1[human_below], human_crits[human_below], **{k:v for k,v in style_below.items() if k!='bg_color'})
    # Plot points above the threshold.
    ax1.scatter(x1[human_above], human_crits[human_above], **{k:v for k,v in style_above.items() if k!='bg_color'})
    # Use background shading to distinguish regions.
    for i in x1:
        bg_color = style_below["bg_color"] if human_crits[i] <= threshold else style_above["bg_color"]
        ax1.axvspan(i-0.5, i+0.5, facecolor=bg_color, alpha=0.3)
    # Label each token with its criterion value.
    for i, c in enumerate(human_crits):
        ax1.annotate(f"{c:.3f}", (i, c), xytext=(0, 8), textcoords='offset points', 
                     ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(human_tokens, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel("Crit Value", fontsize=9)
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Threshold={threshold:.4f}')
    ax1.legend(frameon=False, loc='upper right', fontsize=8)
    ax1.grid(True, ls='--', alpha=0.3, axis='y')
    ax1.set_ylim(bottom=0)

    # Right panel. Machine text tokens and their criterion values.
    ax2.set_title("LLM Text | Token ↔ Corresponding Crit Value", fontsize=10, fontweight='bold', pad=15)
    x2 = np.arange(len(llm_tokens))
    # Use NumPy indices compatible with one-dimensional arrays.
    llm_below = np.where(llm_crits <= threshold)[0]
    llm_above = np.where(llm_crits > threshold)[0]
    # Plot points at or below the threshold.
    ax2.scatter(x2[llm_below], llm_crits[llm_below], **{k:v for k,v in style_below.items() if k!='bg_color'})
    # Plot points above the threshold.
    ax2.scatter(x2[llm_above], llm_crits[llm_above], **{k:v for k,v in style_above.items() if k!='bg_color'})
    # Use background shading to distinguish regions.
    for i in x2:
        bg_color = style_below["bg_color"] if llm_crits[i] <= threshold else style_above["bg_color"]
        ax2.axvspan(i-0.5, i+0.5, facecolor=bg_color, alpha=0.3)
    # Label each token with its criterion value.
    for i, c in enumerate(llm_crits):
        ax2.annotate(f"{c:.3f}", (i, c), xytext=(0, 8), textcoords='offset points', 
                     ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(llm_tokens, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel("Crit Value", fontsize=9)
    ax2.set_xlabel("Token Sequence (One Score per Token)", fontsize=9)
    ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Threshold={threshold:.4f}')
    ax2.legend(frameon=False, loc='upper right', fontsize=8)
    ax2.grid(True, ls='--', alpha=0.3, axis='y')
    ax2.set_ylim(bottom=0)

    # Save the figure.
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ {prefix} Token-Crit Align Threshold Plot saved to: {save_path}")
    return save_path