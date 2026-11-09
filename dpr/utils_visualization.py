import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
import warnings
# Suppress unrelated warnings.
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Global font, minus-sign, and layout settings.
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # Render minus signs using the selected fonts.
plt.rcParams['figure.autolayout'] = True    # Use automatic figure layout.
COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E", "#7209B7"] # Plot palette.

# Visualization helpers.
def min_max_normalize(arr):
    """Normalize criterion values to [0, 1] while preserving their relative order."""
    arr = np.array(arr, dtype=np.float32)
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max - arr_min < 1e-8: # Avoid division by zero.
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

def flatten_to_1d(arr):
    """Flatten nested lists and tensors into a one-dimensional numeric list."""
    if isinstance(arr, (list, np.ndarray)):
        return np.array([float(x) for x in np.array(arr).flatten()], dtype=np.float32)
    else:
        return np.array([float(arr)], dtype=np.float32)

# Render token-level threshold highlights.
def render_token_text(ax, tokens, crits, threshold, highlight_color="#FFFFE0"):
    """
    Render tokens with a background highlight when their score exceeds the threshold.
    """
    x, y = 0.03, 0.93  # Set left and top padding.
    max_width = 0.95   # Leave space at the right edge.
    char_width = 0.009  # Use a wider estimate to accommodate non-Latin and special characters.
    line_spacing = 0.15  # Space the lines to prevent overlap.

    # Filter invalid tokens and extra tokenizer markers.
    invalid_tokens = ['<pad>', '<bos>', '<eos>', '[CLS]', '[SEP]', '<unk>', '', '▁', '[PAD]', 'G']
    valid_pairs = []
    for t, c in zip(tokens, crits):
        t = str(t).replace("▁", " ").strip() # Replace the tokenizer's word boundary marker with a space.
        if t not in invalid_tokens and len(t) > 0:
            valid_pairs.append((t, float(c)))
    
    # Draw each token.
    for token, crit in valid_pairs:
        token_width = len(token) * char_width
        # Wrap the line when it exceeds the available width.
        if x + token_width > max_width:
            x = 0.03
            y -= line_spacing # Advance to the next line.
        
        # Check whether the token should be highlighted.
        bbox = None
        if crit > threshold:
            bbox = dict(boxstyle="round,pad=0.1", facecolor=highlight_color, edgecolor="none", alpha=0.8)
        
        # Use a smaller font to reduce token width.
        ax.text(x, y, token, transform=ax.transAxes, fontsize=9, ha="left", va="top", bbox=bbox)
        x += token_width # Advance the horizontal position.

# Generate a side-by-side comparison of human and model text.
def plot_token_crit_highlight(human_tokens, human_crits, llm_tokens, llm_crits, save_dir, img_prefix, epoch=0):
    """
    Show plain human text alongside machine text with token-level threshold highlights.
    """
    # Clean and normalize the data.
    human_crits = flatten_to_1d(human_crits)
    llm_crits = flatten_to_1d(llm_crits)
    human_crits_norm = min_max_normalize(human_crits)
    llm_crits_norm = min_max_normalize(llm_crits)
    all_crits = np.concatenate([human_crits_norm, llm_crits_norm])
    threshold = np.mean(all_crits)
    print(f"\n✅ Token-Crit Highlight Threshold (0~1 Norm): {threshold:.4f}")

    # Increase the figure height to fit more text lines.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharex=False, sharey=False)
    fig.suptitle(f"{img_prefix} | Human Text vs LLM Text (Token-Crit Threshold Highlight)", fontsize=14, fontweight='bold')

    # Left panel for human-written text.
    ax1.set_title(f" Human Text (Highlight > Threshold {threshold:.4f})", fontsize=12, fontweight='bold', pad=20, color="green")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    render_token_text(ax1, human_tokens, human_crits_norm, threshold, highlight_color=None)

    # Right panel for machine-generated text.
    ax2.set_title(f" LLM Text (Highlight > Threshold {threshold:.4f})", fontsize=12, fontweight='bold', pad=20, color="red")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    render_token_text(ax2, llm_tokens, llm_crits_norm, threshold, highlight_color="#FFFFE0")

    # Save the comparison figure.
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"{img_prefix}_epoch_{epoch:03d}_token_crit_highlight.png"
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Saved the token criterion comparison to {save_path}")
    return save_path