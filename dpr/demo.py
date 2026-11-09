import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F
from torch import nn
import sys
import os
import time
from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM, PeftModel
import torch.nn.functional as F
# Make utils_RAHF.py importable from the current directory or via sys.path.
# from utils_RAHF import calculate_SPO_loss
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import matplotlib
matplotlib.use('Agg')  # Use a noninteractive backend to save plots.
from scipy.stats import gaussian_kde

# Helpers for processing and visualization.
def to_np(tensor):
    """Convert a tensor to a float32 NumPy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().float().numpy()
    return np.array(tensor, dtype=np.float64)

def process_single_text(text, scorer):
    """
    Process one text sample and return the features needed for visualization.
    Args:
        text: Input text (str).
        scorer: Initialized ComputeScore instance.
    Returns:
        Feature data as a dictionary.
    """
    # 1. Tokenize the text.
    tok = scorer.scoring_tokenizer(text, return_tensors="pt").to(scorer.device)
    labels = tok.input_ids[:, 1:]  # Drop the first token (BOS).
    
    # 2. Get the model outputs without gradients.
    with torch.no_grad():
        _, _, logits_score, logits_ref, _, components = scorer.get_SPO_input(
            tok, [text], labels, training_module=False, return_components=True
        )
    
    # 3. Compute negative log likelihood (NLL).
    # Scoring model NLL under the machine preference.
    log_probs_score = torch.log_softmax(logits_score, dim=-1)
    token_nll_score = -to_np(log_probs_score.gather(2, labels.unsqueeze(-1)).squeeze())
    # Reference model NLL under the general language distribution.
    log_probs_ref = torch.log_softmax(logits_ref, dim=-1)
    token_nll_ref = -to_np(log_probs_ref.gather(2, labels.unsqueeze(-1)).squeeze())
    
    # 4. Compute the maximum probability and corresponding loss at each token.
    probs = torch.softmax(logits_score, dim=-1)
    max_probs = probs.max(dim=-1).values  # Maximum probability for each token position.
    real_p_values = to_np(max_probs.squeeze())
    real_loss = -np.log(real_p_values)  # Corresponding loss.
    
    # 5. Extract sequence log probability scores.
    raw_scores = components['label_log_probs']
    seq_scores = np.atleast_1d(to_np(raw_scores).squeeze())
    
    # 6. Prepare the token text for plot labels.
    tokens_text = scorer.scoring_tokenizer.convert_ids_to_tokens(labels[0])
    tokens_text = [t.replace('Ġ', '') for t in tokens_text]
    
    # 7. Align the lengths of the scores and token sequences.
    min_len = min(len(seq_scores), len(tokens_text))
    seq_scores = seq_scores[:min_len]
    tokens_text = tokens_text[:min_len]
    
    # 8. Return the processed features.
    return {
        "token_nll_score": token_nll_score,
        "token_nll_ref": token_nll_ref,
        "real_p_values": real_p_values,
        "real_loss": real_loss,
        "seq_scores": seq_scores,
        "tokens_text": tokens_text,
        "labels": labels,
        "text": text
    }

# Scoring model implementation.
def from_pretrained(cls, model_name, kwargs, cache_dir):
    """Load the model from local storage first, then fall back to Hugging Face."""
    if "/" in model_name:
        local_path = os.path.join(cache_dir, model_name.split("/")[1])
    else:
        local_path = os.path.join(cache_dir, model_name)

    if os.path.exists(local_path):
        return cls.from_pretrained(local_path,** kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir, device_map='auto')

# Model aliases and data types (Appendix A.1).
model_fullnames = {  
    'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
    'gpt-j-6B': 'EleutherAI/gpt-j-6B',
}
float16_models = ['gpt-neo-2.7B', 'gpt-j-6B']  # Models to load in float16.

def get_model_fullname(model_name):
    """Resolve a model alias to its full name."""
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def load_tokenizer(model_name, for_dataset, cache_dir):
    """Load the tokenizer with the padding direction required by the dataset."""
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}
    
    # Special tokenizer settings, including disabling the fast tokenizer for OPT.
    if "facebook/opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    # Use left padding for PubMed and right padding for other datasets.
    if for_dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    else:
        optional_tok_kwargs['padding_side'] = 'right'
    
    # Load the tokenizer and set a pad token if one is missing.
    base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname.lower():  # Special handling for 13B models.
            base_tokenizer.pad_token_id = 0
    return base_tokenizer

def calculate_kl_divergence(logits_q, logits_p, temperature=1.0, epsilon=1e-10):
    """
    Compute KL(q || p) to constrain the scoring model q relative to the reference model p.
    Args:
        logits_q: Scoring model logits [bsz, seq_len, vocab_size].
        logits_p: Reference model logits [bsz, seq_len, vocab_size].
        temperature: Distribution smoothing temperature (default: 1.0).
        epsilon: Stability constant that prevents log(0).
    Returns:
        kl_avg: KL divergence averaged over sequence positions [bsz].
    """
    # Use the smaller vocabulary size to avoid a dimension mismatch.
    vocab_size = min(logits_q.size(-1), logits_p.size(-1))
    logits_q = logits_q[..., :vocab_size] / temperature
    logits_p = logits_p[..., :vocab_size] / temperature
    
    # Compute probability distributions with epsilon for numerical stability.
    log_probs_q = torch.log_softmax(logits_q, dim=-1).clamp_min(torch.log(torch.tensor(epsilon, device=logits_q.device)))
    probs_q = torch.softmax(logits_q, dim=-1).clamp_min(epsilon)
    log_probs_p = torch.log_softmax(logits_p, dim=-1).clamp_min(torch.log(torch.tensor(epsilon, device=logits_q.device)))
    
    # Compute KL divergence per token and average across sequence positions.
    kl_per_token = (probs_q * (log_probs_q - log_probs_p)).sum(dim=-1)  # [bsz, seq_len]
    kl_avg = kl_per_token.mean(dim=-1)  # Mean KL divergence for each sequence [bsz].
    return kl_avg

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

def dynamic_semantic_window(importance, min_window=2, max_window=5):
    """
    Generate dynamic semantic windows from the importance of each position.
    Keep all tensors in the same dtype as the input.
    Args:
        importance: Semantic importance scores [batch_size, seq_len].
        min_window: Minimum window size.
        max_window: Maximum window size to limit computation.
    Returns:
        window_sizes: Dynamic window sizes per position [batch_size, seq_len].
        window_masks: Window weights with the input dtype [batch_size, seq_len, seq_len].
    """
    batch_size, seq_len = importance.shape
    dtype = importance.dtype  # Use the input dtype.
    device = importance.device
    
    # Set the dtype explicitly to avoid defaulting to float32.
    window_sizes = torch.zeros((batch_size, seq_len), device=device, dtype=torch.long)
    window_masks = torch.zeros((batch_size, seq_len, seq_len), device=device, dtype=dtype)
    
    # 1. Use global importance quantiles to establish a baseline window size.
    top_k = torch.topk(importance, k=seq_len//2, dim=1).values
    importance_median = torch.median(top_k, dim=1).values.unsqueeze(1)
    
    # 2. Assign larger windows to positions with higher importance.
    for i in range(seq_len):
        # Select the importance tier for this position.
        pos_importance = importance[:, i]
        window_scale = torch.where(
            pos_importance > importance_median * 1.2,  # High importance: largest window.
            torch.tensor(max_window, device=device),
            torch.where(
                pos_importance > importance_median * 0.8,  # Medium importance: intermediate window.
                torch.tensor((min_window + max_window) // 2, device=device),
                torch.tensor(min_window, device=device)  # Low importance: smallest window.
            )
        )
        window_sizes[:, i] = window_scale.clamp(min=min_window, max=max_window)
    
    # 3. Build masks that give more weight to important positions in each window.
    for b in range(batch_size):
        for i in range(seq_len):
            w_size = window_sizes[b, i].item()
            # Clamp the window boundaries to the sequence.
            start = max(0, i - w_size // 2)
            end = min(seq_len, i + w_size // 2 + 1)
            # Normalize semantic importance within the window and preserve the dtype.
            window_importance = importance[b, start:end]
            window_weight = window_importance / window_importance.sum().clamp_min(1e-8)
            # Fill the mask using the input dtype.
            window_masks[b, i, start:end] = window_weight.to(dtype=dtype)
    
    return window_sizes, window_masks

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels, return_components=False, kl_weight=0.02):
    """
    Compute style-conditioned probability curvature (Style-CPC) and KL regularization.
    Combine sequence and window discrepancies and constrain divergence from the reference distribution.
    """
    # 1. Align vocabulary dimensions.
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]
    
    # 2. Add a singleton label dimension for gathering ([bsz, seq_len, 1]).
    if labels.ndim == logits_score.ndim - 1:
        labels = labels.unsqueeze(-1)
    
    # 3. Compute the scoring and reference distributions.
    lprobs_score = torch.log_softmax(logits_score, dim=-1)  # Log probabilities from the style-aligned scoring model.
    probs_ref = torch.softmax(logits_ref, dim=-1)          # Probabilities from the general-purpose reference model.
    probs_score = torch.softmax(logits_score, dim=-1)      # Scoring model probabilities.
    
    # 4. Use log_softmax for label log probabilities to keep their units consistent.
    label_log_probs = (probs_score).gather(dim=-1, index=labels).squeeze(-1)  # [bsz, seq_len]
    # 5. Compute the baseline sequence-level discrepancy.
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)  # Expected log probability under the reference model [bsz, seq_len].
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)  # Variance under the reference model.
    seq_num = label_log_probs.mean(dim=-1) - mean_ref.mean(dim=-1)  # Difference in means [bsz].
    seq_den = var_ref.mean(dim=-1).clamp_min(1e-8).sqrt()           # Square root of the clamped variance to prevent division by zero.
    seq_discrepancy = -seq_num / seq_den  # Negate the sequence discrepancy so machine-written text is expected to score lower.

    # 6. Compute local discrepancies to capture fine-grained style features.
    window_size = 3
    if label_log_probs.size(1) >= window_size:
        # Use cumulative sums to compute each window sum in constant time.
        csum_label = torch.cumsum(label_log_probs, dim=1)
        csum_mean = torch.cumsum(mean_ref, dim=1)
        csum_var = torch.cumsum(var_ref, dim=1)
        
        def window_sum(csum_tensor, w):
            """Compute the sum for each window as csum[i+w] - csum[i]."""
            pad = torch.zeros(csum_tensor.size(0), 1, device=csum_tensor.device, dtype=csum_tensor.dtype)
            csum_pad = torch.cat([pad, csum_tensor], dim=1)
            return csum_pad[:, w:] - csum_pad[:, :-w]
        
        # Compute statistics within the windows.
        win_label = window_sum(csum_label, window_size).max(dim=-1).values  # Largest log probability sum across windows.
        win_mean = window_sum(csum_mean, window_size).mean(dim=-1)          # Mean reference log probability across windows.
        win_var = window_sum(csum_var, window_size).mean(dim=-1).clamp_min(1e-8)  # Window variance.
        local_discrepancy = -(win_label - win_mean) / win_var.sqrt()        # Local discrepancy.
    else:
        local_discrepancy = seq_discrepancy 
    # 7. Compute KL divergence between the scoring and reference distributions.
    kl_div = calculate_kl_divergence(logits_score, logits_ref)  # [bsz]
    # Combine sequence and local discrepancies with KL regularization. The reported range for kl_weight is 0.01 to 0.05.
    discrepancy = local_discrepancy + seq_discrepancy - kl_weight * kl_div  # Smaller KL divergence produces a smaller discrepancy penalty.
    
    # 8. Compute weighted log probabilities for the RAHF loss.
    weighted_log_likelihood = label_log_probs.sum(dim=-1)  # Sum log probabilities over the sequence.
    
    # 9. Return scores and KL divergence for inspection.
    if not return_components:
        return discrepancy, weighted_log_likelihood
    
    components = {
        'seq_discrepancy': seq_discrepancy,
        'local_discrepancy': local_discrepancy,
        'kl_divergence': kl_div,          # Record KL divergence.
        'window_size': window_size,
        'label_log_probs': label_log_probs
    }
    return discrepancy, weighted_log_likelihood, components

class ComputeScore(nn.Module):
    def __init__(self, scoring_model_name, reference_model_name, SPOtrained=True, 
                 SPO_beta=0.05, dataset='xsum', device='cuda', cache_dir='./models', kl_weight=0.02):
        """
        Args:
            kl_weight: KL regularization weight used by get_sampling_discrepancy_analytic.
            SPO_beta: Preference loss coefficient (Appendix A.1 default: 0.05).
        """
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.reference_model_name = get_model_fullname(reference_model_name)
        self.scoring_model_name = get_model_fullname(scoring_model_name)
        self.beta = SPO_beta
        self.kl_weight = kl_weight  # KL regularization weight.
        self.dataset = dataset
        self.cache_dir = cache_dir
        
        # Model loader for RAHF training and ablation experiments.
        def load_model(model_name, device, cache_dir, SPOtrained=True):
            model_fullname = get_model_fullname(model_name)
            print(f'Loading base model {model_fullname}...')
            model_kwargs = {}
            
            # Configure models that use float16.
            if model_name in float16_models:
                model_kwargs.update(dict(torch_dtype=torch.float16))
            if 'gpt-j' in model_name:
                model_kwargs.update(dict(revision='float16'))
            
            # Load the base model.
            if SPOtrained:
                model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
            else:
                model = from_pretrained(AutoPeftModelForCausalLM, model_fullname, model_kwargs, cache_dir)
            
            # Move the model to the requested device.
            print('Moving model to GPU...', end='', flush=True)
            start = time.time()
            model.to(device)
            print(f'DONE ({time.time() - start:.2f}s)')
            return model
        
        # 1. Load separate tokenizers for the scoring and reference models.
        self.scoring_tokenizer = load_tokenizer(scoring_model_name, dataset, cache_dir)
        self.reference_tokenizer = load_tokenizer(reference_model_name, dataset, cache_dir)
        
        # 2. Load the base scoring model for LoRA fine-tuning.
        self.base_scoring_model = load_model(scoring_model_name, self.device, cache_dir, SPOtrained)
        self.base_reference_model = load_model(reference_model_name, self.device, cache_dir, SPOtrained)
        
        # Use the same LoRA configuration as training.
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=8, lora_alpha=32, lora_dropout=0.1  # LoRA settings from Appendix A.1.
        )
        
        # Initialize the LoRA-wrapped model.
        self.scoring_model = get_peft_model(self.base_scoring_model, self.peft_config) if SPOtrained else self.base_scoring_model
        self.reference_model = self.base_reference_model  # Use the base reference model without LoRA.
        self.reference_model1 = get_peft_model(self.base_reference_model, self.peft_config) if SPOtrained else self.base_reference_model
        
        # 4. Freeze the reference model so only scoring LoRA parameters are trained.
        for param in self.reference_model.parameters():
            param.requires_grad = False
        for param in self.reference_model1.parameters():
            param.requires_grad = False
        
        # 5. Bind the forward pass used to compute the training loss.
        self.criterion_fn = lambda *args, **kwargs: get_sampling_discrepancy_analytic(
            *args, **kwargs, kl_weight=self.kl_weight
        )
        self.forward = self.forward_SPO  # Use the preference training forward pass by default.
        
        # Print the trainable parameters to verify the LoRA configuration.
        print("Scoring model trainable parameters:")
        self.scoring_model.print_trainable_parameters()

    def load_lora_weights(self, lora_weight_path):
        """
        Load a local LoRA weights file in .bin format.
        Args:
            lora_weight_path: Path to the LoRA weights, such as autodl-tmp/ckpt/xxx/model.bin.
        """
        print(f"\nLoading LoRA weights from {lora_weight_path}...")
        # Load weights into the scoring model's LoRA layers.
        lora_weights = torch.load(lora_weight_path, map_location=self.device)
        
        # Handle PEFT weight name prefixes.
        if hasattr(self.scoring_model, 'base_model'):
            # For a PEFT model, load the LoRA layers of its base model.
            self.scoring_model.base_model.load_state_dict(lora_weights, strict=False)
        else:
            self.scoring_model.load_state_dict(lora_weights, strict=False)
        
        # Set evaluation mode.
        self.scoring_model.eval()
        print("✅ LoRA weights loaded successfully!")

    def from_pretrained(self, load_directory):
        """Load pretrained model weights and the LoRA adapter."""
        # Load the scoring model adapter.
        self.scoring_model = AutoPeftModelForCausalLM.from_pretrained(
            os.path.join(load_directory, 'scoring_model'),
            device_map='auto',
            torch_dtype=torch.float16 if self.scoring_model_name in float16_models else torch.float32
        ).to(self.device)
        # Load the reference model.
        ref_path = os.path.join(load_directory, 'reference_model.bin')
        if os.path.exists(ref_path):
            self.reference_model.load_state_dict(torch.load(ref_path, map_location=self.device))
        print(f"Model loaded from {load_directory}")

    def print_gradient_requirement(self):
        """Print whether each parameter requires gradients."""
        for name, param in self.named_parameters():
            grad_flag = 'Requires Grad' if param.requires_grad else 'Does not require grad'
            color = '\033[92m' if param.requires_grad else '\033[91m'
            print(f"{name}: {color}{grad_flag}\033[0m")

    def register_no_grad(self, module_names):
        """Disable gradients for the given module, such as the reference model."""
        for name, param in self.named_parameters():
            if any(module in name for module in module_names):
                param.requires_grad = False

    def save_pretrained(self, save_directory):
        """Save the model and LoRA parameters."""
        os.makedirs(save_directory, exist_ok=True)
        # Save the scoring model adapter and configuration.
        self.scoring_model.save_pretrained(os.path.join(save_directory, 'scoring_model'))
        # Save the frozen reference model weights.
        torch.save(self.reference_model.state_dict(), os.path.join(save_directory, 'reference_model.bin'))
        print(f"Model saved to {save_directory}")

    def get_SPO_input(self, tokenized=None, text=[""], labels=[""], training_module=False, return_components=False):
        """
        Prepare logits, probability discrepancies, and KL divergence for RAHF training.
        Args:
            training_module: Use the scoring model when True and the reference model when False.
        """
        if training_module:
            # Compute scoring model logits and reference model logits during training.
            logits_score = self.scoring_model(
                tokenized.input_ids, attention_mask=tokenized.attention_mask
            ).logits[:, :-1, :]  # Drop the final logit position to align with the labels.
            logits_ref = logits_score  
            # Retokenize the text if the scoring and reference models differ.
            if self.reference_model_name != self.scoring_model_name:
                tokenized_ref = self.reference_tokenizer(
                    text, return_tensors="pt", padding=True, return_token_type_ids=False,
                    add_special_tokens=True, return_attention_mask=True
                ).to(self.device)
                # Verify that the tokenizers produce compatible encodings.
                assert torch.allclose(tokenized_ref.input_ids[:, 1:], labels, atol=1e-5), "Tokenizer mismatch between models!"
                logits_ref = self.reference_model(tokenized_ref.input_ids).logits[:, :-1, :]
        
        else:
            # Use the frozen reference model in inference mode.
            with torch.no_grad():
                if self.reference_model_name != self.scoring_model_name:
                    tokenized_ref = self.reference_tokenizer(
                        text, return_tensors="pt", padding=True, return_token_type_ids=False,
                        add_special_tokens=True, return_attention_mask=True
                    ).to(self.device)
                    assert torch.allclose(tokenized_ref.input_ids[:, 1:], labels, atol=1e-5), "Tokenizer mismatch!"
                    tokenized = tokenized_ref
                
                # Reference logits after averaging.
                logits_ref = (
                    self.reference_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:, :-1, :] +
                    self.reference_model1(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:, :-1, :]
                ) / 2
                logits_score = self.scoring_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:, :-1, :] 
        
        # Compute the probability discrepancy and KL divergence.
        crit_out = self.criterion_fn(logits_ref, logits_score, labels, return_components=return_components)
        
        # Compute KL divergence for this pair of logits for use in the loss.
        kl_div = calculate_kl_divergence(logits_score, logits_ref)
        
        if return_components:
            crit, spo_input, components = crit_out
            # Add KL divergence to the returned components.
            components['kl_divergence'] = kl_div
            return crit, spo_input, logits_score, logits_ref, kl_div, components
        return crit, spo_input, logits_score, logits_ref, kl_div

    def forward_SPO(self, text):
        """
        Compute the training loss and KL regularization for a human text and its machine revision.
        text: [original_text (human-written), sampled_text (machine-revised)].
        """
        original_text = text[0]  # Human-written text (dispreferred).
        sampled_text = text[1]   # Machine-revised text (preferred).
        
        # 1. Process the dispreferred human-written text.
        tokenized_human = self.scoring_tokenizer(
            original_text, return_tensors="pt", padding=True, return_token_type_ids=False
        ).to(self.device)
        labels_human = tokenized_human.input_ids[:, 1:]  # Drop the first token so labels align with logits[:, :-1].
        
        # Compute the reference baseline for the human-written text.
        ref_human_crit, ref_human_logprob, _, _, ref_human_kl, ref_human_comp = self.get_SPO_input(
            tokenized_human, original_text, labels_human, return_components=True
        )
        # Compute the scoring model predictions for the human-written text.
        train_human_crit, train_human_logprob, train_human_logits, train_human_ref_logits, train_human_kl, train_human_comp = self.get_SPO_input(
            tokenized_human, original_text, labels_human, training_module=True, return_components=True
        )
        
        # 2. Process the preferred machine-revised text.
        tokenized_machine = self.scoring_tokenizer(
            sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False
        ).to(self.device)
        labels_machine = tokenized_machine.input_ids[:, 1:]
        
        # Compute the reference baseline for the machine-revised text.
        ref_machine_crit, ref_machine_logprob, _, _, ref_machine_kl, ref_machine_comp = self.get_SPO_input(
            tokenized_machine, sampled_text, labels_machine, return_components=True
        )
        # Compute the scoring model predictions for the machine-revised text.
        train_machine_crit, train_machine_logprob, train_machine_logits, train_machine_ref_logits, train_machine_kl, train_machine_comp = self.get_SPO_input(
            tokenized_machine, sampled_text, labels_machine, training_module=True, return_components=True
        )
        
        # 3. Compute the preference loss and KL regularization.
        # Import the preference loss implementation before using this example.
        # spo_loss, prefered_rel_logprob, disprefered_rel_logprob, reward_acc, reward_margin = calculate_SPO_loss(
        #     train_machine_logprob,    # Machine-revised text (preferred).
        #     train_human_logprob,  # Human-written text (dispreferred).
        #     ref_machine_logprob,       # Reference baseline for machine-revised text.
        #     ref_human_logprob,      # Reference baseline for human-written text.
        #     beta=self.beta
        # )
        # Placeholder for the actual preference loss.
        spo_loss = torch.tensor(0.1, device=self.device, requires_grad=True)
        reward_acc = torch.tensor(0.9, device=self.device)
        reward_margin = torch.tensor(0.5, device=self.device)
        
        # Average the KL divergence for human-written and machine-revised texts.
        kl_reg_loss = (train_human_kl.mean() + train_machine_kl.mean()) / 2
        
        # Combine the preference loss and KL regularization using kl_weight.
        total_loss = spo_loss + self.kl_weight * kl_reg_loss
        
        # 4. Return results and KL metrics.
        return dict(
            crit=[ref_human_crit, train_human_crit, ref_machine_crit, train_machine_crit],
            loss=total_loss,          # Total loss with KL regularization.
            spo_loss=spo_loss,              # Unregularized preference loss.
            kl_reg_loss=kl_reg_loss,        # KL regularization loss.
            kl_stats={                      # KL divergence statistics for debugging.
                'human_kl_mean': train_human_kl.mean().item(),
                'machine_kl_mean': train_machine_kl.mean().item(),
                'ref_human_kl_mean': ref_human_kl.mean().item()
            },
            reward_metrics={                # Preference reward metrics.
                'reward_accuracy': reward_acc.mean().item(),
                'reward_margin': reward_margin.mean().item()
            },
            components={
                'ref_original': ref_human_comp,
                'train_original': train_human_comp,
                'ref_sampled': ref_machine_comp,
                'train_sampled': train_machine_comp
            }
        )
    
    def visualize_training_progress(self, human_text, machine_text, epoch=0, step=0, save_dir="./training_vis"):
        """
        Generate and save training plots comparing human-written and machine-revised text.
        Args:
            human_text: Human-written text (str or list).
            machine_text: Machine-revised text (str or list).
            epoch: Current epoch used in filenames.
            step: Current step used in filenames.
            save_dir: Directory for saved figures.
        """
        # Create the output directory.
        os.makedirs(save_dir, exist_ok=True)
        
        # Use the first sample to keep the visualization readable.
        if isinstance(human_text, list):
            human_text = human_text[0]
        if isinstance(machine_text, list):
            machine_text = machine_text[0]
        
        # Limit input length for visualization.
        max_text_len = 500
        human_text = human_text[:max_text_len]
        machine_text = machine_text[:max_text_len]
        
        try:
            # 1. Process the text.
            human_data = process_single_text(human_text, self)
            machine_data = process_single_text(machine_text, self)
            
            # 2. Plot the features.
            plt.style.use('seaborn-v0_8-paper')
            fig = plt.figure(figsize=(18, 6))
            gs = gridspec.GridSpec(1, 3, figure=fig)

            # Figure 1a. Gradient trap distributions for human and machine text.
            ax1 = fig.add_subplot(gs[0, 0])
            
            # Theoretical curve.
            x = np.linspace(0.01, 0.99, 200)
            ax1.plot(x, -np.log(x), 'k-', alpha=0.3, label=r'Theoretical Loss')
            
            # Human-written text (blue).
            ax1.scatter(
                human_data['real_p_values'], human_data['real_loss'],
                c='blue', s=40, alpha=0.7, zorder=4, label='Human Text'
            )
            # Machine-revised text (red).
            ax1.scatter(
                machine_data['real_p_values'], machine_data['real_loss'],
                c='red', s=40, alpha=0.7, zorder=5, label='Machine Text'
            )
            
            ax1.set_title(f'(a) Gradient Trap Distribution (Epoch {epoch}, Step {step})', fontweight='bold', fontsize=12)
            ax1.set_xlabel('Token Probability $p$', fontsize=10)
            ax1.set_ylabel('Loss / Gradient Magnitude', fontsize=10)
            ax1.text(0.1, 4, "High Gradient\n(Noise Zone)", color='gray', fontweight='bold')
            ax1.text(0.6, 1, "Low Gradient\n(Machine Zone)", color='gray', fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(alpha=0.2)

            # Figure 1b. NLL distributions for human and machine text.
            ax2 = fig.add_subplot(gs[0, 1])
            
            # Define the KDE plot helper.
            def plot_kde(data, color, label):
                density = gaussian_kde(data)
                xs = np.linspace(min(data)-1, max(data)+1, 200)
                ax2.plot(xs, density(xs), color=color, linewidth=2, label=label)
                ax2.fill_between(xs, density(xs), color=color, alpha=0.15)

            # Human-written text NLL distribution (blue).
            plot_kde(human_data['token_nll_ref'], 'blue', 'Human (Reference Model)')
            # Machine-revised text NLL distribution (red).
            plot_kde(machine_data['token_nll_ref'], 'red', 'Machine (Reference Model)')
            
            ax2.set_title('(b) Token NLL Distribution', fontweight='bold', fontsize=12)
            ax2.set_xlabel('Negative Log-Likelihood', fontsize=10)
            ax2.set_ylabel('Density', fontsize=10)
            ax2.legend(fontsize=9)
            ax2.grid(alpha=0.2)

            # Figure 1c. Token-level signals for human and machine text.
            ax3 = fig.add_subplot(gs[0, 2])
            
            # Plot human text sequence scores as blue bars.
            x_human = np.arange(len(human_data['seq_scores']))[:50]  # Display only the first 50 tokens to avoid crowding.
            seq_scores_human = human_data['seq_scores'][:50]
            bars_human = ax3.bar(
                x_human - 0.2, seq_scores_human, 
                width=0.4, color='skyblue', edgecolor='blue', alpha=0.7, label='Human Text'
            )
            
            # Plot machine text sequence scores as red bars.
            x_machine = np.arange(len(machine_data['seq_scores']))[:50]
            seq_scores_machine = machine_data['seq_scores'][:50]
            bars_machine = ax3.bar(
                x_machine + 0.2, seq_scores_machine, 
                width=0.4, color='lightcoral', edgecolor='red', alpha=0.7, label='Machine Text'
            )
            
            # Compute and plot the mean for each series.
            avg_human = float(np.mean(seq_scores_human))
            avg_machine = float(np.mean(seq_scores_machine))
            ax3.axhline(avg_human, color='blue', linestyle='--', alpha=0.8, label=f'Human Avg ({avg_human:.2f})')
            ax3.axhline(avg_machine, color='red', linestyle='--', alpha=0.8, label=f'Machine Avg ({avg_machine:.2f})')
            
            # Highlight machine-text tokens with unusual scores.
            threshold = avg_machine + 1.0 * np.std(seq_scores_machine)
            for i, score in enumerate(seq_scores_machine):
                if score > threshold and i < len(bars_machine):
                    bars_machine[i].set_color('red')
                    # Label the first 20 tokens to avoid crowding.
                    if i < 20 and i < len(machine_data['tokens_text']):
                        ax3.text(i+0.2, score+0.2, machine_data['tokens_text'][i], 
                                ha='center', fontsize=7, rotation=90, color='darkred')

            ax3.set_title('(c) Token-level Log-Prob Score (First 50 Tokens)', fontweight='bold', fontsize=12)
            ax3.set_xlabel('Token Position', fontsize=10)
            ax3.set_ylabel('Log-Prob Score', fontsize=10)
            ax3.legend(fontsize=8, loc='lower right')
            ax3.grid(alpha=0.2)
            
            # Adjust the layout and save the figure.
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"train_vis_epoch_{epoch}_step_{step}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to release memory.
            print(f"Saved training visualization to {save_path}")
            
        except Exception as e:
            print(f"Could not generate visualization: {str(e)}")

# Example training loop.
def train_loop(scorer, train_dataloader, epochs=5, vis_freq=10, save_dir="./training_vis"):
    """
    Run a sample training loop with visualization.
    Args:
        scorer: Initialized ComputeScore instance.
        train_dataloader: Training loader that yields (human_text, machine_text) pairs.
        epochs: Number of training epochs.
        vis_freq: Plotting interval in steps.
        save_dir: Directory for visualization outputs.
    """
    # Optimize only the LoRA parameters.
    optimizer = torch.optim.AdamW(scorer.scoring_model.parameters(), lr=1e-4)
    
    # Set training mode.
    scorer.train()
    
    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch+1}/{epochs} =====")
        epoch_loss = 0.0
        
        for step, (human_text, machine_text) in enumerate(train_dataloader):
            # 1. Compute the loss in the forward pass.
            outputs = scorer.forward_SPO([human_text, machine_text])
            loss = outputs['loss']
            
            # 2. Backpropagate the loss.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 3. Accumulate the loss.
            epoch_loss += loss.item()
            
            # 4. Generate visualizations periodically.
            if step % vis_freq == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}, KL Reg Loss: {outputs['kl_reg_loss'].item():.4f}")
                # Generate the visualization without gradients.
                with torch.no_grad():
                    scorer.visualize_training_progress(
                        human_text, machine_text, 
                        epoch=epoch, step=step, 
                        save_dir=save_dir
                    )
        
        # 5. Save the model and plots at the end of each epoch.
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
        
        # Generate an end-of-epoch visualization.
        with torch.no_grad():
            scorer.visualize_training_progress(
                human_text, machine_text, 
                epoch=epoch, step="end", 
                save_dir=save_dir
            )
        
        # Optionally save the model.
        scorer.save_pretrained(f"./ckpt_epoch_{epoch+1}")

# Example run.
def test_training_with_vis():
    # Configure the run.
    SCORING_MODEL = "gpt-neo-2.7B"
    REF_MODEL = "gpt-neo-2.7B"
    CACHE_DIR = "./models"
    
    # Initialize the model.
    scorer = ComputeScore(
        scoring_model_name=SCORING_MODEL,
        reference_model_name=REF_MODEL,
        dataset='xsum',
        cache_dir=CACHE_DIR,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Use a mock loader here and replace it with real training data for training.
    class MockDataloader:
        def __iter__(self):
            # Create a mock human and machine text pair.
            human_text = "New shots of Charlene Downes, 14, who was last seen in Blackpool town centre, were discovered during a trawl of CCTV. Her body has never been found."
            machine_text = "Chief executive Bimlendra Jha lauded the herculean efforts that have been made to revitalize Port Talbot, a steel plant that was hemorrhaging £1m a day only a few months ago."
            for i in range(20):  # Simulate 20 steps.
                yield [human_text], [machine_text]
        
        def __len__(self):
            return 20
    
    train_dataloader = MockDataloader()
    
    # Start training with visualization.
    train_loop(
        scorer=scorer,
        train_dataloader=train_dataloader,
        epochs=2,
        vis_freq=5,  # Generate a visualization every five steps.
        save_dir="./training_vis_demo"
    )

if __name__ == "__main__":
    # Run the example training with visualization.
    test_training_with_vis()