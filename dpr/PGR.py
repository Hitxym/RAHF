import torch
from torch import nn
import sys
import os
import time
from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM
import torch.nn.functional as F
from utils_RAHF import calculate_SPO_loss
from transformers import AutoModelForCausalLM, AutoTokenizer


def from_pretrained(cls, model_name, kwargs, cache_dir):
    """Load the model from local storage first, then fall back to Hugging Face."""
    if "/" in model_name:
        local_path = os.path.join(cache_dir, model_name.split("/")[1])
    else:
        local_path = os.path.join(cache_dir, model_name)

    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
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
    Args:
        log_probs: Label log probabilities [batch_size, seq_len].
        mean_ref: Reference distribution means [batch_size, seq_len].
        var_ref: Reference distribution variances [batch_size, seq_len].
    Returns:
        importance: Semantic importance scores between 0 and 1 [batch_size, seq_len].
    """
    # 1. Measure deviation from the reference distribution.
    prob_deviation = torch.abs(log_probs - mean_ref) / torch.sqrt(var_ref.clamp_min(1e-8))
    # 2. Use lower entropy as a signal of greater semantic importance.
    probs = torch.exp(log_probs.clamp(max=0))  # Prevent overflow in exp.
    entropy = -torch.sum(probs * log_probs.clamp(min=-100), dim=-1, keepdim=True)  # Clamp log probabilities for numerical stability.
    # 3. Normalize to the range [0, 1].
    prob_deviation = F.normalize(prob_deviation, p=1, dim=1)
    entropy = F.normalize(entropy, p=1, dim=1)
    # 4. Combine higher deviation and lower entropy into a higher importance score.
    importance = (1 - entropy) * prob_deviation
    importance = F.normalize(importance, p=1, dim=1)  # Apply final normalization.
    return importance

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
    # label_log_probs = (probs_score).gather(dim=-1, index=labels).squeeze(-1)  # [bsz, seq_len]
    # 5. Compute the baseline sequence-level discrepancy.
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)  # Expected log probability under the reference model [bsz, seq_len].
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)  # Variance under the reference model.
    seq_num = label_log_probs.sum(dim=-1) - mean_ref.sum(dim=-1)  # [bsz]
    seq_den = var_ref.sum(dim=-1).sqrt() 
    # seq_num = label_log_probs.mean(dim=-1) - mean_ref.mean(dim=-1)  # Difference in means [bsz].
    # seq_den = var_ref.mean(dim=-1).clamp_min(1e-8).sqrt()           # Square root of the clamped variance to prevent division by zero.
    seq_discrepancy = -seq_num / seq_den  # Negate the sequence discrepancy so machine-written text is expected to score lower.
    
    # 6. Compute local discrepancies to capture fine-grained style features.
    window_size = 3  # Baseline window size. Semantic importance determines the actual size.
    if label_log_probs.size(1) >= window_size:
        # 1. Compute semantic importance in the input dtype.
        semantic_importance = semantic_importance_scoring(label_log_probs, mean_ref, var_ref)
        
        # 2. Generate dynamic semantic windows in the input dtype.
        dynamic_win_sizes, dynamic_win_masks = dynamic_semantic_window(
            semantic_importance, 
            min_window=2, 
            max_window=5
        )
        
        # 3. Compute weighted statistics with matching tensor dtypes.
        # Expand dimensions for matrix multiplication and align dtypes.
        label_expanded = label_log_probs.unsqueeze(1).to(dtype=label_log_probs.dtype)  # [B, 1, L]
        mean_expanded = mean_ref.unsqueeze(1).to(dtype=mean_ref.dtype)                # [B, 1, L]
        var_expanded = var_ref.unsqueeze(1).to(dtype=var_ref.dtype)                  # [B, 1, L]
        
        # Match the mask dtype to the input tensor dtype.
        dynamic_win_masks = dynamic_win_masks.to(dtype=label_expanded.dtype)
        
        # Use semantic weights when summing within each window.
        win_label = torch.bmm(label_expanded, dynamic_win_masks).squeeze(1)  # [B, L]
        win_mean = torch.bmm(mean_expanded, dynamic_win_masks).squeeze(1)    # [B, L]
        win_var = torch.bmm(var_expanded, dynamic_win_masks).squeeze(1)      # [B, L]
        # Take the largest weighted log probability sum across windows.
        win_label = win_label.max(dim=-1).values
        # Compute the weighted mean within the windows.
        win_mean = win_mean.mean(dim=-1)
        win_var = win_var.mean(dim=-1).clamp_min(1e-8)
        
        # Compute the local discrepancy.
        local_discrepancy = (win_label - win_mean) / win_var.sqrt()
    else:
        local_discrepancy = seq_discrepancy  # Reuse the sequence discrepancy for short texts.
    
    # 7. Compute KL divergence between the scoring and reference distributions.
    kl_div = calculate_kl_divergence(logits_score, logits_ref)  # [bsz]
    # Combine sequence and local discrepancies with KL regularization. The reported range for kl_weight is 0.01 to 0.05.
    #discrepancy = local_discrepancy + seq_discrepancy - kl_weight * kl_div  # Smaller KL divergence produces a smaller discrepancy penalty.
    discrepancy = seq_discrepancy
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
        'label_log_probs': label_log_probs.mean(dim=-1)
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
            print(f'Loading model {model_fullname}...')
            model_kwargs = {}
            
            # Configure models that use float16.
            if model_name in float16_models:
                model_kwargs.update(dict(torch_dtype=torch.float16))
            if 'gpt-j' in model_name:
                model_kwargs.update(dict(revision='float16'))
            
            # Load the base model for LoRA training or an existing PEFT model for ablation.
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
        
        # 2. Load the scoring model and fine-tune only the LoRA parameters.
        scoring_model = load_model(scoring_model_name, self.device, cache_dir, SPOtrained)
        reference_model = load_model(reference_model_name, self.device, cache_dir, SPOtrained)
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=8, lora_alpha=32, lora_dropout=0.1  # LoRA settings from Appendix A.1.
        )
        self.scoring_model = get_peft_model(scoring_model, self.peft_config) if SPOtrained else scoring_model
        self.reference_model = get_peft_model(reference_model, self.peft_config) if SPOtrained else self.reference_model
        # 3. Load the frozen reference model.
        # self.reference_model = load_model(reference_model_name, self.device, cache_dir, SPOtrained)
        # # Optional LoRA reference model for averaging reference logits.
        # self.reference_model1 = get_peft_model(self.reference_model, self.peft_config) if SPOtrained else self.reference_model
        
        # 4. Freeze the reference model so only scoring LoRA parameters are trained.
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        # 5. Bind the forward pass used to compute the training loss.
        self.criterion_fn = lambda *args, **kwargs: get_sampling_discrepancy_analytic(
            *args, **kwargs, kl_weight=self.kl_weight
        )
        self.forward = self.forward_SPO  # Use the preference training forward pass by default.
        
        # Print the trainable parameters to verify the LoRA configuration.
        print("Scoring model trainable parameters:")
        self.scoring_model.print_trainable_parameters()

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
            
            # Reference logits can be averaged across the base and LoRA reference models.
            # logits_ref = (
            #     self.reference_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:, :-1, :] +
            #     self.reference_model1(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:, :-1, :]
            # ) / 2
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
                logits_score = self.reference_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:] # shape: [bsz, sentence_len, dim]
                logits_ref = logits_score
                
                # logits_ref = (
                #     self.reference_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:, :-1, :] +
                #     self.reference_model1(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:, :-1, :]
                # ) / 2
                # logits_score = logits_ref  # Use reference logits as scoring logits in inference mode.
        
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
        # Compute the preference loss using utils_RAHF (Section 2.3).
        spo_loss, prefered_rel_logprob, disprefered_rel_logprob, reward_acc, reward_margin = calculate_SPO_loss(
            train_machine_logprob,    # Machine-revised text (preferred).
            train_human_logprob,  # Human-written text (dispreferred).
            ref_machine_logprob,       # Reference baseline for machine-revised text.
            ref_human_logprob,      # Reference baseline for human-written text.
            beta=self.beta
        )
        
        # Average the KL divergence for human-written and machine-revised texts.
        kl_reg_loss = (train_human_kl.mean() + train_machine_kl.mean()) / 2
        
        # Combine the preference loss and KL regularization using kl_weight.
        # total_loss = spo_loss + self.kl_weight * kl_reg_loss
        total_loss = spo_loss
        
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