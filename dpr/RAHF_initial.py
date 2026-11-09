import torch
from torch import nn
import sys

from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM
import os
import torch.nn.functional as F
from utils_RAHF import calculate_SPO_loss
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    if "/" in model_name:
        local_path = os.path.join(cache_dir, model_name.split("/")[1])
    else:
        local_path = os.path.join(cache_dir, model_name)

    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir, device_map='auto')

model_fullnames = {  
                    'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
                    'gpt-j-6B': 'EleutherAI/gpt-j-6B',
                    }
float16_models = ['gpt-neo-2.7B','gpt-j-6B']

def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def load_tokenizer(model_name, for_dataset, cache_dir):
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}
    if "facebook/opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if for_dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    else:
        optional_tok_kwargs['padding_side'] = 'right'
    base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    return base_tokenizer


# def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):

#     if logits_ref.size(-1) != logits_score.size(-1):
#         vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
#         logits_ref = logits_ref[:, :, :vocab_size]
#         logits_score = logits_score[:, :, :vocab_size]

#     labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
#     lprobs_score = torch.log_softmax(logits_score, dim=-1)
#     probs_ref = torch.softmax(logits_ref, dim=-1)
    
#     log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
#     mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
#     var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
#     discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    
#     return discrepancy, log_likelihood.sum(dim=-1)

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels, return_components=False):

    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    probs_score = torch.softmax(logits_score, dim=-1)
    
    # original per-token log-likelihood for labels
    label_log_probs = (probs_score).gather(dim=-1, index=labels).squeeze(-1)
    # label_log_probs = (lprobs_score).gather(dim=-1, index=labels).squeeze(-1)
    # discrepancy (sequence-level, kept as baseline)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)  # shape: [bsz, seq_len]
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)  # [bsz, seq_len]
    
    # seq_num = label_log_probs.sum(dim=-1) - mean_ref.sum(dim=-1)  # [bsz]
    # seq_den = var_ref.sum(dim=-1).sqrt() 
    seq_num = label_log_probs.mean(dim=-1) - mean_ref.mean(dim=-1)  # [bsz]
    seq_den = var_ref.mean(dim=-1).sqrt() 
    
    seq_discrepancy = -seq_num / seq_den  # [bsz]
    # seq_discrepancy = seq_num / seq_den
    # multi-scale: local window discrepancy (e.g., 3-gram)
    window_size = 3
    if label_log_probs.size(1) >= window_size:
        # cumulative sums for O(1) window sums
        csum_label = torch.cumsum(label_log_probs, dim=1)  # [bsz, seq_len]
        csum_mean = torch.cumsum(mean_ref, dim=1)
        csum_var = torch.cumsum(var_ref, dim=1)

        # helper to get window sum via cumsum
        def window_sum(csum_tensor, w):
            # pad a zero at start to align indices
            pad = torch.zeros(csum_tensor.size(0), 1, device=csum_tensor.device, dtype=csum_tensor.dtype)
            csum_pad = torch.cat([pad, csum_tensor], dim=1)
            # sums over windows: S[i..i+w-1] = csum[i+w]-csum[i]
            start = csum_pad[:, :-w]
            end = csum_pad[:, w:]
            return end - start  # [bsz, seq_len - w + 1]

        # win_label = window_sum(csum_label, window_size)
        # win_mean = window_sum(csum_mean, window_size)
        # win_var = window_sum(csum_var, window_size).clamp_min(1e-8)
        # local_discrepancies = (win_label - win_mean) / win_var.sqrt()  # [bsz, num_windows]
        # local_discrepancy = local_discrepancies.mean(dim=1)  # aggregate over windows -> [bsz]
        # win_label = window_sum(csum_label, window_size).mean(dim=-1)
        win_label = window_sum(csum_label, window_size).max(dim=-1).values
        print(window_sum(csum_label, window_size))
        print(win_label)
        win_mean = window_sum(csum_mean, window_size).mean(dim=-1)
        win_var = window_sum(csum_var, window_size).mean(dim=-1)
        local_discrepancy = -(win_label - win_mean) / win_var.sqrt() 
    else:
        local_discrepancy = seq_discrepancy
    # print(local_discrepancy)
    # print("-"*10)
    # print(seq_discrepancy)
    # fuse multi-scale
    
    discrepancy = seq_discrepancy  # [bsz]
    # discrepancy = seq_discrepancy

    weighted_log_likelihood = label_log_probs.sum(dim=-1)
    # weighted_log_likelihood = win_label
    
    # print(label_log_probs.sum(dim=-1))
    # print("-"*10)
    # print(win_label)
    if return_components:
        return discrepancy, weighted_log_likelihood, {
            'seq_discrepancy': seq_discrepancy,
            'local_discrepancy': local_discrepancy,
            'window_size': window_size,
        }
    return discrepancy, weighted_log_likelihood

class ComputeScore(nn.Module):
    def __init__(self, scoring_model_name, reference_model_name, SPOtrained=True, SPO_beta=0.5, dataset='xsum', device='cuda', cache_dir='./models'):
        super().__init__()
        self.device = device
        self.reference_model_name = get_model_fullname(reference_model_name)
        self.scoring_model_name = get_model_fullname(scoring_model_name)
        self.beta = SPO_beta
        
        def load_model(model_name, device, cache_dir, SPOtrained=True):
            model_fullname = get_model_fullname(model_name)
            print(f'Loading model {model_fullname}...')
            model_kwargs = {}
            if model_name in float16_models:
                model_kwargs.update(dict(torch_dtype=torch.float16))
            if 'gpt-j' in model_name:
                model_kwargs.update(dict(revision='float16'))
            if SPOtrained:
                model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
            else: # Load ablation finetuned model
                model = from_pretrained(AutoPeftModelForCausalLM, model_fullname, model_kwargs, cache_dir)
            print('Moving model to GPU...', end='', flush=True)
            start = time.time()
            model.to(device)
            print(f'DONE ({time.time() - start:.2f}s)')
            return model
        
        # load model
        self.scoring_tokenizer = load_tokenizer(scoring_model_name, dataset, cache_dir)
        scoring_model = load_model(scoring_model_name, device, cache_dir, SPOtrained)

        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )

        reference_model = load_model(reference_model_name, device, cache_dir, SPOtrained)
        
        if SPOtrained: 
            self.scoring_model = get_peft_model(scoring_model, self.peft_config)
            # self.reference_model = get_peft_model(reference_model, self.peft_config)
            self.reference_model = reference_model
            self.reference_model1 = get_peft_model(reference_model, self.peft_config)
        else: 
            self.scoring_model = scoring_model
            self.reference_model = reference_model
            
        
        self.reference_tokenizer = load_tokenizer(reference_model_name, dataset, cache_dir)
             
        self.criterion_fn = get_sampling_discrepancy_analytic
        self.forward = self.forward_SPO

        self.scoring_model.print_trainable_parameters()


    def print_gradient_requirement(self):
        for name, param in self.named_parameters():
            gradient_requirement = 'Requires Grad' if param.requires_grad else 'Does not require grad'
            color_code = '\033[92m' if param.requires_grad else '\033[91m'  # Green for requires grad, red for does not require grad
            reset_color = '\033[0m'  # Reset color after printing
            print(f"{name}: {color_code}{gradient_requirement}{reset_color}")

    def register_no_grad(self, module_names):
        for name, param in self.named_parameters():
            for selected_module in module_names:
                # print(selected_module, name)
                if selected_module in name:
                    param.requires_grad = False

    def save_pretrained(self, save_directory):
        """
        Save the model's state_dict to the specified directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "model.bin"))

    def from_pretrained(self, load_directory):
        """
        Load the model's state_dict from the specified directory.
        """
        if not os.path.exists(load_directory):
            raise ValueError(f"Directory {load_directory} does not exist.")

        self.load_state_dict(torch.load(os.path.join(load_directory, "model.bin"), map_location=self.device))

    def get_SPO_input(self, tokenized=None ,text=[""], labels=[""], training_module=False, return_components=False):
        if training_module:
            logits_score = self.scoring_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:]
            # logits_ref = (self.reference_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:] + self.reference_model1(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:])/2
            logits_ref = logits_score
            if self.reference_model_name != self.scoring_model_name:
                tokenized = self.reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False, add_special_tokens=True, return_attention_mask=True).to(self.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.reference_model(tokenized.input_ids).logits[:,:-1,:]
            crit_out = self.criterion_fn(logits_ref, logits_score, labels, return_components=return_components)
            if return_components:
                crit, SPO_input, components = crit_out
            else:
                crit, SPO_input = crit_out
        else:
            with torch.no_grad(): # get reference
                if self.reference_model_name != self.scoring_model_name:
                    tokenized = self.reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False ,add_special_tokens=True, return_attention_mask=True).to(self.device)
                    assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_score = self.reference_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:] # shape: [bsz, sentence_len, dim]
                # logits_score = (self.reference_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:] + self.reference_model1(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:])/2
                # logits_score = (self.reference_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:] + self.reference_model1(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:])/2
                
                # logits_score = self.scoring_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:] # shape: [bsz, sentence_len, dim]
                logits_ref = logits_score
                crit_out = self.criterion_fn(logits_ref, logits_score, labels, return_components=return_components)
                if return_components:
                    crit, SPO_input, components = crit_out
                else:
                    crit, SPO_input = crit_out
        if return_components:
            return crit, SPO_input, logits_score, components
        return crit, SPO_input, logits_score

    def forward_SPO(self, text):
        original_text = text[0]
        sampled_text = text[1]
        
        tokenized = self.scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
        labels = tokenized.input_ids[:, 1:] # shape: [bsz, sentence_len]
        ref_original = self.get_SPO_input(tokenized,original_text,labels, return_components=True)
        ref_original_crit, ref_disprefered_logprob, ref_original_logits_score, ref_original_comp = ref_original
        train_original = self.get_SPO_input(tokenized,original_text,labels,training_module=True, return_components=True)
        train_original_crit, train_disprefered_logprob, train_original_logits_score, train_original_comp = train_original
        
        tokenized = self.scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
        labels = tokenized.input_ids[:, 1:]
        ref_sampled = self.get_SPO_input(tokenized,sampled_text,labels, return_components=True)
        ref_sampled_crit, ref_prefered_logprob, ref_sampled_logits_score, ref_sampled_comp = ref_sampled
        train_sampled = self.get_SPO_input(tokenized,sampled_text,labels,training_module=True, return_components=True)
        train_sampled_crit, train_prefered_logprob, train_sampled_logits_score, train_sampled_comp = train_sampled
        
        SPOloss, prefered_relative_logprob, disprefered_relative_logprob, reward_accuracies, reward_margins = calculate_SPO_loss(train_prefered_logprob,train_disprefered_logprob,ref_prefered_logprob,ref_disprefered_logprob,beta=self.beta)
        output = dict(
            crit=[ref_original_crit, train_original_crit, ref_sampled_crit, train_sampled_crit],
            loss=SPOloss,
            components={
                'ref_original': ref_original_comp,
                'train_original': train_original_comp,
                'ref_sampled': ref_sampled_comp,
                'train_sampled': train_sampled_comp,
            }
        )
        return output

