import json
import os
import datetime
import time
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from metrics import get_roc_metrics, get_precision_recall_metrics
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW


def evaluate_model_SPO(model, data, DEVICE, save_dir=None, img_prefix="eval"):
    from utils_visualization import plot_token_crit_highlight

    model.to(DEVICE)
    model.eval()
    loss = 0
    eval_loader = DataLoader(data, batch_size=1, shuffle=False)
    epoch_crit_train_original, epoch_crit_train_sampled = [],[]
    seq_real_list, loc_real_list = [], []
    seq_sampled_list, loc_sampled_list = [], []
    
    # Read the first sample as raw text before tokenization.
    first_human_text = ""
    first_llm_text = ""
    first_human_tokens, first_human_crits = [], []
    first_llm_tokens, first_llm_crits = [], []
    # Get the first raw text pair directly from the dataset.
    if len(data) > 0:
        first_sample = data[0]  # The dataset is expected to return (human_text, llm_text).
        first_human_text = first_sample[0]
        first_llm_text = first_sample[1]
        print(f"\nSelected sample: {first_human_text[:50]}...")

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(eval_loader, desc="Evaluating")):
            text = batch
            output = model(text)
            loss += output['loss'].item()
            epoch_crit_train_original.extend(output['crit'][1].tolist())
            epoch_crit_train_sampled.extend(output['crit'][3].tolist())
            
            # Collect the scatter plot data.
            comp_train_orig = output['components']['train_original']
            comp_train_samp = output['components']['train_sampled']
            seq_real = comp_train_orig['seq_discrepancy'].detach().cpu().numpy().tolist()
            loc_real = comp_train_orig['local_discrepancy'].detach().cpu().numpy().tolist()
            seq_samp = comp_train_samp['seq_discrepancy'].detach().cpu().numpy().tolist()
            loc_samp = comp_train_samp['local_discrepancy'].detach().cpu().numpy().tolist()
            if isinstance(seq_real, list):
                seq_real_list.extend(seq_real)
                loc_real_list.extend(loc_real)
                seq_sampled_list.extend(seq_samp)
                loc_sampled_list.extend(loc_samp)
            else:
                seq_real_list.append(seq_real)
                loc_real_list.append(loc_real)
                seq_sampled_list.append(seq_samp)
                loc_sampled_list.append(loc_samp)
            
            # Compute tokens and criterion values from raw text once.
            if batch_idx == 0 and first_human_text != "" and first_llm_text != "":
                first_human_tokens, first_human_crits = model.get_token_level_crit(first_human_text)
                first_llm_tokens, first_llm_crits = model.get_token_level_crit(first_llm_text)
        
        # Compute evaluation metrics.
        print(f"Total time: {time.time() - start_time:.4f}s")
        avg_loss = loss / len(eval_loader)
        fpr, tpr, roc_auc = get_roc_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
        p, r, pr_auc = get_precision_recall_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
    
    # Build the results dictionary.
    results_dict = {
        "name": "SPO",
        'val_loss': avg_loss,
        'val_ROC_AUC': roc_auc,
        'val_PR_AUC': pr_auc,
        'val_crit_real_mean': np.mean(epoch_crit_train_original),
        'val_crit_real_std': np.std(epoch_crit_train_original),
        'val_crit_sampled_mean': np.mean(epoch_crit_train_sampled),
        'val_crit_sampled_std': np.std(epoch_crit_train_sampled),
        'crit_real': epoch_crit_train_original,
        'crit_sampled': epoch_crit_train_sampled,
        'fpr': fpr,
        'tpr': tpr,
        'vis_data': {
            'human_text': first_human_text,
            'human_tokens': first_human_tokens,
            'human_crits': first_human_crits,
            'llm_text': first_llm_text,
            'llm_tokens': first_llm_tokens,
            'llm_crits': first_llm_crits
        }
    }

    # Save the scatter plot.
    try:
        if save_dir is None:
            save_dir = "evaluation"
        os.makedirs(save_dir, exist_ok=True)
        
        existing_files = [f for f in os.listdir(save_dir) if f.startswith(img_prefix) and f.endswith('.png')]
        if existing_files:
            numbers = []
            for f in existing_files:
                try:
                    num = int(f.replace(img_prefix, '').replace('_seq_vs_window_scatter.png', '').strip('_'))
                    numbers.append(num)
                except:
                    pass
            next_num = max(numbers) + 1 if numbers else 1
        else:
            next_num = 1
        
        fig = plt.figure(figsize=(4.5,4))
        plt.scatter(seq_sampled_list, loc_sampled_list, s=18, alpha=0.6, label='LLM Text')
        plt.scatter(seq_real_list, loc_real_list, s=18, alpha=0.6, marker='^', label='Human Text')
        if len(seq_sampled_list) > 0:
            cx = float(np.mean(seq_sampled_list)); cy = float(np.mean(loc_sampled_list))
            plt.scatter([cx], [cy], marker='*', s=160, edgecolor='k', label='LLM Centroid')
        if len(seq_real_list) > 0:
            cx = float(np.mean(seq_real_list)); cy = float(np.mean(loc_real_list))
            plt.scatter([cx], [cy], marker='*', s=160, edgecolor='k', facecolors='none', label='Human Centroid')
        plt.xlabel('Sequence-level Score')
        plt.ylabel('Window-level Score')
        plt.grid(True, ls='--', alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        
        fig_path = os.path.join(save_dir, f'{img_prefix}_{next_num:04d}_seq_vs_window_scatter.png')
        plt.savefig(fig_path, dpi=300)
        plt.close(fig)
        results_dict['scatter_path'] = fig_path
        print(f"Scatter plot saved to: {fig_path}")
    except Exception as e:
        print(f"[warn] failed to save scatter: {e}")

    # Plot using tokens from the original text.
    try:
        if save_dir and len(first_human_tokens) > 0 and len(first_llm_tokens) > 0:
            plot_token_crit_highlight(
                human_tokens=first_human_tokens,
                human_crits=first_human_crits,
                llm_tokens=first_llm_tokens,
                llm_crits=first_llm_crits,
                save_dir=save_dir,
                img_prefix=img_prefix,
                epoch=next_num
            )
    except Exception as e:
        print(f"[warn] failed to plot token-crit highlight: {str(e)[:200]}")

    return results_dict


def fine_tune_ours(model, data, DEVICE, ckpt_dir='./ckpt', args=None):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create a directory for this run.
    run_dir = f"{args.task_name}_spo_lr_{args.lr}_beta_{args.beta}_a_{args.a}_{current_time}"
    log_dir = f"logs/{run_dir}/train_ai_detection"
    eval_img_dir = f"evaluation/{run_dir}"  # Image output directory.
    
    os.makedirs(eval_img_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    train_loader = DataLoader(data[0], batch_size=1, shuffle=True)
    epochs = args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs, eta_min=0,
                                  last_epoch=-1)

    scaler = GradScaler()
    model.to(DEVICE)

    # Number of iterations for gradient accumulation
    accumulation_steps = args.a
    epoch_losses, i, loss = [], 0, torch.tensor(0.0).to(DEVICE)
    epoch_crit_train_original, epoch_crit_train_sampled = [],[]
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        start_time = time.time()
        for batch in tqdm.tqdm(train_loader, desc=f"Fine-tuning: {epoch} epoch"):
            text = batch
            scheduler.step()
            with autocast():
                outputs_1 = model(text)
                epoch_crit_train_original.extend(outputs_1['crit'][1].tolist())
                epoch_crit_train_sampled.extend(outputs_1['crit'][3].tolist())
                loss += (outputs_1['loss'].to(torch.float32)) / accumulation_steps
            
            
            if ((i + 1) % accumulation_steps) == 0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()
                writer.add_scalar('Loss/train', loss.item(), i)  
                epoch_losses.append(loss.item())
                loss = torch.tensor(0.0).to(DEVICE)
            epoch_losses.append(loss.item())
            i += 1
        print(f"Total time: {time.time() - start_time:.4f}s") 
        fpr, tpr, roc_auc = get_roc_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
        p, r, pr_auc = get_precision_recall_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
        
        print(f"ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
        print(f"Real mean/std: {np.mean(epoch_crit_train_original):.2f}/{np.std(epoch_crit_train_original):.2f}, Samples mean/std: {np.mean(epoch_crit_train_sampled):.2f}/{np.std(epoch_crit_train_sampled):.2f}")
        epoch_avg_loss = np.mean(epoch_losses)
        
        writer.add_scalar('Loss/epoch', epoch_avg_loss, epoch)
        writer.add_scalar('ROC_AUC/epoch', roc_auc, epoch)
        writer.add_scalar('PR_AUC/epoch', pr_auc, epoch)
        writer.add_scalar('Real_mean/epoch',np.mean(epoch_crit_train_original),epoch)
        writer.add_scalar('Real_std/epoch',np.std(epoch_crit_train_original),epoch)
        writer.add_scalar('Sampled_mean/epoch',np.mean(epoch_crit_train_sampled),epoch)
        writer.add_scalar('Sampled_std/epoch',np.std(epoch_crit_train_sampled),epoch)
        epoch_crit_train_original, epoch_crit_train_sampled = [],[] # reset crit
        print(f"\nAverage Loss for Epoch {epoch}: {epoch_avg_loss}")
        
        if ((epoch+1) % args.val_freq)==0: # eval and save model every val_freq epoch
            
            dataset_name = args.eval_dataset.split("/")[-1].split("_")[0]
            print(f"Evaluating on {dataset_name}...")
            
            # Pass the output directory and a prefix that includes the epoch.
            img_prefix = f"epoch_{epoch+1:03d}_{dataset_name}"
            val_result = evaluate_model_SPO(model, data[1], DEVICE, 
                                           save_dir=eval_img_dir, 
                                           img_prefix=img_prefix)
            
            valwriter = SummaryWriter(log_dir=f"logs/{run_dir}/val_{dataset_name}")
            valwriter.add_scalar('val_Loss/epoch', val_result['val_loss'], epoch)
            valwriter.add_scalar('val_ROC_AUC/epoch', val_result['val_ROC_AUC'], epoch)
            valwriter.add_scalar('val_PR_AUC/epoch', val_result['val_PR_AUC'], epoch)
            valwriter.add_scalar('val_Real_mean/epoch', val_result['val_crit_real_mean'], epoch)
            valwriter.add_scalar('val_Real_std/epoch', val_result['val_crit_real_std'], epoch)
            valwriter.add_scalar('val_Sampled_mean/epoch', val_result['val_crit_sampled_mean'], epoch)
            valwriter.add_scalar('val_Sampled_std/epoch', val_result['val_crit_sampled_std'], epoch)
            valwriter.close()
            
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    model.save_pretrained(ckpt_dir)
    print(f"Saved finetuned model to {os.path.join(ckpt_dir, 'ours-finetuned.pth')}")
    
    writer.close()
    return model


def run(
    model, 
    data, 
    DEVICE,
    args,
    ckpt_dir='./ckpt',
    ):
    
    if args.ebt or args.eval_only:
        print("Evaluating model before tuning...")
        dataset_name = args.eval_dataset.split("/")[-1].split("_")[0]
        
        # Create a dedicated output directory in evaluation-only mode.
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        eval_img_dir = f"evaluation/eval_only_{dataset_name}"
        
        d = evaluate_model_SPO(model, data[1], DEVICE, 
                              save_dir=eval_img_dir, 
                              img_prefix="eval")
        
        if args.SPOtrained:
            output_path = os.path.join(args.output_file, f"{dataset_name}.spo.json" )
        else:
            method_name=args.base_model.split("_")[-1]
            output_path = os.path.join(args.output_file, f"{dataset_name}.{method_name}.json" )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as j:
            json.dump(d,j)
        print(f"Results saved to {output_path}.")
    
    if args.eval_only:
        return

    print('Fine-tuning model...')
    model = fine_tune_ours(
        model, 
        data,
        DEVICE=DEVICE, 
        ckpt_dir=ckpt_dir,
        args=args
    )