import torch.nn.functional as F

def calculate_SPO_loss(model_prefered_entropy, model_disprefered_entropy,
                       ref_prefered_entropy, ref_disprefered_entropy,
                       beta=0.5):

    # Calculate entropy differences instead of log probability differences
    prefered_relative_entropy = model_prefered_entropy - ref_prefered_entropy
    disprefered_relative_entropy = model_disprefered_entropy - ref_disprefered_entropy

    reward_accuracies = (prefered_relative_entropy > disprefered_relative_entropy).float().mean(dim=-1)
    reward_margins = (prefered_relative_entropy - disprefered_relative_entropy).mean(dim=-1)

    # Use entropy differences in the loss calculation
    loss = -F.logsigmoid(beta * (prefered_relative_entropy - disprefered_relative_entropy)).mean(dim=-1)
    
    return loss, prefered_relative_entropy.mean(dim=-1), disprefered_relative_entropy.mean(dim=-1), reward_accuracies, reward_margins