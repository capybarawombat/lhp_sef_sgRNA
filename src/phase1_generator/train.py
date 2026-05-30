"""Denoising Markov training loop with Classifier-Free Guidance and Offline RL Penalty Feedback."""

import torch
import torch.nn as nn
from src.phase1_generator.feedback_agent import DualOracleFeedbackAgent

def train_epoch(model, dataloader, optimizer, feedback_agent, device):
    """
    Standard training loop updated with Classifier-Free Guidance (CFG),
    Thermodynamic Loss (L_Allele), and Closed-Loop Feedback Loss (L_Feedback).
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        # Unpack batch
        # x_0: target sequence, condition_vector: conditional inputs c
        x_0, condition_vector = batch
        x_0 = x_0.to(device)
        condition_vector = condition_vector.to(device)
        
        optimizer.zero_grad()
        
        # 1. Standard Variational Lower Bound (VLB) Discrete Diffusion Loss
        # We sample a random timestep t and add categorical Markov noise
        t = torch.randint(0, 100, (x_0.size(0),), device=device)
        vlb_loss, generated_logits = model.compute_vlb_loss(x_0, condition_vector, t)
        
        # 2. Thermodynamic Penalty (L_Allele)
        # Evaluates spacer binding energy mutant vs wild-type to guide selective cleavage
        l_allele = model.compute_thermodynamic_penalty(generated_logits, condition_vector)
        
        # 3. Closed-Loop Feedback Loss (L_Feedback) from DualOracleFeedbackAgent
        # Penalizes generated distributions that closely resemble cached failures
        l_feedback = feedback_agent.compute_feedback_penalty(generated_logits)
        
        # Differentiable scalar feedback loss summed together
        loss = vlb_loss + l_allele + l_feedback
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

if __name__ == "__main__":
    print("PyTorch training module initialized.")
