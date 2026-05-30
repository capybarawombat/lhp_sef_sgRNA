import os
import json
import torch
import torch.nn as nn

class DualOracleFeedbackAgent:
    """
    Offline Reinforcement Learning Feedback Agent.
    Caches failed sequences (Negative Samples) rejected by downstream Oracles (Evo 2 & STRAND)
    and computes a differentiable gradient penalty tensor to steer C-DDM denoising steps
    away from repeating those mistakes.
    """
    def __init__(self, cache_path="configs/failed_sequences_cache.json", alphabet="ACGU"):
        self.cache_path = cache_path
        self.alphabet = alphabet
        self.char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
        self.cache = []
        self._load_cache()

    def _load_cache(self):
        """Loads cached failed sequences from a local JSON buffer."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = []
        else:
            self.cache = []

    def _save_cache(self):
        """Saves current cache to the local JSON buffer."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=4)

    def log_failed_sequence(self, sequence, penalty_score, oracle_source):
        """
        Logs a failed sequence rejected by Phase 3 (Evo 2 or STRAND) with its penalty score.
        
        Args:
            sequence (str): The 20-bp sgRNA string.
            penalty_score (float): Numeric score indicating failure intensity.
            oracle_source (str): Source of failure (e.g. 'Evo2_Safety', 'STRAND_Efficacy').
        """
        clean_seq = sequence.upper().replace("T", "U")
        # Prevent duplicates
        if not any(item["sequence"] == clean_seq for item in self.cache):
            self.cache.append({
                "sequence": clean_seq,
                "penalty_score": float(penalty_score),
                "oracle_source": oracle_source
            })
            self._save_cache()

    def compute_feedback_penalty(self, generated_logits):
        """
        Compares currently generating sequence logits against cached failed sequences
        and returns a differentiable PyTorch penalty tensor.
        
        Args:
            generated_logits (torch.Tensor): Logits of shape (batch_size, sequence_length, alphabet_size)
                                            representing categorical nucleotide probabilities.
        Returns:
            torch.Tensor: Differentiable scalar penalty loss (L_Feedback).
        """
        if not self.cache:
            return torch.tensor(0.0, device=generated_logits.device, requires_grad=True)

        batch_size, seq_len, vocab_size = generated_logits.shape
        probs = torch.softmax(generated_logits, dim=-1) # Shape: (B, L, V)
        
        penalty_loss = 0.0
        
        # Calculate differentiable similarity to failed cached sequences
        for item in self.cache:
            seq = item["sequence"]
            penalty_score = item["penalty_score"]
            
            if len(seq) != seq_len:
                continue
            
            # Construct one-hot target tensor for the failed sequence
            target_indices = []
            for char in seq:
                idx = self.char_to_idx.get(char, 0)
                target_indices.append(idx)
            
            # Target tensor shape: (L,)
            target_indices = torch.tensor(target_indices, device=generated_logits.device, dtype=torch.long)
            
            # Gather probability of generating this specific failed sequence per batch item
            # probs: (B, L, V) -> select across dim=-1 using target_indices (L,)
            # For each batch element, compute joint probability (or average log-prob) of sequence matching the failure
            failed_seq_probs = probs[:, torch.arange(seq_len), target_indices] # Shape: (B, L)
            
            # Joint probability of generating this failed sequence (product over sequence length)
            # Use log-space for stability, then exponentiate: exp(sum(log(p + eps)))
            joint_prob = torch.exp(torch.sum(torch.log(failed_seq_probs + 1e-8), dim=-1)) # Shape: (B,)
            
            # Add penalty scaled by the oracle's failure severity
            penalty_loss += torch.mean(joint_prob) * penalty_score

        return penalty_loss
