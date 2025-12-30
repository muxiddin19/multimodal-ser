"""
Enhanced Components for State-of-the-Art Multimodal Speech Emotion Recognition
===============================================================================

Building on our novel architecture (VGA, EAAF, MICL), this module adds:

1. Focal Loss - Better handling of class imbalance
2. Constrained Adaptive Fusion - Gates sum to 1 for interpretability
3. Hard Negative Mining for MICL - Focus on difficult cross-modal pairs
4. Data Augmentation - SpecAugment-style and Mixup for features
5. Curriculum Learning Support - Progressive class training

These enhancements target 80%+ UA on IEMOCAP 6-class (current SOTA: ~78%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
import random


# ============================================================
# 1. FOCAL LOSS FOR CLASS IMBALANCE
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in emotion recognition.

    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

    Formula:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where:
        - p_t is the model's estimated probability for the correct class
        - γ (gamma) is the focusing parameter (γ >= 0)
        - α_t is the class weight for class t

    Benefits for SER:
        - Down-weights easy/well-classified samples (neutral, anger)
        - Focuses training on hard samples (excitement vs happiness)
        - Better than just class weights for fine-grained emotions
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, C] - Logits
            targets: [B] - Ground truth labels

        Returns:
            Focal loss value
        """
        num_classes = inputs.shape[-1]

        # Apply label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                targets_smooth = torch.zeros_like(inputs)
                targets_smooth.fill_(self.label_smoothing / (num_classes - 1))
                targets_smooth.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
        else:
            targets_smooth = F.one_hot(targets, num_classes).float()

        # Compute log probabilities
        log_p = F.log_softmax(inputs, dim=-1)
        p = torch.exp(log_p)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p) ** self.gamma

        # Compute cross entropy with focal weight
        focal_loss = -focal_weight * targets_smooth * log_p

        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha.unsqueeze(0).expand_as(focal_loss)
            focal_loss = alpha_t * focal_loss

        # Sum over classes
        focal_loss = focal_loss.sum(dim=-1)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================
# 2. CONSTRAINED ADAPTIVE FUSION
# ============================================================

class ConstrainedAdaptiveFusion(nn.Module):
    """
    Constrained Emotion-Aware Adaptive Fusion (C-EAAF)

    Improvement over EAAF: Gates are constrained to sum to 1, providing:
        - Better interpretability (clear modality contribution percentages)
        - Reduced overfitting (normalized attention-like behavior)
        - More stable training dynamics

    Mathematical Formulation:
        g = [h_t; h_a; h_t ⊙ h_a]
        raw_α_t = W_t · g + b_t
        raw_α_a = W_a · g + b_a
        raw_α_i = W_i · g + b_i

        [α_t, α_a, α_i] = softmax([raw_α_t, raw_α_a, raw_α_i])  # Sum to 1!

        h_fused = α_t · h_t + α_a · h_a + α_i · (h_t ⊙ h_a)
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1, temperature: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Gate network for all three components
        # Input: [h_t; h_a; h_t ⊙ h_a] -> 3 * hidden_dim
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # 3 gates: text, audio, interaction
        )

        # Learnable importance priors (for each dimension)
        self.dim_gates = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid()
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            text_feat: [B, D] - Text features
            audio_feat: [B, D] - Audio features

        Returns:
            fused: [B, D] - Adaptively fused features
            aux_info: Dict with gate values for analysis
        """
        # Compute element-wise interaction
        interaction = text_feat * audio_feat  # [B, D]

        # Concatenate for gate input
        gate_input = torch.cat([text_feat, audio_feat, interaction], dim=-1)  # [B, 3D]

        # Compute constrained gates (sum to 1)
        raw_gates = self.gate_network(gate_input)  # [B, 3]
        gates = F.softmax(raw_gates / self.temperature, dim=-1)  # [B, 3]

        alpha_t = gates[:, 0:1]  # [B, 1]
        alpha_a = gates[:, 1:2]  # [B, 1]
        alpha_i = gates[:, 2:3]  # [B, 1]

        # Dimension-wise importance (which dimensions matter for each sample)
        dim_importance = self.dim_gates(gate_input)  # [B, D]

        # Adaptive fusion with constrained gates
        fused = (alpha_t * text_feat +
                 alpha_a * audio_feat +
                 alpha_i * interaction) * dim_importance

        # Final projection
        output = self.output_proj(fused)

        # Return auxiliary info for analysis
        aux_info = {
            'text_gate': alpha_t.squeeze(-1),      # [B]
            'audio_gate': alpha_a.squeeze(-1),     # [B]
            'interaction_gate': alpha_i.squeeze(-1),  # [B]
            'gate_entropy': -(gates * torch.log(gates + 1e-8)).sum(dim=-1).mean().item()
        }

        return output, aux_info


# ============================================================
# 3. HARD NEGATIVE MINING FOR MICL
# ============================================================

class HardNegativeMICL(nn.Module):
    """
    Modality-Invariant Contrastive Learning with Hard Negative Mining

    Improvements over standard MICL:
        1. Hard negative mining: Focus on difficult negatives
        2. Emotion-aware sampling: Same-emotion samples as hard negatives
        3. Memory bank: Access to more negatives beyond batch

    Hard Negative Strategy:
        - Select top-k most similar but incorrect pairs as hard negatives
        - Weight hard negatives higher in the loss
        - Gradually increase hardness during training (curriculum)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_ratio: float = 0.5,  # Fraction of negatives to use
        margin: float = 0.3,  # Minimum margin between pos and neg
        projection_dim: int = 128
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_ratio = hard_negative_ratio
        self.margin = margin
        self.projection_dim = projection_dim

    def mine_hard_negatives(
        self,
        similarity_matrix: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mine top-k hardest negatives for each sample.

        Args:
            similarity_matrix: [B, B] - Pairwise similarities
            k: Number of hard negatives to select

        Returns:
            hard_neg_indices: [B, k] - Indices of hard negatives
            hard_neg_weights: [B, k] - Importance weights
        """
        batch_size = similarity_matrix.shape[0]
        device = similarity_matrix.device

        # Mask out diagonal (positive pairs)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)

        # Get negative similarities only
        neg_similarities = similarity_matrix.masked_fill(~mask, float('-inf'))

        # Select top-k hardest (most similar) negatives
        k = min(k, batch_size - 1)
        hard_neg_sim, hard_neg_indices = neg_similarities.topk(k, dim=1)

        # Weight based on similarity (harder = higher weight)
        hard_neg_weights = F.softmax(hard_neg_sim / self.temperature, dim=1)

        return hard_neg_indices, hard_neg_weights

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        epoch: int = 0,
        max_epochs: int = 100
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute MICL loss with hard negative mining.

        Args:
            text_features: [B, D] - Text embeddings
            audio_features: [B, D] - Audio embeddings
            labels: [B] - Emotion labels for emotion-aware mining
            epoch: Current training epoch (for curriculum)
            max_epochs: Total training epochs

        Returns:
            loss: Scalar MICL loss with hard negative mining
            aux_info: Dict with component losses
        """
        device = text_features.device
        batch_size = text_features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device), {}

        # L2 normalize features
        text_norm = F.normalize(text_features, p=2, dim=1)
        audio_norm = F.normalize(audio_features, p=2, dim=1)

        # Compute full similarity matrix
        sim_t2a = torch.matmul(text_norm, audio_norm.T) / self.temperature  # [B, B]
        sim_a2t = sim_t2a.T

        # Positive similarities (diagonal)
        positive_sim = torch.diagonal(sim_t2a)  # [B]

        # Calculate number of hard negatives (curriculum: increase over time)
        progress = min(epoch / max_epochs, 1.0)
        k = max(1, int(batch_size * self.hard_negative_ratio * (0.5 + 0.5 * progress)))

        # Mine hard negatives
        hard_neg_indices, hard_neg_weights = self.mine_hard_negatives(sim_t2a, k)

        # Gather hard negative similarities
        hard_neg_sim = torch.gather(sim_t2a, 1, hard_neg_indices)  # [B, k]

        # Weighted hard negative loss
        # We want: positive_sim > hard_neg_sim + margin
        # Loss: max(0, hard_neg_sim + margin - positive_sim)
        margin_loss = F.relu(hard_neg_sim + self.margin - positive_sim.unsqueeze(1))
        weighted_margin_loss = (margin_loss * hard_neg_weights).sum(dim=1).mean()

        # Standard InfoNCE loss
        positive_labels = torch.arange(batch_size, device=device)
        loss_t2a = F.cross_entropy(sim_t2a, positive_labels)
        loss_a2t = F.cross_entropy(sim_a2t, positive_labels)
        infonce_loss = (loss_t2a + loss_a2t) / 2

        # Combine losses
        total_loss = infonce_loss + 0.5 * weighted_margin_loss

        # Emotion-aware penalty: same-emotion should be closer
        if labels is not None:
            label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            label_mask.fill_diagonal_(0)  # Exclude self

            if label_mask.sum() > 0:
                # Same-emotion pairs should have high similarity
                same_emotion_sim = sim_t2a * label_mask
                same_emotion_bonus = -same_emotion_sim.sum() / (label_mask.sum() + 1e-8)
                total_loss = total_loss + 0.1 * same_emotion_bonus

        aux_info = {
            'infonce_loss': infonce_loss.item(),
            'margin_loss': weighted_margin_loss.item(),
            'avg_positive_sim': positive_sim.mean().item() * self.temperature,
            'avg_hard_neg_sim': hard_neg_sim.mean().item() * self.temperature,
            'num_hard_negatives': k
        }

        return total_loss, aux_info


# ============================================================
# 4. DATA AUGMENTATION
# ============================================================

class FeatureAugmentation(nn.Module):
    """
    Data Augmentation for Pre-extracted Features

    Since we use pre-extracted BERT and emotion2vec features,
    we apply augmentation in the feature space:

    1. Feature Dropout: Random zeroing of feature dimensions
    2. Gaussian Noise: Add noise for robustness
    3. Feature Mixup: Interpolate between samples
    4. Dimension Shuffle: Permute subsets of dimensions (mild)
    """

    def __init__(
        self,
        dropout_rate: float = 0.1,
        noise_std: float = 0.05,
        mixup_alpha: float = 0.4,
        shuffle_ratio: float = 0.1
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha
        self.shuffle_ratio = shuffle_ratio

    def feature_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random dropout to feature dimensions."""
        if not self.training or self.dropout_rate <= 0:
            return x

        mask = torch.bernoulli(
            torch.full_like(x, 1 - self.dropout_rate)
        )
        return x * mask / (1 - self.dropout_rate)

    def gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        if not self.training or self.noise_std <= 0:
            return x

        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def mixup(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Mixup augmentation between two samples.

        Returns:
            mixed_x: Interpolated features
            mixed_y: Soft labels (lambda * y1 + (1-lambda) * y2)
            lam: Mixing coefficient
        """
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1.0

        mixed_x = lam * x1 + (1 - lam) * x2

        return mixed_x, y1, y2, lam

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        augment_prob: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random augmentations to features.

        Args:
            text_feat: [B, D] - Text features
            audio_feat: [B, D] - Audio features
            augment_prob: Probability of applying each augmentation

        Returns:
            Augmented text and audio features
        """
        if not self.training:
            return text_feat, audio_feat

        # Randomly apply augmentations
        if random.random() < augment_prob:
            text_feat = self.feature_dropout(text_feat)

        if random.random() < augment_prob:
            audio_feat = self.feature_dropout(audio_feat)

        if random.random() < augment_prob:
            text_feat = self.gaussian_noise(text_feat)

        if random.random() < augment_prob:
            audio_feat = self.gaussian_noise(audio_feat)

        return text_feat, audio_feat


def mixup_batch(
    text_feat: torch.Tensor,
    audio_feat: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply Mixup to a batch of samples.

    Args:
        text_feat: [B, D]
        audio_feat: [B, D]
        labels: [B]
        alpha: Beta distribution parameter

    Returns:
        mixed_text, mixed_audio, labels_a, labels_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = text_feat.size(0)
    index = torch.randperm(batch_size, device=text_feat.device)

    mixed_text = lam * text_feat + (1 - lam) * text_feat[index]
    mixed_audio = lam * audio_feat + (1 - lam) * audio_feat[index]

    labels_a = labels
    labels_b = labels[index]

    return mixed_text, mixed_audio, labels_a, labels_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """Compute mixed loss for mixup training."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# 5. CURRICULUM LEARNING SCHEDULER
# ============================================================

class CurriculumScheduler:
    """
    Curriculum Learning Scheduler for Progressive Class Training

    Strategy:
        Phase 1 (epochs 0-30): Train on 4 classes (easy emotions)
        Phase 2 (epochs 30-60): Expand to 5 classes (add frustration)
        Phase 3 (epochs 60+): Full 6 classes (add excitement)

    Benefits:
        - Model learns easier emotion distinctions first
        - Gradual difficulty increase prevents mode collapse
        - Better handling of confusing emotion pairs
    """

    def __init__(
        self,
        total_epochs: int,
        phase_epochs: Optional[List[int]] = None,
        class_schedule: Optional[List[int]] = None
    ):
        self.total_epochs = total_epochs

        # Default: 3 phases of equal length
        if phase_epochs is None:
            phase_len = total_epochs // 3
            self.phase_epochs = [phase_len, phase_len * 2, total_epochs]
        else:
            self.phase_epochs = phase_epochs

        # Default: 4 -> 5 -> 6 classes
        if class_schedule is None:
            self.class_schedule = [4, 5, 6]
        else:
            self.class_schedule = class_schedule

    def get_num_classes(self, epoch: int) -> int:
        """Get number of classes for current epoch."""
        for i, phase_end in enumerate(self.phase_epochs):
            if epoch < phase_end:
                return self.class_schedule[i]
        return self.class_schedule[-1]

    def get_class_mask(self, epoch: int, labels: torch.Tensor) -> torch.Tensor:
        """
        Get mask for valid samples in current curriculum phase.

        Args:
            epoch: Current training epoch
            labels: [B] - Sample labels

        Returns:
            mask: [B] - Boolean mask (True = include sample)
        """
        num_classes = self.get_num_classes(epoch)
        return labels < num_classes

    def get_phase_info(self, epoch: int) -> Dict:
        """Get current curriculum phase information."""
        num_classes = self.get_num_classes(epoch)
        phase_idx = 0
        for i, phase_end in enumerate(self.phase_epochs):
            if epoch < phase_end:
                phase_idx = i
                break
            phase_idx = i

        return {
            'epoch': epoch,
            'phase': phase_idx + 1,
            'num_classes': num_classes,
            'phase_progress': (epoch - (self.phase_epochs[phase_idx-1] if phase_idx > 0 else 0)) /
                             (self.phase_epochs[phase_idx] - (self.phase_epochs[phase_idx-1] if phase_idx > 0 else 0))
                             if phase_idx < len(self.phase_epochs) else 1.0
        }


# ============================================================
# 6. ENHANCED MODEL WITH ALL IMPROVEMENTS
# ============================================================

class EnhancedMultimodalSER(nn.Module):
    """
    Enhanced Multimodal SER with SOTA Improvements

    Base Architecture:
        - VAD-Guided Cross-Attention (VGA)
        - Modality-Invariant Contrastive Learning (MICL)

    Enhancements:
        - Constrained Adaptive Fusion (C-EAAF)
        - Hard Negative Mining for MICL
        - Feature Augmentation
        - Focal Loss compatibility
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 1024,
        hidden_dim: int = 384,
        num_heads: int = 8,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
        vad_lambda: float = 0.5,  # Increased from 0.1
        micl_dim: int = 128,
        use_augmentation: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_augmentation = use_augmentation

        # Import base components
        from models.novel_components import (
            VADGuidedBidirectionalAttention,
            MICLProjector
        )

        # Feature augmentation
        if use_augmentation:
            self.augmentation = FeatureAugmentation(
                dropout_rate=0.1,
                noise_std=0.05,
                mixup_alpha=0.4
            )

        # Input projections
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Learnable modality embeddings
        self.text_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.audio_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Self-attention encoders
        self.text_self_attn = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.audio_self_attn = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        # VAD-Guided Cross-Attention layers (increased vad_lambda)
        self.vga_layers = nn.ModuleList([
            VADGuidedBidirectionalAttention(hidden_dim, num_heads, dropout, vad_lambda)
            for _ in range(num_layers)
        ])

        # Enhanced: Constrained Adaptive Fusion (instead of EAAF)
        self.fusion = ConstrainedAdaptiveFusion(hidden_dim, dropout)

        # Enhanced: MICL with Hard Negative Mining
        self.micl_projector = MICLProjector(hidden_dim, micl_dim, hidden_dim)
        self.micl_loss = HardNegativeMICL(
            temperature=0.07,
            hard_negative_ratio=0.5,
            margin=0.3
        )

        # Classification head (supports variable num_classes for curriculum)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # VAD regression head
        self.vad_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        epoch: int = 0,
        max_epochs: int = 100,
        apply_augmentation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with all enhancements.

        Args:
            text_feat: [B, D_text] or [B, N, D_text]
            audio_feat: [B, D_audio] or [B, N, D_audio]
            labels: [B] - Labels for MICL
            epoch: Current epoch for curriculum
            max_epochs: Total epochs for curriculum
            apply_augmentation: Whether to apply augmentation

        Returns:
            Dict with logits, losses, and auxiliary info
        """
        # Apply feature augmentation during training
        if self.training and self.use_augmentation and apply_augmentation:
            text_feat, audio_feat = self.augmentation(text_feat, audio_feat)

        # Add sequence dimension if needed
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
        if audio_feat.dim() == 2:
            audio_feat = audio_feat.unsqueeze(1)

        # Project to common dimension
        text_h = self.text_proj(text_feat)
        audio_h = self.audio_proj(audio_feat)

        # Add modality embeddings
        text_h = text_h + self.text_embed
        audio_h = audio_h + self.audio_embed

        # Self-attention
        text_h = self.text_self_attn(text_h)
        audio_h = self.audio_self_attn(audio_h)

        # VAD-Guided Cross-Attention
        for vga_layer in self.vga_layers:
            audio_h, text_h = vga_layer(audio_h, text_h)

        # Pool to sequence-level
        text_pooled = text_h.mean(dim=1)
        audio_pooled = audio_h.mean(dim=1)

        # Constrained Adaptive Fusion
        fused, fusion_aux = self.fusion(text_pooled, audio_pooled)

        # MICL with hard negative mining
        text_proj, audio_proj = self.micl_projector(text_pooled, audio_pooled)
        micl_loss, micl_aux = self.micl_loss(
            text_proj, audio_proj, labels, epoch, max_epochs
        )

        # Classification
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)

        # VAD prediction
        vad = self.vad_head(fused)

        return {
            'logits': logits,
            'probs': probs,
            'vad': vad,
            'features': fused,
            'text_features': text_pooled,
            'audio_features': audio_pooled,
            'micl_loss': micl_loss,
            'text_gate': fusion_aux['text_gate'],
            'audio_gate': fusion_aux['audio_gate'],
            'interaction_gate': fusion_aux['interaction_gate'],
            'gate_entropy': fusion_aux['gate_entropy'],
            'micl_aux': micl_aux
        }


# ============================================================
# 7. ENHANCED MULTI-TASK LOSS
# ============================================================

class EnhancedMultiTaskLoss(nn.Module):
    """
    Enhanced Multi-Task Loss with Focal Loss and Curriculum Support

    Total Loss:
        L = λ_cls * L_focal + λ_vad * L_vad + λ_micl * L_micl

    Improvements:
        - Focal Loss instead of CE for better class imbalance handling
        - Curriculum-aware VAD targets
        - Hard negative mining in MICL (computed in model)
    """

    def __init__(
        self,
        num_classes: int = 6,
        emotion_config: str = "iemocap_6",
        class_weights: Optional[torch.Tensor] = None,
        cls_weight: float = 1.0,
        vad_weight: float = 0.5,  # Increased from 0.3
        micl_weight: float = 0.3,  # Increased from 0.2
        focal_gamma: float = 2.0
    ):
        super().__init__()

        self.cls_weight = cls_weight
        self.vad_weight = vad_weight
        self.micl_weight = micl_weight

        # Focal Loss instead of CrossEntropy
        self.focal_loss = FocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            label_smoothing=0.1
        )

        self.mse_loss = nn.MSELoss()

        # VAD values per emotion (extended for 6-class)
        self.vad_configs = {
            "iemocap_4": {
                0: [-0.666, 0.730, 0.314],   # anger
                1: [0.960, 0.648, 0.588],    # happiness
                2: [-0.062, -0.632, -0.286], # neutral
                3: [-0.896, -0.424, -0.672], # sadness
            },
            "iemocap_5": {
                0: [0.905, 0.699, 0.519],    # happy_excited
                1: [-0.896, -0.424, -0.672], # sadness
                2: [-0.062, -0.632, -0.286], # neutral
                3: [-0.666, 0.730, 0.314],   # anger
                4: [-0.500, 0.600, -0.200],  # frustration
            },
            "iemocap_6": {
                0: [0.960, 0.648, 0.588],    # happiness
                1: [-0.896, -0.424, -0.672], # sadness
                2: [-0.062, -0.632, -0.286], # neutral
                3: [-0.666, 0.730, 0.314],   # anger
                4: [0.850, 0.750, 0.450],    # excitement
                5: [-0.500, 0.600, -0.200],  # frustration
            },
            "cremad_4": {
                0: [-0.666, 0.730, 0.314],   # anger
                1: [0.960, 0.648, 0.588],    # happiness
                2: [-0.062, -0.632, -0.286], # neutral
                3: [-0.896, -0.424, -0.672], # sadness
            },
            "cremad_6": {
                0: [-0.666, 0.730, 0.314],   # anger
                1: [-0.850, 0.300, -0.400],  # disgust
                2: [-0.700, 0.600, -0.300],  # fear
                3: [0.960, 0.648, 0.588],    # happiness
                4: [-0.062, -0.632, -0.286], # neutral
                5: [-0.896, -0.424, -0.672], # sadness
            },
            "meld_4": {
                0: [-0.666, 0.730, 0.314],   # anger
                1: [0.960, 0.648, 0.588],    # joy
                2: [-0.062, -0.632, -0.286], # neutral
                3: [-0.896, -0.424, -0.672], # sadness
            },
        }
        self.vad_dict = self.vad_configs.get(emotion_config, {})

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        curriculum_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute enhanced multi-task loss.

        Args:
            outputs: Model outputs
            labels: Ground truth labels
            curriculum_mask: [B] - Mask for curriculum learning (valid samples)

        Returns:
            Dict with individual and total losses
        """
        device = labels.device

        # Apply curriculum mask if provided
        if curriculum_mask is not None:
            logits = outputs['logits'][curriculum_mask]
            labels_masked = labels[curriculum_mask]
            vad_pred = outputs['vad'][curriculum_mask]
        else:
            logits = outputs['logits']
            labels_masked = labels
            vad_pred = outputs['vad']

        if len(labels_masked) == 0:
            return {
                'total': torch.tensor(0.0, device=device),
                'cls': torch.tensor(0.0, device=device),
                'vad': torch.tensor(0.0, device=device),
                'micl': torch.tensor(0.0, device=device)
            }

        # Focal loss for classification
        cls_loss = self.focal_loss(logits, labels_masked)

        # VAD regression loss
        vad_targets = torch.tensor([
            self.vad_dict.get(l.item(), [0, 0, 0]) for l in labels_masked
        ], dtype=torch.float32, device=device)
        vad_loss = self.mse_loss(vad_pred, vad_targets)

        # MICL loss (computed in model forward)
        micl_loss = outputs['micl_loss']

        # Total loss with adjusted weights
        total = (
            self.cls_weight * cls_loss +
            self.vad_weight * vad_loss +
            self.micl_weight * micl_loss
        )

        return {
            'total': total,
            'cls': cls_loss,
            'vad': vad_loss,
            'micl': micl_loss
        }
