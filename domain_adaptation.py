"""
Domain Adaptation techniques for cross-dataset Speech Emotion Recognition
Implements:
1. Domain Adversarial Neural Network (DANN)
2. Maximum Mean Discrepancy (MMD) loss
3. Coral loss for domain alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np


# ============================================================
# 1. Gradient Reversal Layer (for DANN)
# ============================================================

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# ============================================================
# 2. Domain Discriminator (for DANN)
# ============================================================

class DomainDiscriminator(nn.Module):
    """Discriminates between source and target domains"""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, alpha=1.0):
        self.grl.alpha = alpha
        x = self.grl(x)
        return self.classifier(x)


# ============================================================
# 3. MMD Loss (Maximum Mean Discrepancy)
# ============================================================

def compute_mmd(source_features, target_features, kernel='rbf'):
    """
    Compute Maximum Mean Discrepancy between source and target features
    """
    batch_size = min(source_features.size(0), target_features.size(0))
    source_features = source_features[:batch_size]
    target_features = target_features[:batch_size]

    if kernel == 'rbf':
        # RBF kernel
        gamma = 1.0 / source_features.size(1)

        ss = torch.exp(-gamma * torch.cdist(source_features, source_features, p=2).pow(2))
        tt = torch.exp(-gamma * torch.cdist(target_features, target_features, p=2).pow(2))
        st = torch.exp(-gamma * torch.cdist(source_features, target_features, p=2).pow(2))

        mmd = ss.mean() + tt.mean() - 2 * st.mean()

    elif kernel == 'linear':
        # Linear kernel
        ss = torch.mm(source_features, source_features.t()).mean()
        tt = torch.mm(target_features, target_features.t()).mean()
        st = torch.mm(source_features, target_features.t()).mean()

        mmd = ss + tt - 2 * st

    return mmd


class MMDLoss(nn.Module):
    """MMD loss for domain adaptation"""
    def __init__(self, kernel='rbf'):
        super().__init__()
        self.kernel = kernel

    def forward(self, source_features, target_features):
        return compute_mmd(source_features, target_features, self.kernel)


# ============================================================
# 4. CORAL Loss (Correlation Alignment)
# ============================================================

class CoralLoss(nn.Module):
    """
    CORAL loss - aligns second-order statistics (covariances) of source and target
    """
    def __init__(self):
        super().__init__()

    def forward(self, source_features, target_features):
        d = source_features.size(1)

        # Source covariance
        source_mean = source_features.mean(0, keepdim=True)
        source_centered = source_features - source_mean
        source_cov = source_centered.t().mm(source_centered) / (source_features.size(0) - 1)

        # Target covariance
        target_mean = target_features.mean(0, keepdim=True)
        target_centered = target_features - target_mean
        target_cov = target_centered.t().mm(target_centered) / (target_features.size(0) - 1)

        # CORAL loss
        loss = (source_cov - target_cov).pow(2).sum() / (4 * d * d)

        return loss


# ============================================================
# 5. Domain Adaptation Training Loop
# ============================================================

def train_with_domain_adaptation(
    model,
    domain_disc,
    source_loader,
    target_loader,
    optimizer,
    criterion,
    device,
    epoch,
    num_epochs,
    adaptation_method='dann',  # 'dann', 'mmd', 'coral'
    lambda_domain=0.1
):
    """
    Train with domain adaptation

    Args:
        model: Main emotion classifier
        domain_disc: Domain discriminator (for DANN)
        source_loader: DataLoader for source domain (labeled)
        target_loader: DataLoader for target domain (unlabeled)
        optimizer: Optimizer for all parameters
        criterion: Classification loss
        device: torch device
        epoch: Current epoch
        num_epochs: Total epochs
        adaptation_method: 'dann', 'mmd', or 'coral'
        lambda_domain: Weight for domain adaptation loss
    """
    model.train()
    if domain_disc is not None:
        domain_disc.train()

    # For DANN: gradually increase alpha
    p = epoch / num_epochs
    alpha = 2. / (1. + np.exp(-10 * p)) - 1

    mmd_loss_fn = MMDLoss(kernel='rbf')
    coral_loss_fn = CoralLoss()

    total_cls_loss = 0
    total_domain_loss = 0

    target_iter = iter(target_loader)

    for batch_idx, (text_feat, audio_feat, labels) in enumerate(source_loader):
        # Get target batch
        try:
            target_text, target_audio, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_text, target_audio, _ = next(target_iter)

        # Move to device
        text_feat = text_feat.to(device)
        audio_feat = audio_feat.to(device)
        labels = labels.to(device)
        target_text = target_text.to(device)
        target_audio = target_audio.to(device)

        optimizer.zero_grad()

        # Forward pass - source
        source_outputs = model(text_feat, audio_feat)
        source_features = source_outputs['features']

        # Classification loss (source only - has labels)
        cls_loss = criterion(source_outputs, labels)['total']

        # Forward pass - target (no labels)
        with torch.no_grad() if adaptation_method != 'dann' else torch.enable_grad():
            target_outputs = model(target_text, target_audio)
            target_features = target_outputs['features']

        # Domain adaptation loss
        if adaptation_method == 'dann':
            # Domain adversarial loss
            source_domain_pred = domain_disc(source_features, alpha)
            target_domain_pred = domain_disc(target_features, alpha)

            # Source = 0, Target = 1
            source_domain_labels = torch.zeros(source_features.size(0), 1).to(device)
            target_domain_labels = torch.ones(target_features.size(0), 1).to(device)

            domain_loss = F.binary_cross_entropy_with_logits(source_domain_pred, source_domain_labels) + \
                          F.binary_cross_entropy_with_logits(target_domain_pred, target_domain_labels)

        elif adaptation_method == 'mmd':
            domain_loss = mmd_loss_fn(source_features, target_features)

        elif adaptation_method == 'coral':
            domain_loss = coral_loss_fn(source_features, target_features)

        else:
            domain_loss = torch.tensor(0.0).to(device)

        # Total loss
        total_loss = cls_loss + lambda_domain * domain_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_cls_loss += cls_loss.item()
        total_domain_loss += domain_loss.item()

    return {
        'cls_loss': total_cls_loss / len(source_loader),
        'domain_loss': total_domain_loss / len(source_loader),
        'alpha': alpha
    }


# ============================================================
# 6. Simple usage example
# ============================================================

if __name__ == "__main__":
    print("Domain Adaptation Module for SER")
    print("Available methods:")
    print("  1. DANN - Domain Adversarial Neural Network")
    print("  2. MMD - Maximum Mean Discrepancy")
    print("  3. CORAL - Correlation Alignment")
    print("\nUsage: Import and use train_with_domain_adaptation()")
