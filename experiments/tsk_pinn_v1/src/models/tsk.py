import torch
import torch.nn as nn


class TSKWeightHead(nn.Module):
    """Gaussian antecedent + first-order consequent for 3 loss logits."""

    def __init__(self, in_dim=2, hidden_dim=64, n_rules=6, bottleneck_dim=8, drop_rule=0.0):
        super().__init__()
        self.n_rules = n_rules
        self.drop_rule = drop_rule
        self.centers = nn.Parameter(torch.randn(n_rules, in_dim) * 0.5)
        self.log_sigmas = nn.Parameter(torch.zeros(n_rules, in_dim))

        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.Tanh(),
        )
        self.rule_affine = nn.Linear(bottleneck_dim + in_dim + 1, n_rules * 3)

    def antecedent(self, x):
        # x: [N, in_dim]
        diff = x[:, None, :] - self.centers[None, :, :]
        sigma2 = torch.exp(self.log_sigmas)[None, :, :] ** 2 + 1e-6
        score = -0.5 * (diff**2 / sigma2).sum(dim=-1)
        mu = torch.softmax(score, dim=-1)
        if self.training and self.drop_rule > 0:
            keep = torch.rand_like(mu) > self.drop_rule
            mu = mu * keep
            mu = mu / (mu.sum(dim=-1, keepdim=True) + 1e-6)
        return mu

    def forward(self, x, h):
        z = self.bottleneck(h)
        bias = torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype)
        features = torch.cat([z, x, bias], dim=-1)
        rule_logits = self.rule_affine(features).view(x.size(0), self.n_rules, 3)
        mu = self.antecedent(x).unsqueeze(-1)
        logits = (mu * rule_logits).sum(dim=1)
        weights = torch.softmax(logits, dim=-1)
        return weights, mu.squeeze(-1)
