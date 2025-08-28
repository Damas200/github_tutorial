import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# SAN (Aime & Damas)
# ==================

class SAN(nn.Module):
    def __init__(self, in_channels, out_channels, J=3, J_h=5, dropout=0.5):
        super().__init__()
        self.J = J
        self.J_h = J_h
        self.out_channels = out_channels
        self.W_d = nn.Parameter(torch.randn(J, in_channels, out_channels))
        self.W_u = nn.Parameter(torch.randn(J, in_channels, out_channels))
        self.W_h = nn.Parameter(torch.randn(in_channels, out_channels))
        self.W_b1 = nn.Parameter(torch.randn(J, in_channels, out_channels))
        self.W_b2 = nn.Parameter(torch.randn(J, in_channels, out_channels))
        self.dropout = dropout
        # Fix: Linear layers for attention now correctly take (2 * in_channels) as input size
        self.attn_d = nn.Linear(in_channels * 2, 1)  # Now correctly expecting (2 * in_channels)
        self.attn_u = nn.Linear(in_channels * 2, 1)  # Now correctly expecting (2 * in_channels)

    def harmonic_filter(self, L, x):
        I = torch.eye(L.shape[0], device=x.device)
        eigenvalues = torch.linalg.eigvalsh(L)
        lambda_max = eigenvalues[-1].item()
        epsilon = 0.5 * (2 / lambda_max) if lambda_max > 0 else 0.1
        P = I - epsilon * L
        for _ in range(self.J_h - 1):
            P = P @ P
        return P @ x @ self.W_h

    def compute_attention(self, x, L, attn_fn):
        idx = torch.nonzero(L, as_tuple=True)
        src, dst = idx[0], idx[1]
        
        # Ensure indices are within bounds of the tensor size
        max_index = x.size(0) - 1  # Assuming 0th dimension of x is what we're concerned about
        src = torch.clamp(src, 0, max_index)
        dst = torch.clamp(dst, 0, max_index)
        
        batch_size = x.size(1)  # x shape: (n_edges, batch_size)
        x_src = x[src]  # (n_nonzero, batch_size)
        x_dst = x[dst]  # (n_nonzero, batch_size)
        
        attn_input = torch.cat([x_src, x_dst], dim=-1)  # (n_nonzero, batch_size, 2*in_channels)
        
        # Reshape the input for the linear layer
        attn_input = attn_input.view(-1, attn_input.size(-1))  # (n_nonzero * batch_size, 2 * in_channels)
        
        # Compute attention scores
        attn_scores = attn_fn(attn_input).sigmoid()  # (n_nonzero * batch_size, 1)
        
        # Reshape the attention scores
        attn_scores = attn_scores.view(-1, batch_size, 1)  # (n_nonzero, batch_size, 1)
        
        # Initialize attention matrix with zeros
        L_attn = torch.zeros(L.shape[0], L.shape[1], batch_size, device=x.device)
        L_attn[src, dst] = attn_scores.squeeze(-1)
        L_attn[dst, src] = attn_scores.squeeze(-1)
        
        return L_attn

    def forward(self, z0, z1, z2, L0, L1_d, L1_u, L2, B1, B2):
        # Attention-weighted Laplacians
        L1_d_attn = self.compute_attention(z1, L1_d, self.attn_d)
        L1_u_attn = self.compute_attention(z1, L1_u, self.attn_u)

        # Node signal (Z_0)
        z0_out = torch.zeros(z0.size(0), self.out_channels, device=z0.device)
        for j in range(self.J):
            z0_out += (L0 @ z0) @ self.W_d[j]
            z0_out += (B1 @ z1) @ self.W_b1[j]
        z0_out += self.harmonic_filter(L0, z0)
        z0_out = F.dropout(F.relu(z0_out), p=self.dropout, training=self.training)

        # Edge signal (Z_1)
        z1_out = torch.zeros(z1.size(0), self.out_channels, device=z1.device)
        for j in range(self.J):
            z1_out += (L1_d_attn @ z1) @ self.W_d[j]
            z1_out += (L1_u_attn @ z1) @ self.W_u[j]
            z1_out += (B1.t() @ z0) @ self.W_b1[j]
            z1_out += (B2 @ z2) @ self.W_b2[j]
        z1_out += self.harmonic_filter(L1_d + L1_u, z1)
        z1_out = F.dropout(F.relu(z1_out), p=self.dropout, training=self.training)

        # Triangle signal (Z_2)
        z2_out = torch.zeros(z2.size(0), self.out_channels, device=z2.device)
        for j in range(self.J):
            z2_out += (L2 @ z2) @ self.W_u[j]
            z2_out += (B2.t() @ z1) @ self.W_b2[j]
        z2_out += self.harmonic_filter(L2, z2)
        z2_out = F.dropout(F.relu(z2_out), p=self.dropout, training=self.training)

        return z0_out, z1_out, z2_out

###########################################################################


#  Combine SARNN (Aime & Damas)
# =======================================

#define SARNN
class SAN_RNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=1, J=3, J_h=5, dropout=0.5):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.san = SAN(in_channels=in_channels, out_channels=hidden_channels, J=J, J_h=J_h, dropout=dropout)
        self.rnn = nn.RNN(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            nonlinearity='tanh',
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
        self.to(self.device)

    def forward(self, sample):
        z0, z1, z2, L0, L1_d, L1_u, L2, B1, B2 = sample
        batch_size = z0.size(0)
        seq_length = z0.size(1)
        out = []
        for t in range(seq_length):
            z0_t = z0[:, t, :]
            z1_t = z1[:, t, :]
            z2_t = z2[:, t, :]
            _, z1_out, _ = self.san(z0_t, z1_t, z2_t, L0, L1_d, L1_u, L2, B1, B2)
            out.append(z1_out)
        out = torch.stack(out, dim=1)  # (batch_size, seq_length, hidden_channels)
        out, _ = self.rnn(out)
        out = self.mlp(out[:, -1, :])  # Take last time step
        return out
