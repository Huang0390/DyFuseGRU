"""DyFuseGRU architecture extracted from model_kegong.py."""
import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    bf16-friendly RMSNorm. Keep the normalization math in fp32, then cast back.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return h.to(dtype) * self.weight


class GatedFusion(nn.Module):
    """
    Three-source fusion with source attention, reliability calibration and
    interaction residuals.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.source_embed = nn.Parameter(torch.zeros(1, 1, 3, dim))
        self.source_norm = RMSNorm(dim)

        self.score_proj = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        self.context_gate = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, 3),
        )
        self.reliability = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, 3),
            nn.Sigmoid(),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.interaction = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.out_norm = RMSNorm(dim)
        self.out_proj = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, earthquake: torch.Tensor, tokens: torch.Tensor, append: torch.Tensor
    ) -> torch.Tensor:
        sources = torch.stack([earthquake, tokens, append], dim=2) + self.source_embed
        sources = self.source_norm(sources)

        combined = torch.cat([earthquake, tokens, append], dim=-1)
        source_scores = self.score_proj(sources).squeeze(-1)
        context_scores = self.context_gate(combined)
        reliability = self.reliability(combined).clamp_min(1e-4)

        tau = self.temperature.clamp(0.5, 5.0)
        logits = (source_scores + context_scores + reliability.log()) / tau
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)

        weighted = torch.sum(weights * sources, dim=2)
        residual = self.interaction(combined)
        fused = self.out_norm(weighted + residual)
        fused = fused + self.dropout(self.out_proj(fused))
        return fused


class GRUBlock(nn.Module):
    """
    GRU block over latent spatial positions produced by the floor projection.
    """

    def __init__(self, dim: int, dropout: float = 0.1, bidirectional: bool = False):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        gru_out_dim = dim * (2 if bidirectional else 1)
        self.gru_proj = (
            nn.Linear(gru_out_dim, dim) if gru_out_dim != dim else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.norm(x))
        x = x + self.dropout(self.gru_proj(h))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class DyFuseGRU(nn.Module):
    """
    Dynamic Fusion GRU for time-domain structural response prediction.

    The model performs three-source feature fusion first, then uses residual
    GRU blocks over latent spatial positions to capture inter-story dependencies.
    """

    def __init__(self, floor_num, time_len, pred_lenth, dim, n_layers, dropout=0.1):
        super().__init__()
        self.floor_num = floor_num
        self.time_len = time_len
        self.pred_lenth = pred_lenth
        self.dim = dim
        self.n_layers = n_layers

        self.tok_embeddings_earthquake = nn.Linear(time_len, dim)
        self.tok_embeddings_tokens = nn.Linear(time_len, dim)
        self.tok_embeddings_append = nn.Linear(time_len, dim)

        self.tok_embeddings_earthquake_floor = nn.Linear(floor_num, dim)
        self.tok_embeddings_tokens_floor = nn.Linear(floor_num, dim)
        self.tok_embeddings_append_floor = nn.Linear(floor_num, dim)

        self.dropout_earthquake = nn.Dropout(dropout)
        self.dropout_tokens = nn.Dropout(dropout)
        self.dropout_append = nn.Dropout(dropout)

        self.fusion_layer = GatedFusion(dim, dropout=dropout)

        self.gru_layers_fused = nn.ModuleList(
            [GRUBlock(dim, dropout=dropout) for _ in range(n_layers)]
        )

        self.norm_final = RMSNorm(dim)
        self.output_head_floor = nn.Linear(dim, floor_num, bias=False)
        self.output_head = nn.Linear(dim, pred_lenth, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(
        self, tokens: torch.Tensor, earthquake: torch.Tensor, append: torch.Tensor
    ) -> torch.Tensor:
        tokens = tokens.squeeze(dim=-1)
        earthquake = earthquake.squeeze(dim=-1)
        append = append.squeeze(dim=-1)

        h_earthquake = self.dropout_earthquake(
            self.tok_embeddings_earthquake(earthquake)
        )
        h_earthquake = self.tok_embeddings_earthquake_floor(
            h_earthquake.permute(0, 2, 1)
        ).permute(0, 2, 1)

        h_tokens = self.dropout_tokens(self.tok_embeddings_tokens(tokens))
        h_tokens = self.tok_embeddings_tokens_floor(h_tokens.permute(0, 2, 1)).permute(
            0, 2, 1
        )

        h_append = self.dropout_append(self.tok_embeddings_append(append))
        h_append = self.tok_embeddings_append_floor(h_append.permute(0, 2, 1)).permute(
            0, 2, 1
        )

        h_fused = self.fusion_layer(h_earthquake, h_tokens, h_append)

        for layer in self.gru_layers_fused:
            h_fused = layer(h_fused)

        h_final = self.norm_final(h_fused)
        logits = self.output_head(
            self.output_head_floor(h_final.permute(0, 2, 1)).permute(0, 2, 1)
        ).unsqueeze(dim=-1)
        return logits
