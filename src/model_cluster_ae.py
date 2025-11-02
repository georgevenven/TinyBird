"""Cluster-aware autoencoder model."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from registry_utils import GlobalPrototype


def _build_positional_encoding(max_len: int, dim: int) -> nn.Parameter:
    weight = torch.zeros(max_len, dim)
    nn.init.normal_(weight, mean=0.0, std=0.02)
    return nn.Parameter(weight)


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim: int, n_head: int, dim_ff: int, n_layer: int, dropout: float) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_head,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=attn_mask)


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim: int, n_head: int, dim_ff: int, n_layer: int, dropout: float) -> None:
        super().__init__()
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_head,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layer)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.decoder(x, src_key_padding_mask=attn_mask)


class ClusterAutoEncoder(nn.Module):
    def __init__(
        self,
        config: dict,
        prototypes_by_m: Dict[int, List[GlobalPrototype]],
        num_clusters: int,
        feature_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim

        conv_modules: List[nn.Conv2d] = []
        conv_out_channels = 0
        self.prototype_order: List[GlobalPrototype] = []
        for window, protolist in sorted(prototypes_by_m.items()):
            if not protolist:
                continue
            conv = nn.Conv2d(
                in_channels=1,
                out_channels=len(protolist),
                kernel_size=(feature_dim, window),
                padding=(0, window // 2),
                bias=False,
            )
            conv.weight.data.zero_()
            for idx, proto in enumerate(protolist):
                kernel = torch.zeros(feature_dim, window)
                dims = proto.dims
                exemplar = torch.from_numpy(proto.exemplar)
                kernel[dims, : proto.exemplar.shape[1]] = exemplar
                conv.weight.data[idx, 0] = kernel
                self.prototype_order.append(proto)
            conv_modules.append(conv)
            conv_out_channels += len(protolist)
        if not conv_modules:
            raise ValueError("No prototypes found to initialise the template bank.")
        self.template_convs = nn.ModuleList(conv_modules)
        self.template_out_channels = conv_out_channels

        enc_hidden = config.get("enc_hidden_d", 256)
        dropout = config.get("dropout", 0.1)

        self.temporal_proj = nn.Conv1d(self.template_out_channels, enc_hidden, kernel_size=1)

        self.max_seq = config.get("max_seq", 8192)
        self.pos_enc_enc = _build_positional_encoding(self.max_seq, enc_hidden)

        self.encoder = TransformerEncoder(
            hidden_dim=enc_hidden,
            n_head=config.get("enc_n_head", 8),
            dim_ff=config.get("enc_dim_ff", 1024),
            n_layer=config.get("enc_n_layer", 4),
            dropout=dropout,
        )

        dec_hidden = config.get("dec_hidden_d", enc_hidden)
        self.proj_to_decoder = nn.Linear(enc_hidden, dec_hidden)
        self.pos_enc_dec = _build_positional_encoding(self.max_seq, dec_hidden)

        self.decoder = TransformerDecoder(
            hidden_dim=dec_hidden,
            n_head=config.get("dec_n_head", 8),
            dim_ff=config.get("dec_dim_ff", 1024),
            n_layer=config.get("dec_n_layer", 4),
            dropout=dropout,
        )

        self.recon_head = nn.Linear(dec_hidden, feature_dim)
        self.cls_head = nn.Sequential(
            nn.Linear(dec_hidden, dec_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dec_hidden * 2, num_clusters),
        )

        self.lambda_recon = float(config.get("lambda_recon", 1.0))
        self.lambda_cls = float(config.get("lambda_cls", 1.0))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, 1, D, T)
            mask: (B, 1, 1, T) with 1 for valid steps
        Returns:
            recon: (B, 1, D, T)
            logits: (B, C)
            h: (B, hidden)
        """

        feats = []
        for conv in self.template_convs:
            feats.append(conv(x).squeeze(2))  # (B, C_i, T)
        template_out = torch.cat(feats, dim=1)  # (B, C_total, T)
        template_out = self.temporal_proj(template_out)  # (B, enc_hidden, T)
        template_out = template_out.transpose(1, 2)  # (B, T, enc_hidden)

        T = template_out.shape[1]
        if T > self.max_seq:
            template_out = template_out[:, : self.max_seq]
            mask = mask[..., : self.max_seq]
            T = self.max_seq

        pos_enc = self.pos_enc_enc[:T].unsqueeze(0)
        enc_input = template_out + pos_enc

        attn_mask = (mask.squeeze(1).squeeze(1) == 0)  # (B, T) bool
        enc_out = self.encoder(enc_input, attn_mask)

        valid = mask.squeeze(1).squeeze(1)  # (B, T)
        denom = torch.clamp(valid.sum(dim=1, keepdim=True), min=1.0)
        h = (enc_out * valid.unsqueeze(-1)).sum(dim=1) / denom  # (B, enc_hidden)

        dec_hidden = self.proj_to_decoder(h)  # (B, dec_hidden)
        dec_input = dec_hidden.unsqueeze(1).repeat(1, T, 1)
        dec_input = dec_input + self.pos_enc_dec[:T].unsqueeze(0)
        dec_out = self.decoder(dec_input, attn_mask)

        recon = self.recon_head(dec_out).transpose(1, 2)  # (B, D, T)
        recon = recon.unsqueeze(1)

        logits = self.cls_head(dec_hidden)

        return recon, logits, dec_hidden

    def compute_loss(
        self,
        recon: torch.Tensor,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """Compute reconstruction + classification loss."""

        if mask.shape != recon.shape:
            valid_mask = mask.expand_as(recon)
        else:
            valid_mask = mask
        mse = F.mse_loss(recon * valid_mask, target * valid_mask, reduction="sum")
        denom = torch.clamp(valid_mask.sum(), min=1.0)
        mse = mse / denom

        ce = F.cross_entropy(logits, labels)

        loss = self.lambda_recon * mse + self.lambda_cls * ce
        return {"loss": loss, "loss_recon": mse, "loss_cls": ce}
