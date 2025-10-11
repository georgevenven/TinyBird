import torch, math
from torch import nn

import torch
import torch.nn as nn
import math


class TinyBird(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.patch_size = config["patch_size"]
        self.max_seq = config["max_seq"]
        self.mask_p = config["mask_p"]

        self.patch_projection = nn.Conv2d(
            in_channels=1, out_channels=config["enc_hidden_d"], kernel_size=self.patch_size, stride=self.patch_size
        )

        self.encoder_transformer_block = nn.TransformerEncoderLayer(
            d_model=config["enc_hidden_d"],
            nhead=config["enc_n_head"],
            dim_feedforward=config["enc_dim_ff"],
            dropout=config["dropout"],
            batch_first=True,
            norm_first=True,
        )

        self.decoder_transformer_block = nn.TransformerEncoderLayer(
            d_model=config["dec_hidden_d"],
            nhead=config["dec_n_head"],
            dim_feedforward=config["dec_dim_ff"],
            dropout=config["dropout"],
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(self.encoder_transformer_block, num_layers=config["enc_n_layer"])
        self.decoder = nn.TransformerEncoder(self.decoder_transformer_block, num_layers=config["dec_n_layer"])

        self.encoder_to_decoder = nn.Linear(config["enc_hidden_d"], config["dec_hidden_d"])
        self.decoder_to_pixel = nn.Linear(config["dec_hidden_d"], self.patch_size[0] * self.patch_size[1])

        self.sep_param = nn.Parameter(torch.zeros(1, 1, 1))
        # NOTE: expansion needs to occur at run time to be on the correct device
        # self.sep_token  = self.sep_param.expand(1, config["mels"],1)

        self.pad_token = nn.Parameter(torch.zeros(1, 1, config["dec_hidden_d"]))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config["dec_hidden_d"]))

        self.pos_enc = nn.Parameter(torch.zeros(1, config["max_seq"], config["enc_hidden_d"]))

        # blockwise mask width in **tokens** along W' (time)
        self.mask_block_w = config.get("mask_block_w", 32)

        self.init_weights()

    def init_weights(self):
        # randomly initialize the weights with a chosen seed
        torch.manual_seed(42)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def tokenize_spectrogram(self, x):
        # x  (B, C=1, H , W)
        z = self.patch_projection(x)  # (B, D_enc, H', W')
        H, W = z.shape[-2], z.shape[-1]
        return z.flatten(2, 3).transpose(1, 2), H, W  # (B, T, D_enc)

    def apply_position_encoding(self, z: torch.Tensor):
        # z: (B, T, D_enc)

        B, T, D = z.shape
        assert T <= self.max_seq, f"T={z.size(1)} exceeds max_seq={self.max_seq}"

        return z + self.pos_enc[:, :T, :]  # (B, T, D_enc)

    def compactify_data(self, x: torch.Tensor, xi: torch.Tensor, N: torch.Tensor):
        """
        Remove silence from spectrograms keeping chirp blocks.

        Args:
            x: Spectrograms (B, C, H, W)
            xi: Chirp boundaries (B, N_max, 2) [start, end)
            N: Valid chirp counts per item (B,)

        Returns:
            x_new: Compacted spectrograms (B, C, H, w_max)
            xi_new: Updated boundaries in compressed coordinates (B, N_max, 2)
        """
        # x:  (B, C, H, W) #spectrogram
        # xi: (B, W, 2)    #[start, end) of slices,  N slicesper item
        # N:  (B, )        #number of valid slices per item

        B, C, H, W = x.shape

        N = N.view(B, 1)
        N_max = int(N.max())

        xi = xi[:, :N_max, :]
        xi_new = xi.clone()

        w = (xi[:, :, 1] - xi[:, :, 0]).clamp(min=0)
        idx_row = torch.arange(N_max, device=x.device).unsqueeze(0).expand(B, -1)
        w_valid = w * (idx_row < N)
        csum = torch.cumsum(w_valid, dim=1)
        seps_before = idx_row
        xi_new[:, :, 0] = (csum - w_valid + seps_before).to(xi.dtype)
        xi_new[:, :, 1] = (csum + seps_before).to(xi.dtype)
        xi_new[(idx_row >= N)] = 0
        w_max = w_valid.sum(dim=1) + (N.squeeze(1).clamp(min=1) - 1)

        # xi_new: (B, N_max, 2)  #[start, end) of slices with silence removed,  N slices per item
        # w_max: (B, )           #width of each item

        x_new = torch.zeros_like(x)  # ensures columns beyond valid region are zero

        # Expand the learnable separator scalar to a column on the **current** device/dtype
        sep_col_vec = self.sep_param.expand(1, x.size(2), 1)[0, :, 0]
        sep_col_vec = sep_col_vec.to(device=x.device, dtype=x.dtype)  # (H,) matches x_new[b, 0, :, idx]

        for b in range(B):
            n_valid = int(N[b, 0].item())
            for i in range(n_valid):
                s0, e0 = int(xi[b, i, 0].item()), int(xi[b, i, 1].item())
                sn, en = int(xi_new[b, i, 0].item()), int(xi_new[b, i, 1].item())

                if en > sn and e0 > s0:
                    x_new[b, 0, :, sn:en] = x[b, 0, :, s0:e0]

                if i < n_valid - 1 and en < W:
                    x_new[b, 0, :, en] = sep_col_vec

        x_new = x_new[:, :, :, : int(w_max.max().item())]
        # x_new:  (B, C, H, w_max) #spectrogram, with silence removed

        return x_new, xi_new

    def sample_data(self, x: torch.Tensor, xi: torch.Tensor, N: torch.Tensor, n_blocks: int, start: int = -1):
        """
        Sample random contiguous windows of chirp blocks.

        Args:
            x: Spectrograms (B, C, H, W)
            xi: Chirp boundaries (B, N_max, 2)
            N: Valid chirp counts per item (B,)
            n_blocks: Number of adjacent blocks to sample
            start: Start index for sampling (or -1 for random) [ensures n_blocks contiguous blocks]

        Returns:
            x_out: Windowed spectrograms (B, C, H, max_width)
            xi_out: Remapped boundaries in window coordinates (B, n_blocks, 2)
        """
        B, C, H, W = x.shape
        device = x.device
        N = N.view(B, 1)

        n_blocks = int(min(n_blocks, N.min().item()))  # ensure feasible for all items

        # Per item in batch start index in [0, N[b]-n_blocks]
        if start >= 0:
            start = torch.full((B,), int(start), device=device, dtype=torch.long)
        else:
            start_max = (N.squeeze(1) - n_blocks + 1).clamp_min(1)
            start = (torch.rand(B, device=device) * start_max.float()).floor().to(device=device, dtype=torch.long)
        end = start + n_blocks  # (B,)

        b_ix = torch.arange(B, device=device)

        # Raw window bounds in ORIGINAL x-coords: start at first slice's start, end at last slice's end
        st_raw = xi[b_ix, start, 0].to(device=device, dtype=torch.long)  # (B,)
        en_raw = xi[b_ix, end - 1, 1].to(device=device, dtype=torch.long)  # (B,) end is exclusive

        # Per-item widths and common batch width
        widths = (en_raw - st_raw).clamp_min(1)  # (B,)
        max_width = int(widths.max().item())

        x_out = torch.zeros(B, C, H, max_width, device=device, dtype=x.dtype)
        xi_out = torch.zeros(B, n_blocks, 2, device=device, dtype=xi.dtype)

        for b in range(B):
            en = int(en_raw[b].item())
            st = max(0, en - max_width)
            en = st + max_width  # right-align to end

            x_out[b, :, :, :] = x[b, :, :, st:en]  # crop the spectrogram to the new window

            xi_out[b] = xi[b, start[b] : end[b], :].clone()  # remap the boundaries to the new window
            xi_out[b, :, 0] = xi_out[b, :, 0] - st
            xi_out[b, :, 1] = xi_out[b, :, 1] - st

        return x_out, xi_out

    def build_column_mask(
        self,
        xi: torch.Tensor,
        hw: (None, None),
        masked_blocks: int = -1,
        frac: float = -1.0,
        mblock: list = [],
        iblock: list = [],
    ):
        """
        Generate column-wise masking pattern for spectrogram patches.

        Args:
            xi: Chirp boundaries (B, N, 2)
            hw: Spatial dimensions (H, W)
            masked_blocks: Number of chirp blocks to mask (or 0)
            frac: Fraction of columns to mask 0.0-1.0 (or 0.0)
            mblock: Index of chirp block to mask (or -1) (if specified, sets masked_blocks and frac to 1 and 0)
            iblock: Index of chirp block to isolate (or -1) (mblock and iblock are exclusive, and mblock must be set if iblock is set)
        Returns:
            Boolean mask (B, H*W) where True = masked patches

        Note: Exactly one of masked_blocks or frac must be > 0
        """
        assert xi.dim() == 3 and xi.size(2) == 2, f"xi must be (B, N, 2), got {tuple(xi.shape)}"
        H, W = hw
        device = xi.device
        B, N, _ = xi.shape

        if len(mblock) > 0:
            assert all((0 <= i < N) for i in mblock), f"invalid mblock indices N={N}, got {mblock}"
            masked_blocks, frac = 0, 0.0  # disable masked_blocks and frac behavior if mblock is set
        else:
            # if mblock is not set, ensure masked_blocks and frac are valid
            assert (masked_blocks > 0) ^ (
                frac > 0
            ), f"either masked_blocks or frac must be greater than 0 not both, got {masked_blocks} and {frac}"
            assert masked_blocks < N, f"masked_blocks must be less than N, got {masked_blocks} and {N}"
            assert frac < 1, f"frac must be between 0 and 1, got {frac}"

        if len(iblock) > 0:
            # if iblock is set, ensure mblock is set
            assert all((0 <= i < N) for i in iblock), f"invalid iblock indices N={N}, got {iblock}"
            assert len(mblock) > 0, f"mblock len={len(mblock)}. mblock must be set if iblock is set"

        starts = xi[:, :, 0].to(torch.long).clamp(min=0, max=W)  # (B, N)
        ends = xi[:, :, 1].to(torch.long).clamp(min=0, max=W)  # (B, N)
        widths = (ends - starts).clamp(min=0)  # (B, N)

        # get n_block random blocks between 0 and N ensure there are no duplicates
        mask_blocks = []
        if len(mblock) > 0:
            mask_blocks = mblock
            m_w = max([int(widths[b, mask_blocks].sum().item()) for b in range(B)])  # max width of the blocks
        elif masked_blocks > 0:
            mask_blocks = torch.randperm(N, device=device)[:masked_blocks].tolist()  # randomly select n_blocks blocks
            m_w = max([int(widths[b, mask_blocks].sum().item()) for b in range(B)])  # max width of the blocks
        else:  # frac > 0
            m_w = max(0, min(W - 1, int(round(float(frac) * W))))  # width of the mask, no block selected

        pad2d = torch.zeros(B, W, dtype=torch.bool, device=device)  # pad2d is boolean padding of the spectrogram
        mask2d = torch.zeros(B, W, dtype=torch.bool, device=device)  # mask2d is a boolean mask of the spectrogram
        for b in range(B):
            st_i = [int(v) for v in starts[b].tolist()]
            end_i = [int(v) for v in ends[b].tolist()]

            # ensure that "remaining" can't include the isolated blocks, as this is where information comes from.
            if len(iblock) > 0:
                for ib in iblock:
                    pad2d[b, st_i[ib] : end_i[ib]] = True  # this will be reset to False later

            # Specified mask block will not be padded, it will be masked even if iblock was set
            for blk in mask_blocks:
                mask2d[b, st_i[blk] : end_i[blk]] = True

            # randomly select remaining columns to keep mask width constant for each item in the batch
            remaining = ((~mask2d[b]) & (~pad2d[b])).nonzero(as_tuple=False).squeeze(1)
            remaining = remaining[torch.randperm(remaining.numel(), device=device)[: m_w - int(mask2d[b].sum().item())]]
            mask2d[b, remaining] = True

            # if iblock is set, ensure iblock is not padded and that the pad does not ovelap with the mask
            if len(iblock) > 0:
                # pad2d should be True everywhere except the isolated block and where the mask is true
                pad2d[b, :] = True  # start fully padded
                for ib in iblock:
                    pad2d[b, st_i[ib] : end_i[ib]] = False  # do not pad isolated blocks
            else:
                # iblock is not set, pad the parital blocks
                pad2d[b, 0 : min(st_i)] = True  # pad partial blocks if iblock is not set
                pad2d[b, max(end_i) : W] = True  # pad partial blocks if iblock is not set

            pad2d[b, mask2d[b]] = False  # any masked columns will not be padded

        pad2d = (
            pad2d.unsqueeze(1).expand(-1, H, -1).flatten(1, 2).to(device=device, dtype=torch.bool)
        )  # (B,W) -> (B, H, W) -> (B, H*W)
        mask2d = (
            mask2d.unsqueeze(1).expand(-1, H, -1).flatten(1, 2).to(device=device, dtype=torch.bool)
        )  # (B,W) -> (B, H, W) -> (B, H*W)
        assert not (pad2d & mask2d).any(), "pad2d and mask2d overlap (both True at some positions)"

        print("================================================")
        print(f"mblock={mblock}, iblock={iblock}, frac={frac}")
        print(f"pad2d[0,0,:]={pad2d[0,0,:]}")
        print(f"mask2d[0,0,:]={mask2d[0,0,:]}")
        print("================================================")

        return pad2d, mask2d

    def mask(self, z: torch.Tensor, bool_mask: torch.Tensor):
        """
        Apply column masking and reorder tokens for decoder.

        Args:
            z: Token embeddings (B, T, D)
            bool_mask: Boolean mask (B, T) where True = masked

        Returns:
            z_keep: Kept tokens (B, keep_count, D)
            idx_restore: Permutation indices to restore original order (B, T)
        """

        B, T, D = z.shape
        B, T = bool_mask.shape

        # key = mask_flag * (T+1) + position  → keeps original order within kept/masked groups
        pos = torch.arange(T, device=z.device).unsqueeze(0).expand(B, T)  # (B,T)
        key = bool_mask.to(torch.int64) * (T + 1) + pos  # (B,T)
        order = key.argsort(dim=1)  # (B,T), kept-first, masked second

        # --- gather kept-first tokens; slice to max keep (rectangular) ---
        z_perm = torch.gather(z, 1, order.unsqueeze(-1).expand(B, T, D))  # (B,T,D)
        keep_count = int((~bool_mask[0, :]).sum().item())
        z_keep = z_perm[:, :keep_count, :].contiguous()  # (B,max_keep,D)

        # --- inverse permutation to restore original order ---
        idx_restore = order.argsort(dim=1)  # (B,T)

        return z_keep, idx_restore

    def forward_encoder(self, x: torch.Tensor, xi: torch.Tensor, **column_mask_args):
        """
        Patchify → add positional encodings → build a **column-wise** mask from chirp boundaries →
        keep only unmasked tokens → encode with the Transformer encoder.

        Args:
            x:  (B, 1, H, W) spectrograms after any upstream compaction/windowing.
            xi: (B, N, 2) chirp [start, end) boundaries **in the current W frame**.

            column_mask_args:
            masked_blocks: Number of chirp blocks to randomly mask (or 0)
            frac: Fraction of columns to mask 0.0-1.0 (or 0.0) [ masked_blocks and frac are exclusive]
            mblock: Index of chirp block to mask (or -1) (if specified, sets masked_blocks to 1 and frac to 0)
            iblock: Index of chirp block to isolate (or -1) (mblock and iblock are exclusive, and mblock must be set if iblock is set)

        Returns:
            h:           (B, keep, D_enc) encoder outputs for **kept** (unmasked) tokens only.
            idx_restore: (B, T) permutation that restores the original token order after we
                         temporarily move all kept tokens before masked ones. Use this to place
                         decoder mask tokens back into the sequence.
            bool_mask:   (B, T) boolean mask over tokens (True = **masked** token to be predicted).
            T:           int total tokens in this sample = H * W (after patching), i.e. sequence length.

        Notes:
            • The mask is constructed **per column** across all mel rows: a time column that is masked
              for a given item is masked for every row in that column. This yields a mask of shape (B, H*W).
            • The encoder never sees masked tokens. It processes only the kept tokens; the decoder later
              reconstructs the full sequence by inserting learned mask tokens and unshuffling with
              `idx_restore`.
        """
        # x:  (B, C=1, H, W)
        # xi: (B, N, 2)

        z, H, W = self.tokenize_spectrogram(x)  # (B, T, D_enc), H, W
        z = self.apply_position_encoding(z)  # (B, T, D_enc)

        bool_pad, bool_mask = self.build_column_mask(xi, hw=(H, W), **column_mask_args)
        # bool_pad  : (B, H*W), True = padded (column-wise across H)
        # bool_mask : (B, H*W), True = masked (column-wise across H) pad_mask & bool_mask are exclusive

        z_keep, idx_restore = self.mask(
            z, (bool_pad | bool_mask)
        )  # ensure the encoder never sees padded or masked tokens
        # z_keep: Kept tokens (B, keep_count, D)
        # idx_restore: Permutation indices to restore original order (B, T)

        h = self.encoder(z_keep)  # (B, keep, D_enc)
        return h, idx_restore, bool_mask, bool_pad, H * W  # (B, keep, D_enc), (B, T), (B, T), (B,T), T

    def forward_encoder_inference(self, x: torch.Tensor):
        """
        Patchify → add pos enc → mask → Transformer encoder.
        Returns:
        h: (B, T, D_enc)
        """
        # x:  (B, C=1, H, W)

        z, H, W = self.tokenize_spectrogram(x)  # (B, T, D_enc)
        z = self.apply_position_encoding(z)  # (B, T, D_enc)
        return self.encoder(z)  # (B, T, D_enc)

    def forward_decoder(
        self,
        h: torch.Tensor,
        idx_restore: torch.Tensor,
        T: int,
        bool_pad: torch.Tensor = None,
        attend_to_padded: bool = True,
    ):
        """
        Project encoder outputs to decoder width → insert learned mask tokens → unshuffle back to the
        original token order → add decoder positional encodings → run the Transformer decoder → predict
        pixel values for each patch.

        Args:
            h:           (B, T_enc, D_enc) encoder outputs for **kept** tokens only (T_enc ≤ T).
            idx_restore: (B, T) permutation indices that map the kept-first layout back to the original
                         token order (kept ∪ masked). Produced by `mask(...)` in the encoder path.
            T:           int total number of tokens in the **full** sequence (kept + masked) after patching.
            bool_pad:    (B, T) boolean mask of padded tokens.

        Returns:
            pred:        (B, T, P) per-token predictions, where P = patch_height * patch_width.

        Decoding logic:
            1) A linear `encoder_to_decoder` projects encoder width D_enc → D_dec so encoder tokens match
               the decoder’s channel size.
            2) We create learned `mask_token`s of shape (B, T - T_enc, D_dec) to stand in for the masked
               tokens that the encoder did not process.
            3) Concatenate `[projected_kept, mask_tokens]` (kept-first layout) and apply `idx_restore`
               to recover the original interleaving of kept+masked tokens.
            4) Add decoder positional embeddings by projecting the encoder’s positional encodings to D_dec
               (this keeps encoder/decoder positions consistent without maintaining a second embedding table).
            5) Run the decoder stack and map to pixel space with `decoder_to_pixel` to obtain P values per token.

        Important:
            • `idx_restore` must be computed from the **same mask** used in the encoder step; otherwise the
              unshuffle will place tokens incorrectly and reconstruction quality will collapse.
            • The model learns to predict **only** masked regions during training (via `loss_mse` indexing),
              but `pred` is returned for all tokens for convenience when depatchifying/visualizing.
        """
        B, T_enc, _ = h.shape

        # 1) Match channel sizes: project encoder tokens (D_enc) to decoder width (D_dec).
        y = self.encoder_to_decoder(h)  # (B, T_enc, D_dec)
        D_dec = self.decoder_to_pixel.in_features

        # 2) Insert learned mask tokens for the missing (masked) positions, then unshuffle back to original order.
        #    Learned mask token (defined in __init__) provides a consistent placeholder embedding for masked slots.
        mask_tokens = self.mask_token.expand(B, T - T_enc, D_dec)  # (B, T-keep, D_dec)
        y_full = torch.cat([y, mask_tokens], dim=1)  # kept-first layout
        y_full = torch.gather(y_full, 1, idx_restore.unsqueeze(-1).expand(B, T, D_dec))

        # 3) Replace padded positions with pad token.
        if bool_pad is not None:
            y_full = torch.where(bool_pad.unsqueeze(-1), self.pad_token.expand(B, T, D_dec), y_full)

        pos_dec = self.encoder_to_decoder(
            self.pos_enc[:, :T, :]
        )  # (1, T, D_dec) decoder pos-encs derived by projecting encoder pos-encs
        y_full = y_full + pos_dec

        if attend_to_padded:
            d = self.decoder(y_full)  # (B, T, D_dec)
        else:
            d = self.decoder(y_full, src_key_padding_mask=bool_pad)
        pred = self.decoder_to_pixel(d)  # Final per-token patch prediction: (B, T, P). P is pixels per patch
        return pred

    def loss_mse(self, x: torch.Tensor, pred: torch.Tensor, bool_mask: torch.Tensor):
        """
        Compute MSE on masked patches only.
        x:    (B, 1, H, W)
        pred: (B, T, P)
        """
        # Ensure bool_mask is on the same device as pred for indexing during backward
        bool_mask = bool_mask.to(pred.device)
        unfold = nn.Unfold(
            kernel_size=self.patch_size, stride=self.patch_size
        )  # unfolds spectrogram into patches of size P
        target = (
            unfold(x).transpose(1, 2).to(device=pred.device, dtype=pred.dtype)
        )  # (B, T, P), where T = num patches, P = patch_height * patch_width

        # Normalize target patches
        # target_mean = target.mean(dim=-1, keepdim=True)  # (B, T, 1), mean per patch
        # target_std = target.std(dim=-1, keepdim=True)    # (B, T, 1), std per patch
        # target = (target - target_mean) / (target_std + 1e-6)  # normalized target patches, shape (B, T, P)

        # # Numerically stable per-patch normalization in FP32 to avoid NaNs under AMP/FP16
        t32 = target.float()  # (B, T, P) in FP32
        mean32 = t32.mean(dim=-1, keepdim=True)  # (B, T, 1)
        # Use variance + eps, unbiased=False for stability
        var32 = t32.var(dim=-1, keepdim=True, unbiased=False)  # (B, T, 1)
        eps = 1e-5
        std32 = torch.sqrt(torch.clamp(var32, min=0.0) + eps)  # (B, T, 1)
        norm32 = (t32 - mean32) / std32  # (B, T, P)
        # Replace any residual NaN/Inf (e.g., completely constant or empty patches)
        norm32 = torch.nan_to_num(norm32, nan=0.0, posinf=0.0, neginf=0.0)  # (B, T, P)
        target = norm32.to(dtype=pred.dtype)  # (B, T, P) match pred dtype

        # loss = ((pred - target) ** 2)[bool_mask].mean()  # compute MSE only on masked patches; pred is (B, T, P), bool_mask is (B, T)

        # MSE per token, then masked mean across tokens
        per_pixel = (pred - target).pow(2)  # (B, T, P)
        per_token = per_pixel.mean(dim=-1)

        # Defensive guard: detect NaNs/Inf in per_token
        if torch.isnan(per_token).any() or torch.isinf(per_token).any():
            # Print one-time diagnostics to help trace bad values
            if not hasattr(self, "_loss_nan_warned"):
                print("[loss_mse] WARNING: NaN/Inf detected in per_token. Dumping stats once.")
                print("  target stats:", float(t32.min()), float(t32.max()), float(var32.mean()))
                self._loss_nan_warned = True
            per_token = torch.nan_to_num(per_token, nan=0.0, posinf=0.0, neginf=0.0)

            # (B, T)
        masked_sum = (per_token * bool_mask.float()).sum()  # scalar
        masked_count = bool_mask.sum().clamp_min(1).to(per_token.dtype)  # scalar
        loss = masked_sum / masked_count  # scalar

        return loss


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader

    # pull dataset class from your existing file
    from data_loader import SpectogramDataset

    parser = argparse.ArgumentParser(description="TinyBird test")
    parser.add_argument("--input_dir", required=True, type=Path, help="input directory")
    args = parser.parse_args()

    # create config with default values from the main method in pretrain.py
    config = {
        # training meta (not used here but included for parity/completeness)
        "steps": 500_000,
        "lr": 2e-4,
        "batch_size": 256,
        "dropout": 0.1,
        "mask_p": 0.75,
        "eval_every": 500,
        "amp": False,
        "weight_decay": 0.0,
        # data geometry
        "mels": 128,
        "num_timebins": 30001,
        "patch_height": 32,
        "patch_width": 1,
        "max_seq": 30001,
        "patch_size": (32, 1),
        # encoder
        "enc_hidden_d": 12,
        "enc_n_head": 1,
        "enc_n_layer": 1,
        "enc_dim_ff": 12,
        # decoder
        "dec_hidden_d": 12,
        "dec_n_head": 1,
        "dec_n_layer": 1,
        "dec_dim_ff": 12,
        # extra used by TinyBird
        "mask_block_w": 32,  # present in model, default from your code
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[TinyBird test] device={device}, mels={config["mels"]}, timebins={config["num_timebins"]}, "
        f"patch={config['patch_size']}, max_seq={config['max_seq']}"
    )

    dataset = SpectogramDataset(dir=str(args.input_dir), n_mels=config["mels"], n_timebins=config["num_timebins"])

    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)
    model = TinyBird(config).to(device)
    model.eval()
    batch = next(iter(loader))  # (spec, chirp_intervals, N, filename)
    spectrograms, chirp_intervals, N, filenames = batch

    x = spectrograms.to(device, non_blocking=False)  # (B=2, 1, H=mels, W=time)
    xi = chirp_intervals.to(device, non_blocking=False)  # (B=2, N_max, 2)
    N = N.to(device, non_blocking=False)  # (B=2, 1)

    print(f"[TinyBird test] batch shapes: x={tuple(x.shape)}, xi={tuple(xi.shape)}, N={tuple(N.shape)}")
    print(f"[TinyBird test] files: {filenames[0]} | {filenames[1]}")

    # --- 3) encoder ---
    with torch.no_grad():
        print(f"[TinyBird test] before prepare_data: x={tuple(x.shape)}, xi={tuple(xi.shape)}, N={tuple(N.shape)}")
        x, xi = model.compactify_data(x, xi, N)
        print(f"[TinyBird test] after prepare_data: x={tuple(x.shape)}, xi={tuple(xi.shape)}")
        x, xi = model.sample_data(x, xi, N, n_blocks=3)
        print(f"[TinyBird test] after window_data: x={tuple(x.shape)}, xi={tuple(xi.shape)}")

        h, idx_restore, bool_mask, bool_pad, T = model.forward_encoder(x, xi)

    print(
        f"[TinyBird test] encoder outputs:"
        f" h={tuple(h.shape)}, idx_restore={tuple(idx_restore.shape)},"
        f" bool_mask={tuple(bool_mask.shape)}, T={T}"
    )

    # --- 4) decoder on same batch ---
    with torch.no_grad():
        pred = model.forward_decoder(h, idx_restore, T)  # (B, T, P)

    print(f"[TinyBird test] decoder output: pred={tuple(pred.shape)}")

    # (optional) compute MSE on masked patches, like training loop
    with torch.no_grad():
        loss = model.loss_mse(x, pred, bool_mask).item()
    print(f"[TinyBird test] masked-patch MSE loss: {loss:.6f}")

    print("[TinyBird test] ✅ success")
