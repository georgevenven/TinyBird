import torch
from torch import nn
import torch.nn.functional as F
import random


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
        self.decoder_to_label = nn.Linear(config["dec_hidden_d"], 2)

        # classifier head for label prediction
        # 0 = Left channel
        # 1 = Right channel

        self.label_enc = nn.Embedding(4, config["enc_hidden_d"])
        # 0 = Left channel
        # 1 = Right channel
        # 2 = separator
        # 3 = [LABEL_MASK]  # for masked columns
        self.sep_class_id  = 2
        self.mask_class_id = 3

        self.sep_param = nn.Parameter(torch.zeros(1, 1, 1))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config["dec_hidden_d"]))

        self.pos_enc = nn.Parameter(torch.zeros(1, config["max_seq"], config["enc_hidden_d"]))

        self.init_weights()

    def init_weights(self):
        # randomly initialize the weights with a chosen seed
        torch.manual_seed(42)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Freeze and zero out unwanted rows
        with torch.no_grad():
            self.label_enc.weight[ self.sep_class_id ].zero_()  # separator
            self.label_enc.weight[ self.mask_class_id].zero_()  # [LABEL_MASK]

        # Make sure gradients never change these rows
        self.label_enc.weight.register_hook(
            lambda grad: grad.clone().index_fill_(0, torch.tensor([self.sep_class_id, self.mask_class_id], device=grad.device), 0.0)
        )

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

    def apply_label_encoding(self, z: torch.Tensor, xl: torch.Tensor):
        # z: (B, T, D_enc)
        # xl: (B, T)
        B, T, D_enc = z.shape
        Txl         = xl.shape[1]

        assert T == Txl, f"Size of z and xl must be the same, got x size is {T} and xl size is {Txl}"

        return z + self.label_enc( xl )  # (B, T, D_enc)


    def randomize_label(self, x_l: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            return x_l  # no change

        out = x_l.clone()
        mask01 = (out == 0) | (out == 1)
        out[mask01] = 1 - out[mask01]  # flips 0<->1
        return out

    def compactify_data(self, x: torch.Tensor, xi: torch.Tensor, N: torch.Tensor, xl: torch.Tensor = None):
        """
        Remove silence from spectrograms keeping chirp blocks.

        Args:
            x: Spectrograms (B, C, H, W)
            xi: Chirp boundaries (B, N_max, 2) [start, end)
            N: Valid chirp counts per item (B,)
            xl: Left channel labels (B, N_max)
        Returns:
            x_new: Compacted spectrograms (B, C, H, w_max)
            xi_new: Updated boundaries in compressed coordinates (B, N_max, 2)
        """
        # x:  (B, C, H, W) #spectrogram
        # xi: (B, W, 2)    #[start, end) of slices,  N slicesper item
        # N:  (B, )        #number of valid slices per item
        # xl: (B, W)       #channel labels

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
        if xl is not None:
            xl_new = torch.zeros(B, w_max.max().item(), device=x.device, dtype=xl.dtype)

        # Expand the learnable separator scalar to a column on the **current** device/dtype
        sep_col_vec = self.sep_param.expand(1, x.size(2), 1)[0, :, 0]
        sep_col_vec = sep_col_vec.to(device=x.device, dtype=x.dtype)  # (H,) matches x_new[b, 0, :, idx]

        for b in range(B):
            n_valid = int(N[b, 0].item())
            for i in range(n_valid):
                s0, e0 = int(xi[b, i, 0].item()), int(xi[b, i, 1].item())
                sn, en = int(xi_new[b, i, 0].item()), int(xi_new[b, i, 1].item())

                if en > sn and e0 > s0:
                    x_new[b, :, :, sn:en] = x[b, :, :, s0:e0]
                    if xl_new is not None:
                        xl_new[b, sn:en] = xl[b, s0:e0]
                if i < n_valid - 1 and en < W:
                    x_new[b, :, :, en] = sep_col_vec
                    if xl_new is not None:
                        xl_new[b, en] = self.sep_class_id
        x_new = x_new[:, :, :, : int(w_max.max().item())]

        # x_new:  (B, C, H, w_max) #spectrogram, with silence removed
        if xl_new is not None:
            return x_new, xi_new, xl_new
        else:
            return x_new, xi_new

    def sample_data_seq_length(
        self,
        x: torch.Tensor,
        xi: torch.Tensor,
        N: torch.Tensor,
        seq_len: int,
        mblock: int = -1,
        xl: torch.Tensor = None,
    ):
        """
        Sample random or fixed contiguous window of data, with the last block masked.

        Args:
            x: Spectrograms (B, C, H, W)
            xi: Chirp boundaries (B, N_max, 2)
            N: Valid chirp counts per item (B,)
            seq_len: Number of adjacent columns to sample
            mblock: masked block index (or -1 for random)

        Returns:
            x_out: Windowed spectrograms (B, C, H, seq_len)
            xi_out: Remapped boundaries in window coordinates (B, n_blocks, 2)
        """

        B, C, H, W = x.shape
        device = x.device

        Nv = N.view(-1).long()  # (B,)
        ends = xi[:, :, 1].long()  # (B, N_max)

        # mask out invalid blocks beyond each item's N
        idx = torch.arange(ends.size(1), device=xi.device).unsqueeze(0)  # (1, N_max)
        valid_mask = idx < Nv.unsqueeze(1)  # (B, N_max)
        ends_valid = ends.masked_fill(~valid_mask, -1)

        # per-item max end, then batch-min cap
        max_end_per_item = ends_valid.max(dim=1).values  # (B,)
        seq_len = min(int(seq_len), int(max_end_per_item.min().item()))

        # --- choose masked block per item ---
        if mblock >= 0:
            # Ensure mblock is valid across batch and its end >= seq_len (so we can right-align window)
            n_min = int(N.min().item())
            assert 0 <= mblock < n_min, f"mblock index out of range for min(N)={n_min}: {mblock}"
            ends = xi[:, mblock, 1].to(dtype=torch.long)
            assert bool((ends >= seq_len).all().item()), (
                f"masked block end must be >= seq_len for all items; seq_len={seq_len},"
                f" min_end={int(ends.min().item())}"
            )
            mb_idx = torch.full((B,), int(mblock), device=device, dtype=torch.long)
        else:
            # Pick a (possibly different) valid block for each item with end >= seq_len
            mb_idx = torch.zeros(B, dtype=torch.long, device=device)
            for b in range(B):
                n_b = int(Nv[b].item())
                ends_b = ends[b, :n_b]
                candidates = torch.nonzero(ends_b >= seq_len, as_tuple=False).squeeze(1)
                assert candidates.numel() > 0, (
                    f"no valid mask block with end>=seq_len for item {b}; seq_len={seq_len},"
                    f" max_end={int(ends_b.max().item()) if n_b > 0 else -1}"
                )
                # choose one candidate uniformly at random
                r = int(torch.randint(low=0, high=candidates.numel(), size=(1,), device=device).item())
                mb_idx[b] = candidates[r]

        # --- window the spectrogram x_out to the desired seq_len. Ensure that the window ends on the last column of the masked block. ---
        x_out = torch.zeros(B, C, H, seq_len, device=device, dtype=x.dtype)
        if xl is not None:
            xl_out = torch.zeros(B, seq_len, device=device, dtype=xl.dtype)

        mb_start_idx = torch.zeros(B, dtype=torch.long, device=device)
        for b in range(B):
            end_col = int(xi[b, mb_idx[b], 1].item())
            assert end_col >= seq_len, f"end_col={end_col} must be >= seq_len={seq_len}"
            start_col = end_col - seq_len

            x_out[b, :, :, :] = x[b, :, :, start_col:end_col]
            if xl is not None:
                xl_out[b] = xl[b, start_col:end_col]

            blocks = torch.nonzero(xi[b, :, 0] >= start_col, as_tuple=False).squeeze(1)
            mb_start_idx[b] = blocks[0] if len(blocks) > 0 else 0

        n_blocks = (mb_idx - mb_start_idx).clamp(min=0).min().item() + 1

        xi_out = torch.zeros(B, n_blocks, 2, device=device, dtype=xi.dtype)
        for b in range(B):
            # start index so that we take exactly n_blocks ending at mb_idx[b] (inclusive)
            start_idx = int(mb_idx[b].item()) - (n_blocks - 1)
            # By construction, start_idx >= mb_start_idx[b], because n_blocks is the batch-min.
            xi_slice = xi[b, start_idx : int(mb_idx[b].item()) + 1, :].clone()
            # xi_slice has shape (n_blocks, 2)
            xi_out[b] = xi_slice
            offset = xi_out[b, -1, 1].item() - seq_len
            xi_out[b, :, :] = xi_out[b, :, :] - offset

        assert (xi_out[:, -1, 1] == seq_len).all().item(), (
            f"xi_out[b,:,1] should always be seq_len for all items in the batch. got {xi_out[:, :, 1]}"
        )

        if xl is not None:
            return x_out, xi_out, xl_out
        else:
            return x_out, xi_out

    def sample_data(
        self,
        x: torch.Tensor,
        xi: torch.Tensor,
        N: torch.Tensor,
        n_blocks: int,
        start: int = -1,
        xl: torch.Tensor = None,
    ):
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
        if xl is not None:
            xl_out = torch.zeros(B, max_width, device=device, dtype=xl.dtype)

        for b in range(B):
            en = int(en_raw[b].item())
            st = max(0, en - max_width)
            en = st + max_width  # right-align to end

            x_out[b, :, :, :] = x[b, :, :, st:en]  # crop the spectrogram to the new window
            if xl is not None:
                xl_out[b] = xl[b, st:en]

            xi_out[b] = xi[b, start[b] : end[b], :].clone()  # remap the boundaries to the new window
            xi_out[b, :, 0] = xi_out[b, :, 0] - st
            xi_out[b, :, 1] = xi_out[b, :, 1] - st

        if xl is not None:
            return x_out, xi_out, xl_out
        else:
            return x_out, xi_out

    def sample_data_indices(self, x: torch.Tensor, xi: torch.Tensor, N: torch.Tensor, indices, xl: torch.Tensor = None):
        """
        Sample a window of chirp blocks using explicit block indices (not necessarily contiguous).

        Args:
            x: Spectrograms (B, C, H, W)
            xi: Chirp boundaries (B, N_max, 2)
            N: Valid chirp counts per item (B,)
            indices: Iterable of integer block indices to extract, in the order to be concatenated
            xl: channel labels (B, N_max)

        Returns:
            x_out:  Concatenated spectrograms of the selected blocks with 1-col separators (B, C, H, max_width)
            xi_out: Remapped boundaries in the new window coordinates, one row per selected index (B, K, 2)
            is xl is passed in, returns xl_out: channel labels (B, max_width)
        """
        B, C, H, W = x.shape
        device = x.device
        N = N.view(B, 1)
        W = int(W)

        # Normalize to a Python list of ints and **preserve the exact input order**
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().tolist()
        else:
            indices = list(indices)
        indices = [int(i) for i in indices]
        assert len(indices) > 0, "indices must contain at least one element"

        # Keep only indices valid for *all* items in the batch, preserving caller order
        n_min = int(N.min().item())
        valid = [i for i in indices if 0 <= i < n_min]
        assert len(valid) > 0, f"no valid indices within min(N)={n_min}; got {indices}"
        K = len(valid)

        # Compute per-item output widths = sum of widths for selected blocks + (K-1) separators
        # separators are 1 column wide, consistent with compactify_data()
        b_ix = torch.arange(B, device=device)
        starts = xi[b_ix.unsqueeze(1), torch.tensor(valid, device=device), 0].long()  # (B, K)
        ends = xi[b_ix.unsqueeze(1), torch.tensor(valid, device=device), 1].long()  # (B, K)
        widths = (ends - starts).clamp_min(0)  # (B, K)
        widths_sum = widths.sum(dim=1)  # (B,)
        out_widths = widths_sum + (K - 1)  # (B,)
        max_width = int(out_widths.max().item())

        x_out = torch.zeros(B, C, H, max_width, device=device, dtype=x.dtype)
        xi_out = torch.zeros(B, K, 2, device=device, dtype=xi.dtype)
        if xl is not None:
            xl_out = torch.zeros(B, max_width, device=device, dtype=xl.dtype)

        for b in range(B):
            pos = 0
            divider_pos = int(xi[b, 0, 1].item()) if len(valid) > 1 else -1
            for k, idx in enumerate(valid):
                s0 = int(xi[b, idx, 0].item())
                e0 = int(xi[b, idx, 1].item())
                w = max(0, e0 - s0)

                # Copy all channels, not just channel 0
                x_out[b, :, :, pos : pos + w] = x[b, :, :, s0:e0]
                if xl is not None:
                    xl_out[b, pos : pos + w] = xl[b, s0:e0]

                xi_out[b, k, 0] = pos
                xi_out[b, k, 1] = pos + w
                pos += w

                # Insert separator if not the last selected block and room remains
                if k < K - 1 and divider_pos > 0:
                    # Use a separator column; e0 is end-exclusive so clamp to W-1
                    x_out[b, :, :, pos] = x[b, :, :, divider_pos]
                    if xl is not None:
                        xl_out[b, pos] = self.sep_class_id
                    pos += 1

        if xl is not None:
            return x_out, xi_out, xl_out
        else:
            return x_out, xi_out

    def remap_boundaries(
        self, x: torch.Tensor, xi: torch.Tensor, N: torch.Tensor, move_block: int = -1, xl: torch.Tensor = None
    ):
        B, _, H, W = x.shape
        device = x.device
        N = N.view(B, 1)

        if move_block >= 0:
            assert 0 <= move_block < N.min().item(), (
                f"move_block index out of range for min(N)={N.min().item()}: {move_block}"
            )

        x_out = x.clone()
        xi_out = xi.clone()
        if xl is not None:
            xl_out = xl.clone()

        for b in range(B):
            # pick a random block between 0 and N[b]-1
            Nb = int(N[b].item())
            if Nb <= 0:
                continue

            if move_block >= 0 and move_block < Nb - 1:
                # remap chosen block
                block = min(int(move_block), Nb - 2)
            else:
                # remap random block
                block = int(torch.randint(low=0, high=Nb - 1, size=(1,), device=device).item())

            next_block = block + 1
            last_block = Nb - 1

            add_offset = xi[b, last_block, 1] - xi[b, block, 1]
            sub_offset = xi[b, next_block, 0]

            xi_out[b, 0 : Nb - next_block, :] = xi[b, next_block:Nb, :] - sub_offset
            xi_out[b, Nb - next_block : Nb, :] = xi[b, 0:next_block, :] + add_offset

            be = int(xi[b, block, 1].item())
            nbb = be + 1
            lbe = int(xi[b, last_block, 1].item())

            x_out[b, :, :, 0 : lbe - nbb] = x[b, :, :, nbb:lbe]
            x_out[b, :, :, lbe - nbb] = x[b, :, :, be]
            x_out[b, :, :, lbe - nbb + 1 : lbe] = x[b, :, :, 0:be]

            if xl is not None:
                xl_out[b, 0 : lbe - nbb] = xl[b, nbb:lbe]
                xl_out[b, lbe - nbb] = xl[b, be]
                xl_out[b, lbe - nbb + 1 : lbe] = xl[b, 0:be]

        if xl is not None:
            return x_out, xi_out, xl_out
        else:
            return x_out, xi_out

    def build_column_mask(
        self,
        xi: torch.Tensor,
        hw: (None, None),
        mblock: list = [],
        iblock: list = [],
        masked_blocks: int = 0,
        half_mask: bool = False,
    ):
        """
        Generate column-wise masking pattern for spectrogram patches.

        Args:
            xi: Chirp boundaries (B, N, 2)
            hw: Spatial dimensions (H, W)
            masked_blocks: Number of chirp blocks to mask (or 0)
            mblock: Index of chirp block to mask (or -1)
            iblock: Index of chirp block to isolate (or -1) (mblock and iblock are exclusive, and mblock must be set if iblock is set)
        Returns:
            Boolean mask (B, H*W) where True = masked patches

        """
        assert xi.dim() == 3 and xi.size(2) == 2, f"xi must be (B, N, 2), got {tuple(xi.shape)}"
        H, W = hw
        device = xi.device
        B, N, _ = xi.shape

        if N == 1:
            half_mask = True

        if len(mblock) > 0:
            assert all((0 <= i < N) for i in mblock), f"invalid mblock indices N={N}, got {mblock}"
            masked_blocks = 0  # disable masked_blocks
        else:
            assert 0 < masked_blocks <= N, (
                f"masked_blocks must be greater than 0 and less than or equal to N, got {masked_blocks} and {N}"
            )

        if len(iblock) > 0:
            # if iblock is set, ensure mblock is set
            assert all((0 <= i < N) for i in iblock), f"invalid iblock indices N={N}, got {iblock}"
            assert len(mblock) > 0, f"mblock len={len(mblock)}. mblock must be set if iblock is set"

        starts = xi[:, :, 0].to(torch.long).clamp(min=0, max=W)  # (B, N)
        ends   = xi[:, :, 1].to(torch.long).clamp(min=0, max=W)  # (B, N)

        if len(mblock) > 0:
            mask_blocks = mblock
        elif masked_blocks > 0:
            mask_blocks = torch.randperm(N, device=device)[:masked_blocks].tolist()  # randomly select n_blocks blocks

        mask2d = torch.zeros(B, W, dtype=torch.bool, device=device)  # mask2d is a boolean mask of the spectrogram

        # use the start and end of the 0th item in the batch to define the mask for all items in the batch
        for block in mask_blocks:
            if half_mask:
                start = starts[0, block] + max(0, (ends[0, block] - starts[0, block]) // 2)
            else:
                start = starts[0, block]
            mask2d[:, start : ends[0, block]] = True

        mask = mask2d.unsqueeze(1).expand(B, H, W).reshape(B, H * W)

        return mask

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

    def forward_encoder(self, x: torch.Tensor, xi: torch.Tensor, xl: torch.Tensor = None, **column_mask_args):
        """
        Patchify → add positional encodings → build a **column-wise** mask from chirp boundaries →
        keep only unmasked tokens → encode with the Transformer encoder.

        Args:
            x:  (B, 1, H, W) spectrograms after any upstream compaction/windowing.
            xi: (B, N, 2) chirp [start, end) boundaries **in the current W frame**.

            column_mask_args:
            masked_blocks: Number of chirp blocks to randomly mask (or 0)
            mblock: Index of chirp block to mask (or -1)
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
        B = x.shape[0]
        z, H, W = self.tokenize_spectrogram(x)  # (B, T, D_enc), H, W
        z = self.apply_position_encoding(z)  # (B, T, D_enc)

        bool_mask = self.build_column_mask(xi, hw=(H, W), **column_mask_args)
        # bool_mask : (B, H*W), True = masked (column-wise across H) bool_mask are exclusive

        if xl is not None:
            xl_tok = xl.unsqueeze(1).expand(B, H, W).reshape(B, H * W)
            xl_tok_cond = xl_tok.clone().long()
            xl_tok_cond[bool_mask] = self.mask_class_id
            z = self.apply_label_encoding(z,xl_tok_cond)

        z_keep, idx_restore = self.mask(z, bool_mask)  # ensure the encoder never sees masked tokens
        # z_keep: Kept tokens (B, keep_count, D)
        # idx_restore: Permutation indices to restore original order (B, T)


        h = self.encoder(z_keep)  # (B, keep, D_enc)


        if xl is not None:
            return h, idx_restore, bool_mask, H * W, xl_tok_cond
        else:
            return h, idx_restore, bool_mask, H * W  # (B, keep, D_enc), (B, T), (B, T), (B,T), T

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

    def forward_decoder(self, h: torch.Tensor, idx_restore: torch.Tensor, T: int, xl_tok_cond: torch.Tensor = None):
        """
        Project encoder outputs to decoder width → insert learned mask tokens → unshuffle back to the
        original token order → add decoder positional encodings → run the Transformer decoder → predict
        pixel values for each patch.

        Args:
            h:           (B, T_enc, D_enc) encoder outputs for **kept** tokens only (T_enc ≤ T).
            idx_restore: (B, T) permutation indices that map the kept-first layout back to the original
                         token order (kept ∪ masked). Produced by `mask(...)` in the encoder path.
            T:           int total number of tokens in the **full** sequence (kept + masked) after patching.

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

        pos_dec = self.encoder_to_decoder(
            self.pos_enc[:, :T, :]
        )  # (1, T, D_dec) decoder pos-encs derived by projecting encoder pos-encs

        y_full = y_full + pos_dec

        d = self.decoder(y_full)  # (B, T, D_dec)
        pred = self.decoder_to_pixel(d)  # Final per-token patch prediction: (B, T, P). P is pixels per patch
        if xl_tok_cond is not None:
            logits_label = self.decoder_to_label(d)
            return pred, logits_label
        else:
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

        # MSE per pixel; compute a single masked mean across all masked pixels (size-invariant)
        per_pixel = (pred - target).pow(2)  # (B, T, P)
        # Defensive guard: sanitize before reduction
        if torch.isnan(per_pixel).any() or torch.isinf(per_pixel).any():
            if not hasattr(self, "_loss_nan_warned"):
                print("[loss_mse] WARNING: NaN/Inf detected in per_pixel. Dumping stats once.")
                print("  target stats:", float(t32.min()), float(t32.max()), float(var32.mean()))
                self._loss_nan_warned = True
            per_pixel = torch.nan_to_num(per_pixel, nan=0.0, posinf=0.0, neginf=0.0)

        # Expand the (B,T) mask to pixels and take a single mean over all masked pixels
        mask3d = bool_mask.unsqueeze(-1).expand_as(per_pixel)  # (B, T, P)
        masked_pixels = per_pixel[mask3d]
        if masked_pixels.numel() == 0:
            # Fallback: avoid division by zero if mask is empty (e.g., visualization mode)
            loss = per_pixel.mean()
        else:
            loss = masked_pixels.mean()

        return loss

    def loss_label(self, logits_label: torch.Tensor, xl: torch.Tensor, bool_mask: torch.Tensor, W: int):
        """
        logits_label: (B, T, 2)
        xl:           (B, T) in {0,1,2}  (2 = separator)
        bool_mask:    (B, T)
        hw:           (H, W)
        """

        B, T, C = logits_label.shape
        H = T // W

        # Per-column logits by averaging across rows (H)
        logits_hw = logits_label.view(B, H, W, C)  # (B,H,W,2)
        logits_col = logits_hw.mean(dim=1)  # (B,W,2)

        # Masked columns: any row masked in that column
        masked_cols = bool_mask.view(B, H, W).any(dim=1)  #  (B,H,W) -> (B,W)

        # Valid = masked & non-separator
        is_sep = xl == self.sep_class_id  # (B,W)
        valid = masked_cols & (~is_sep)  # (B,W)

        if not valid.any():
            return logits_label.new_tensor(0.0)

        logits = logits_col[valid]  # (N_valid, 2)
        targets = xl[valid].long()  # (N_valid,) in {0,1}
        return F.cross_entropy(logits, targets)
