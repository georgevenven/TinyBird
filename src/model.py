import torch, math 
from torch import nn

class TinyBird(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.patch_size = config["patch_size"]
        self.max_seq = config["max_seq"]
        self.mask_p = config["mask_p"]

        self.patch_projection = nn.Conv2d(
            in_channels = 1,
            out_channels = config["enc_hidden_d"],
            kernel_size = self.patch_size,
            stride = self.patch_size 
        )

        self.encoder_transformer_block = nn.TransformerEncoderLayer(
            d_model=config["enc_hidden_d"], nhead=config["enc_n_head"],
            dim_feedforward=config["enc_dim_ff"], dropout=config["dropout"],
            batch_first=True, norm_first=True 
        )

        self.decoder_transformer_block = nn.TransformerEncoderLayer(
            d_model=config["dec_hidden_d"], nhead=config["dec_n_head"],
            dim_feedforward=config["dec_dim_ff"], dropout=config["dropout"],
            batch_first=True, norm_first=True  
        )

        self.encoder = nn.TransformerEncoder(self.encoder_transformer_block, num_layers=config["enc_n_layer"])
        self.decoder = nn.TransformerEncoder(self.decoder_transformer_block, num_layers=config["dec_n_layer"])

        self.encoder_to_decoder = nn.Linear(config["enc_hidden_d"], config["dec_hidden_d"])
        self.decoder_to_pixel = nn.Linear(config["dec_hidden_d"], self.patch_size[0] * self.patch_size[1])

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config["dec_hidden_d"]))

        self.pos_enc = nn.Parameter(torch.zeros(1, config["max_seq"], config["enc_hidden_d"]))
        # blockwise mask width in **tokens** along W' (time)
        self.mask_block_w = config.get("mask_block_w", 32)

    def project_to_patch(self, x):
        # x is B, channel, height , width
        z = self.patch_projection(x)
        # p is hidden d, height tokens, width tokens 
        return z

    def encode_pos(self, z):
        z = z.flatten(2,3).transpose(2,1) # into a seq of tokens  B, Dim, tokens
        print(z.shape)
        z += self.pos_enc
        return z # shape batch x seq x dim 

        # ───────────── written by GPT ─────────────
    def mask(self, z: torch.Tensor, hw=None):
        """
        Blockwise stripes + exact target masked count via top-up/trim.
        Returns:
          z_keep: (B, keep, D)
          idx_restore: (B, T)
          bool_mask: (B, T)  True where masked
        """
        if hw is None:
            raise ValueError("mask(hw=...) requires spatial shape (H', W').")
        B, T, D = z.shape
        H, W = hw
        assert H * W == T
        Wb = min(self.mask_block_w, W)

        # target masked tokens
        n_mask = max(1, int(round(self.mask_p * T)))

        # sample stripes (allows overlap = fast)
        n_blocks = max(1, math.ceil(n_mask / Wb))
        rows   = torch.randint(H, (n_blocks,), device=z.device)               # with replacement
        starts = torch.randint(W - Wb + 1, (n_blocks,), device=z.device)      # with replacement
        cols   = starts.unsqueeze(1) + torch.arange(Wb, device=z.device).unsqueeze(0)  # (n_blocks, Wb)

        mask2d = torch.zeros(H, W, dtype=torch.bool, device=z.device)
        mask2d[rows.unsqueeze(1).expand_as(cols).reshape(-1), cols.reshape(-1)] = True
        flat = mask2d.view(-1)                                                # (T,)

        # enforce exact cardinality n_mask (top-up or trim)
        cur = int(flat.sum().item())
        if cur < n_mask:
            need = n_mask - cur
            cand = (~flat).nonzero(as_tuple=False).squeeze(1)
            add  = cand[torch.randperm(cand.numel(), device=z.device)[:need]]
            flat[add] = True
        elif cur > n_mask:
            drop = cur - n_mask
            ones = flat.nonzero(as_tuple=False).squeeze(1)
            rem  = ones[torch.randperm(ones.numel(), device=z.device)[:drop]]
            flat[rem] = False

        bool_mask = flat.view(1, T).expand(B, T)                              # (B,T)
        idx_mask  = flat.nonzero(as_tuple=False).squeeze(1)                   # (n_mask,)
        idx_keep  = (~flat).nonzero(as_tuple=False).squeeze(1)                # (T-n_mask,)
        keep = idx_keep.numel()

        idx_keep_b = idx_keep.unsqueeze(0).expand(B, keep)
        z_keep = torch.gather(z, 1, idx_keep_b.unsqueeze(-1).expand(B, keep, D))

        perm = torch.cat([idx_keep, idx_mask], dim=0)                         # kept-first layout
        idx_restore = perm.argsort().unsqueeze(0).expand(B, T)
        return z_keep, idx_restore, bool_mask

    def forward_encoder(self, x: torch.Tensor):
        """
        Patchify → add pos enc → mask → Transformer encoder.
        Returns:
          h: (B, keep, D_enc), idx_restore, bool_mask, T
        """
        z_img = self.patch_projection(x)               # (B, D_enc, H', W')
        H, W = z_img.shape[-2], z_img.shape[-1]
        z = z_img.flatten(2, 3).transpose(1, 2)        # (B, T, D_enc)
        B, T, D = z.shape
        if T > self.pos_enc.size(1):
            raise ValueError(f"T={T} exceeds max_seq={self.pos_enc.size(1)}")
        z = z + self.pos_enc[:, :T, :]                 # (B, T, D_enc)
        z_keep, idx_restore, bool_mask = self.mask(z, hw=(H, W))  # blockwise mask
        h = self.encoder(z_keep)                       # (B, keep, D_enc)
        return h, idx_restore, bool_mask, T
    
    def forward_encoder_inference(self, x: torch.Tensor):
        """
        Patchify → add pos enc → mask → Transformer encoder.
        Returns:
          h: (B, keep, D_enc), idx_restore, bool_mask, T
        """
        z = self.patch_projection(x)                   # (B, D_enc, H', W')
        z = z.flatten(2, 3).transpose(1, 2)            # (B, T, D_enc)
        B, T, D = z.shape
        if T > self.pos_enc.size(1):
            raise ValueError(f"T={T} exceeds max_seq={self.pos_enc.size(1)}")
        z = z + self.pos_enc[:, :T, :]                 # (B, T, D_enc)

        h = self.encoder(z)                       # (B, keep, D_enc)
        return h

    def forward_decoder(self, h: torch.Tensor, idx_restore: torch.Tensor, T: int):
        """
        Project to decoder dim → insert mask tokens → unshuffle → add pos → decode → predict pixels.
        Returns:
          pred: (B, T, P) where P = patch_size[0]*patch_size[1]
        """
        B = h.size(0)
        # project encoder tokens to decoder width
        y = self.encoder_to_decoder(h)                 # (B, keep, D_dec)
        D_dec = self.decoder_to_pixel.in_features
        keep = y.size(1)

        # build full sequence with mask tokens then unshuffle to original order
        # NOTE: define in __init__: self.mask_token = nn.Parameter(torch.zeros(1,1,D_dec))
        mask_tokens = self.mask_token.expand(B, T - keep, D_dec)     # (B, T-keep, D_dec)
        y_full = torch.cat([y, mask_tokens], dim=1)                  # kept-first layout
        y_full = torch.gather(y_full, 1, idx_restore.unsqueeze(-1).expand(B, T, D_dec))

        # reuse encoder pos via linear map to decoder width
        pos_dec = self.encoder_to_decoder(self.pos_enc[:, :T, :])    # (1, T, D_dec)
        y_full = y_full + pos_dec

        d = self.decoder(y_full)                                     # (B, T, D_dec)
        pred = self.decoder_to_pixel(d)                               # (B, T, P)
        return pred

    def loss_mse(self, x: torch.Tensor, pred: torch.Tensor, bool_mask: torch.Tensor):
        """
        Compute MSE on masked patches only.
        x:    (B, 1, H, W)
        pred: (B, T, P) from decoder
        """
        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        target = unfold(x).transpose(1, 2)                            # (B, T, P)
        
        # Normalize target patches
        target_mean = target.mean(dim=-1, keepdim=True)
        target_std = target.std(dim=-1, keepdim=True)
        target = (target - target_mean) / (target_std + 1e-6)
        
        loss = ((pred - target) ** 2)[bool_mask].mean()
        return loss
