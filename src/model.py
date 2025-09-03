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
    def mask(self, z: torch.Tensor):
        """
        Uniform batch mask like MAE: keep (1-mask_p) fraction in original order.
        Returns:
          z_keep:    (B, keep, D)
          idx_restore: (B, T) inverse perm to restore original order after concat
          bool_mask: (B, T) True for masked locations
        """
        B, T, D = z.shape
        keep = max(1, int(round(T * (1 - self.mask_p))))
        # one shared permutation for the whole batch
        perm = torch.rand(T, device=z.device).argsort()
        idx_keep = perm[:keep].sort().values           # original temporal order
        idx_mask = perm[keep:]
        idx_keep_b = idx_keep.unsqueeze(0).expand(B, keep)

        z_keep = torch.gather(z, 1, idx_keep_b.unsqueeze(-1).expand(B, keep, D))

        bool_mask = torch.ones(B, T, dtype=torch.bool, device=z.device)
        bool_mask.scatter_(1, idx_keep_b, False)

        idx_restore = perm.argsort().unsqueeze(0).expand(B, T)
        return z_keep, idx_restore, bool_mask

    def forward_encoder(self, x: torch.Tensor):
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

        z_keep, idx_restore, bool_mask = self.mask(z)  # mask uniformly across batch
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
