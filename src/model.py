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

        self.sep_param  = nn.Parameter(torch.zeros(1, 1, 1))
        self.sep_token  = self.sep_param.expand(1, config["mels"],1)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config["dec_hidden_d"]))

        self.pos_enc  = nn.Parameter(torch.zeros(1, config["max_seq"], config["enc_hidden_d"] ))
        
        # blockwise mask width in **tokens** along W' (time)
        self.mask_block_w = config.get("mask_block_w", 32)

        self.init_weights()

    def init_weights(self):
        # randomly initialize the weights with a chosen seed
        torch.manual_seed(42)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def project_to_patch(self, x):
        # x is B, channel, height , width
        z = self.patch_projection(x)
        # p is hidden d, height tokens, width tokens 
        return z

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

        N      =  N.view(B, 1)
        N_max  =  int(N.max())

        xi     = xi[:,:N_max,:]
        xi_new = xi.clone()  

        w = (xi[:, :, 1] - xi[:, :, 0]).clamp(min=0)
        idx_row = torch.arange(N_max, device=x.device).unsqueeze(0).expand(B, -1)  
        w_valid = w * (idx_row < N)                                             
        csum = torch.cumsum(w_valid, dim=1)                                      
        seps_before = idx_row                                  
        xi_new[:, :, 0] = (csum - w_valid + seps_before).to(xi.dtype)            
        xi_new[:, :, 1] = (csum           + seps_before).to(xi.dtype)  
        xi_new[ (idx_row >= N)] = 0
        w_max = w_valid.sum(dim=1) + (N.squeeze(1).clamp(min=1) - 1)     

        # xi_new: (B, N_max, 2)  #[start, end) of slices with silence removed,  N slices per item
        # w_max: (B, )           #width of each item

        x_new = torch.zeros_like(x)  # ensures columns beyond valid region are zero
        sep_col_vec = self.sep_token[0, :, 0]  # (H,) matches x_new[b, 0, :, idx]
        for b in range(B): 
            n_valid = int(N[b, 0].item())
            for i in range(n_valid):
                s0, e0 = int(xi[b, i, 0].item())    , int(xi[b, i, 1].item())
                sn, en = int(xi_new[b, i, 0].item()), int(xi_new[b, i, 1].item())

                if en > sn and e0 > s0:
                    x_new[b, 0, :, sn:en] = x[b, 0, :, s0:e0]

                if i < n_valid - 1 and en < W:
                    x_new[b, 0, :, en] = sep_col_vec

        x_new  = x_new[ :, :, :, :int(w_max.max().item())]
        # x_new:  (B, C, H, w_max) #spectrogram, with silence removed

        return x_new, xi_new
        
    def sample_data(self, x: torch.Tensor, xi: torch.Tensor, N: torch.Tensor, n_blocks: int):
        """
        Sample random contiguous windows of chirp blocks.
        
        Args:
            x: Spectrograms (B, C, H, W)
            xi: Chirp boundaries (B, N_max, 2)
            N: Valid chirp counts per item (B,)
            n_blocks: Number of adjacent blocks to sample
        
        Returns:
            x_out: Windowed spectrograms (B, C, H, max_width)
            xi_out: Remapped boundaries in window coordinates (B, n_blocks, 2)
        """
        B, C, H, W = x.shape
        device = x.device
        N = N.view(B, 1)

        n_blocks = int(min(n_blocks, N.min().item()))  # ensure feasible for all items

        # Per item in batch start index in [0, N[b]-n_blocks]
        start_max = (N.squeeze(1) - n_blocks + 1).clamp_min(1)            # (B,)
        start = (torch.rand(B, device=device) * start_max.float()).floor().to(torch.long)  # (B,)
        end   = start + n_blocks                                            # (B,)

        b_ix = torch.arange(B, device=device)

        # Raw window bounds in ORIGINAL x-coords: start at first slice's start, end at last slice's end
        st_raw = xi[b_ix, start, 0].to(torch.long)                        # (B,)
        en_raw = xi[b_ix, end - 1, 1].to(torch.long)                      # (B,) end is exclusive

        # Per-item widths and common batch width
        widths = (en_raw - st_raw).clamp_min(1)                           # (B,)
        max_width = int(widths.max().item())

        x_out  = torch.zeros(B, C, H, max_width, device=device, dtype=x.dtype)
        xi_out = torch.zeros(B, n_blocks, 2, device=device, dtype=xi.dtype)

        for b in range(B):
            en = int(en_raw[b].item())
            st = max(0, en - max_width)      
            en = st + max_width            # right-align to end

            x_out[b, :, :, :] = x[b, :, :, st:en] # crop the spectrogram to the new window

            xi_out[b] = xi[b, start[b]:end[b], :].clone() # remap the boundaries to the new window
            xi_out[b,:,0] = xi_out[b,:,0] - st
            xi_out[b,:,1] = xi_out[b,:,1] - st

        return x_out, xi_out

    
    def build_column_mask(self, xi:torch.Tensor,  hw: (None,None), n_blocks: int = 0, frac: float = 0.0):
        """
        Generate column-wise masking pattern for spectrogram patches.
        
        Args:
            xi: Chirp boundaries (B, N, 2)
            hw: Spatial dimensions (H, W)
            n_blocks: Number of chirp blocks to mask (or 0)
            frac: Fraction of columns to mask 0.0-1.0 (or 0.0)
        
        Returns:
            Boolean mask (B, H*W) where True = masked patches
        
        Note: Exactly one of n_blocks or frac must be > 0
        """
        assert xi.dim() == 3 and xi.size(2) == 2, f"xi must be (B, N, 2), got {tuple(xi.shape)}"
        H, W = hw
        device = xi.device
        B, N, _ = xi.shape
        assert n_blocks < N, f"n_blocks must be less than N, got {n_blocks} and {N}"
        assert frac >=0 and frac < 1, f"frac must be between 0 and 1, got {frac}"
        assert not(n_blocks > 0 and frac > 0), f"either n_blocks or frac must be greater than 0 not both, got {n_blocks} and {frac}"
        assert n_blocks == 0 or frac == 0, f"n_blocks or frac must be greater than 0, got {n_blocks} and {frac}"

        starts = xi[:, :, 0].to(torch.long).clamp(min=0, max=W)   # (B, N)
        ends   = xi[:, :, 1].to(torch.long).clamp(min=0, max=W)   # (B, N)
        widths = (ends - starts).clamp(min=0)                     # (B, N)

        # get n_block random blocks between 0 and N ensure there are no duplicates
        blocks = []
        if n_blocks > 0:
           blocks = torch.randperm(N, device=device)[:n_blocks]  # randomly select n_blocks blocks
           m_w = max([ int(widths[b, blocks].sum().item()) for b in range(B) ]) # max width of the blocks
        else :  # frac > 0
           m_w = max(0, min(W - 1, int(round(float(frac) * W)))) # width of the mask, no block selected


    
        mask2d = torch.zeros(B, W, dtype=torch.bool, device=device) # mask2d is a boolean mask of the spectrogram
        for b in range(B):
            for block in blocks:
                mask2d[b, starts[b, block]:ends[b, block]] = True
            remaining = (~mask2d[b]).nonzero(as_tuple=False).squeeze(1) # remaining is the indices of the unmasked blocks
            remaining = remaining[torch.randperm(remaining.numel(), device=device)[:m_w - int(mask2d[b].sum().item())]] # randomly select remaining columns to keep mask width constant for each item in the batch
            mask2d[b, remaining] = True

        return mask2d.unsqueeze(1).expand(-1, H, -1).flatten(1,2) # (B,W) -> (B, H, W) -> (B, H*W) 

    def mask(self, z: torch.Tensor, bool_mask: torch.Tensor ) :
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
        B, T    = bool_mask.shape 

        # key = mask_flag * (T+1) + position  → keeps original order within kept/masked groups
        pos = torch.arange(T, device=z.device).unsqueeze(0).expand(B, T)          # (B,T)
        key = bool_mask.to(torch.int64) * (T + 1) + pos                           # (B,T)
        order = key.argsort(dim=1)                                                # (B,T), kept-first, masked second

        # --- gather kept-first tokens; slice to max keep (rectangular) ---
        z_perm     = torch.gather(z, 1, order.unsqueeze(-1).expand(B, T, D))      # (B,T,D)
        keep_count = int((~bool_mask[0,:]).sum().item())
        z_keep     = z_perm[:, :keep_count, :].contiguous()                       # (B,max_keep,D)

        # --- inverse permutation to restore original order ---
        idx_restore = order.argsort(dim=1)                                         # (B,T)


        return z_keep, idx_restore


    def forward_encoder(self, x: torch.Tensor, xi : torch.Tensor):
        """
        Patchify → add pos enc → mask → Transformer encoder.
        Returns:
        h: (B, keep, D_enc), idx_restore, bool_mask, T
        """

        z_img = self.patch_projection(x)               # (B, D_enc, H', W')
        H, W = z_img.shape[-2], z_img.shape[-1]
        z = z_img.flatten(2, 3).transpose(1, 2)        # (B, T, D_enc)
        B, T, D = z.shape
        assert T <= self.max_seq, f"T={z.size(1)} exceeds max_seq={self.max_seq}"
        z = z + self.pos_enc[:, :T, :]                 # (B, T, D_enc)

        bool_mask = self.build_column_mask(xi, hw=(H, W), n_blocks=1 )

        # z_keep, idx_restore, bool_mask = self.mask(z, hw=(H, W))    
        z_keep, idx_restore = self.mask(z, bool_mask)


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
        assert T <= self.max_seq, f"T={z.size(1)} exceeds max_seq={self.max_seq}"
        z = z + self.pos_enc[:, :T, :]                  # (B, T, D_enc)
        h = self.encoder(z)                             # (B, keep, D_enc)
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
        target = unfold(x).transpose(1, 2) 
        
        print(f"[TinyBird test] loss_mse: target={tuple(target.shape)}, pred={tuple(pred.shape)}, bool_mask={tuple(bool_mask.shape)}")
        
        # Normalize target patches
        target_mean = target.mean(dim=-1, keepdim=True)
        target_std = target.std(dim=-1, keepdim=True)
        target = (target - target_mean) / (target_std + 1e-6)
        
        loss = ((pred - target) ** 2)[bool_mask].mean()
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
        "max_seq" : 30001,
        "patch_size" : (32, 1),

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
    print(f"[TinyBird test] device={device}, mels={config["mels"]}, timebins={config["num_timebins"]}, "
          f"patch={config['patch_size']}, max_seq={config['max_seq']}")

    dataset = SpectogramDataset(
        dir=str(args.input_dir),
        n_mels=config["mels"],
        n_timebins=config["num_timebins"],
    )
 
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)
    model = TinyBird(config).to(device)
    model.eval()
    batch = next(iter(loader))  # (spec, chirp_intervals, N, filename)
    spectrograms, chirp_intervals, N, filenames = batch

    x  = spectrograms.to(device, non_blocking=False)     # (B=2, 1, H=mels, W=time)
    xi = chirp_intervals.to(device, non_blocking=False)  # (B=2, N_max, 2)
    N  = N.to(device, non_blocking=False)                # (B=2, 1)

    print(f"[TinyBird test] batch shapes: x={tuple(x.shape)}, xi={tuple(xi.shape)}, N={tuple(N.shape)}")
    print(f"[TinyBird test] files: {filenames[0]} | {filenames[1]}")

    # --- 3) encoder ---
    with torch.no_grad():

        print(f"[TinyBird test] before prepare_data: x={tuple(x.shape)}, xi={tuple(xi.shape)}, N={tuple(N.shape)}")
        x, xi = model.compactify_data(x, xi, N)
        print(f"[TinyBird test] after prepare_data: x={tuple(x.shape)}, xi={tuple(xi.shape)}")
        x ,xi  = model.sample_data(x, xi, N, n_blocks=3)
        print(f"[TinyBird test] after window_data: x={tuple(x.shape)}, xi={tuple(xi.shape)}")

        h, idx_restore, bool_mask, T = model.forward_encoder(x, xi)

    print(f"[TinyBird test] encoder outputs:"
          f" h={tuple(h.shape)}, idx_restore={tuple(idx_restore.shape)},"
          f" bool_mask={tuple(bool_mask.shape)}, T={T}")

    # --- 4) decoder on same batch ---
    with torch.no_grad():
        pred = model.forward_decoder(h, idx_restore, T)  # (B, T, P)

    print(f"[TinyBird test] decoder output: pred={tuple(pred.shape)}")

    # (optional) compute MSE on masked patches, like training loop
    with torch.no_grad():
        loss = model.loss_mse(x, pred, bool_mask).item()
    print(f"[TinyBird test] masked-patch MSE loss: {loss:.6f}")

    print("[TinyBird test] ✅ success")