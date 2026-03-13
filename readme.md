# SongMAE

SongMAE is the current project name. The codebase still contains many `TinyBird` references in module names, scripts, run directories, and saved configs because the project was renamed after we ran into naming conflicts with other repos, models, and projects. When the README says `SongMAE`, it refers to the same model family that is still called `TinyBird` in much of the code.

## Model Summary

- SongMAE is a pretrained foundation model for birdsong representation learning at the syllable and within-song level.
- It is trained as a masked autoencoder on spectrograms; after pretraining, the decoder is typically discarded and the encoder is used as the representation model.
- Its embeddings can be used for downstream analysis tasks such as unsupervised syllable clustering and individual identification.
- The pretrained encoder can be supervised fine-tuned with `src/supervised_train.py` for song detection, unit detection, and syllable classification.

## Pretrained Checkpoints

Current repo-local checkpoints and configs:

The XCM checkpoints below include weights trained through step `500000` for the long-run XCM pretraining runs.

| Variant | Spectrogram grid | Weights | Config | Notes |
| --- | --- | --- | --- | --- |
| SongMAE `32h x 1w` | `mels=128`, `patch_height=32`, `patch_width=1`, `num_timebins=1024` | [weights](runs/xcm_voronoi_mask_no_normalize_32h_1w/weights/model_step_499999.pth) | [config](runs/xcm_voronoi_mask_no_normalize_32h_1w/config.json) | `max_seq=4096` |
| SongMAE `64h x 1w` | `mels=128`, `patch_height=64`, `patch_width=1`, `num_timebins=1024` | [weights](runs/xcm_voronoi_mask_no_normalize_64h_1w/weights/model_step_499999.pth) | [config](runs/xcm_voronoi_mask_no_normalize_64h_1w/config.json) | `max_seq=2048` |
| SongMAE `32h x 10w` | `mels=128`, `patch_height=32`, `patch_width=10`, `num_timebins=1000` | [weights](runs/xcm_voronoi_mask_no_normalize_32h_10w_zf_scratch10k_bs24_20260312_143950/weights/model_step_009999.pth) | [config](runs/xcm_voronoi_mask_no_normalize_32h_10w_zf_scratch10k_bs24_20260312_143950/config.json) | `max_seq=400` |
| SongMAE `128h x 1w` | `mels=128`, `patch_height=128`, `patch_width=1`, `num_timebins=1024` | [weights](runs/xcm_voronoi_mask_no_normalize_128h_1w_bf_scratch10k_bs24_20260312_153130/weights/model_step_009999.pth) | [config](runs/xcm_voronoi_mask_no_normalize_128h_1w_bf_scratch10k_bs24_20260312_153130/config.json) | `max_seq=1024` |

## Mels vs. Patch Config

`mels` is not a model hyperparameter you set by hand inside `pretrain.py`. It is loaded from `audio_params.json` in the spectrogram directory and represents the number of mel-frequency bins in each spectrogram.

- `mels`: spectrogram height in mel bins.
- `patch_height`: how many mel bins each patch spans vertically.
- `patch_width`: how many spectrogram timebins each patch spans horizontally.
- `num_timebins`: the input window width, in spectrogram timebins, used for training or extraction.
- `max_seq`: the total number of patches per input window.

The code computes:

```text
patch_size = (patch_height, patch_width)
max_seq = (num_timebins / patch_width) * (mels / patch_height)
```

A few concrete examples from the saved configs in `runs/`:

- `mels=128`, `patch_height=32`, `patch_width=1`, `num_timebins=1024` gives `4 x 1024 = 4096` patches.
- `mels=128`, `patch_height=128`, `patch_width=1`, `num_timebins=1024` gives `1 x 1024 = 1024` patches.
- `mels=128`, `patch_height=32`, `patch_width=10`, `num_timebins=1000` gives `4 x 100 = 400` patches.

Practical caveats:

- `mels` must be divisible by `patch_height`.
- `num_timebins` must be divisible by `patch_width`.
- Smaller `patch_width` keeps finer temporal resolution.
- Larger `patch_width` compresses time more aggressively and produces fewer tokens.
- Larger `patch_height` merges more of the frequency axis into each token.

## Annotation JSON Format

SongMAE embedding extraction and the supervised data pipeline use an annotation JSON with millisecond timing. The expected structure is:

```json
{
  "metadata": {
    "units": "ms"
  },
  "recordings": [
    {
      "recording": {
        "filename": "clip.wav",
        "bird_id": "bird_001",
        "detected_vocalizations": 3
      },
      "detected_events": [
        {
          "onset_ms": 120.5,
          "offset_ms": 860.0,
          "units": [
            {
              "onset_ms": 120.5,
              "offset_ms": 180.0,
              "id": 0
            },
            {
              "onset_ms": 240.0,
              "offset_ms": 320.0,
              "id": 1
            }
          ]
        }
      ]
    }
  ]
}
```

Notes:

- All times are in milliseconds.
- `recording.filename` is matched by stem during embedding extraction, so `clip.wav` in the JSON must match `clip.npy` in the spectrogram directory.
- For embedding extraction, filename stems need to match exactly. Chunk suffixes such as `__ms_<start>_<end>` are not resolved by `src/extract_embedding.py`.
- `bird_id` is optional for general parsing, but it is used when filtering with `--bird`.
- `detected_events` are song-level regions.
- `units` are the finer annotations inside each event.
- Unit IDs are stored as integers and become the per-timebin labels used for embedding plots.
- If `units` is empty, extraction still works for the event windows, but labels stay `-1` and the UMAP plots will be unlabeled.

## Embedding Evaluation

[`scripts/eval/eval_embedding.py`](scripts/eval/eval_embedding.py) is a thin wrapper around [`src/extract_embedding.py`](src/extract_embedding.py). It extracts event-level embeddings, writes an `.npz`, then renders UMAP plots and spectrogram snapshots.

The script writes:

- `results_dir/embeddings.npz`
- `results_dir/metrics.json`
- `results_dir/umap/*.png` and `results_dir/umap/*.pdf`
- `results_dir/spectrograms/*.png`

### `--json_path` is required in practice

`--json_path` is marked optional in the CLI, but for this workflow it is effectively required.

- `src/extract_embedding.py` only builds event windows from the annotation JSON.
- If no JSON is provided, no events are matched.
- Unmatched files are skipped.
- The script later concatenates the collected embedding lists, so running without annotations leaves nothing to save.

Use a command like:

```bash
python scripts/eval/eval_embedding.py \
  --results_dir results/example_eval \
  --spec_dir /path/to/specs \
  --run_dir /path/to/run \
  --checkpoint model_step_010000.pth \
  --json_path /path/to/annotations.json \
  --bird bird_001
```

If you want embeddings for evaluation, you should always provide:

- `--spec_dir`
- `--run_dir`
- `--json_path`

Add `--bird` when you want to restrict extraction to a single bird ID from the JSON.

## Embedding NPZ Format

The extraction step writes an `.npz` containing spectrogram slices, labels, embeddings, and the metadata needed to interpret them.

| Key | Shape | Meaning |
| --- | --- | --- |
| `spectrograms` | `(total_timebins, mels)` | Concatenated spectrogram frames for the extracted event regions. |
| `labels_original` | `(total_timebins,)` | Per-timebin labels before patch downsampling. Silence or unlabeled regions are `-1`. |
| `labels_downsampled` | `(N_patches,)` | Labels pooled to patch resolution using `patch_width`. |
| `encoded_embeddings_before_pos_removal` | `(N_patches, num_patches_height * D)` | Encoder outputs before subtracting position-wise means. |
| `encoded_embeddings_after_pos_removal` | `(N_patches, num_patches_height * D)` | Encoder outputs after subtracting the mean vector for each temporal position ID. |
| `patch_embeddings_before_pos_removal` | `(N_patches, num_patches_height * D)` | Patch-projected features before the encoder and before position-mean removal. |
| `patch_embeddings_after_pos_removal` | `(N_patches, num_patches_height * D)` | Patch-projected features after position-mean removal. |
| `pos_ids` | `(N_patches,)` | Temporal patch index within the model window. |
| `audio_sr` | scalar | Sample rate from `audio_params.json`. |
| `audio_n_mels` | scalar | Number of mel bins from `audio_params.json`. |
| `audio_hop_size` | scalar | Hop size from `audio_params.json`. |
| `audio_fft` | scalar | FFT size from `audio_params.json`. |
| `patch_height` | scalar | Vertical patch size in mel bins. |
| `patch_width` | scalar | Horizontal patch size in spectrogram timebins. |
| `num_patches_time` | scalar | Number of temporal patches in one model window. |
| `num_patches_height` | scalar | Number of vertical patches in one model window. |
| `checkpoint` | scalar string | Checkpoint name used for extraction, if supplied. |
| `model_num_timebins` | scalar | Model input width in timebins from the saved run config. |
| `mels` | scalar | Mel-bin count from the saved run config. |

Two details matter when reading these arrays:

- Each embedding row is one temporal patch after flattening the vertical patch axis, so the feature width is `num_patches_height * D`, not just `D`.
- `encoded_embeddings_after_pos_removal` is the array used by `scripts/eval/eval_embedding.py` for the default UMAP plots.

## To Do Next SWE Goals

- [ ] We need docs pretty bad to explain the codebase (when updating / cleaning up code make it)
- [ ] INHERITANCE IN THE DATACLASS, AND TWO TRAINING SCRIPTS (also inheritance)
- [ ] Classification mixes individuals if folders not separated, fix this
- [ ] Breaking apart song for inference context length should be its own script
- [ ] Collate by song length to not do extreme padding
- [ ] Settings to set root? for path
- [ ] Add ability to add song detector to audio2spec.py
- [ ] A "dataloader" util for birdset iteration
- [ ] audio2spec needs to be refactored and simplified
- [ ] Standardize the data fields that are saved into the json files
- [ ] Discrepancy in VRAM usage between
- [ ] Rename to SongMAE everywhere
- [ ] Reconstructions should all be noramlized patch wise (if used during training) to prevent jarring viz
- [ ] We need a central util for managing the json format ... this will make life so much easier
- [ ] snippification of datasets (for easy upload and faster training)
- [ ] remove spec subfolder when sbatch audio2spec
- [ ] early stopping for supervised train and maybe pretrain
- [ ] revert to the old extract embedding and figure out a way to exlude padding
- [ ] refactor all code that was touched by AI
- [ ] make sure that plotting utils can be called independetly as well as part of the scripts they are a part of

## Bugs

- [ ] irregularity with number of specs made in log for XCM vs HSN_test
