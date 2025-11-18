## JSON Format

```jsonc
{
  "metadata": {"units": "ms"},
  "recordings": [
    {
      "recording": {"filename": str, "bird_id": str, ...},
      "detected_events": [
        {
          "onset_ms": float,
          "offset_ms": float,
          "units": [
            {"onset_ms": float, "offset_ms": float, "id": int},
            ...
          ]
        },
        ...
      ]
    },
    ...
  ]
}
```

- Times are milliseconds
- all values are relative to the absolute start and end of files 


#### 
audio2spec.py --> compute_statistics_of_spectrograms.py --> pretrain.py --> plot_embedding.py

#### To Do Next SWE GOALS

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
- [ ] Reconstructions should all be noramlized patch wise (if used during training) to precent jarring viz 
- [ ] We need a central util for managing the json format ... this will make life so much easier 
- [ ] snippification of datasets (for easy upload and faster training)
- [ ] remove spec subfolder when sbatch audio2spec

#### Bugs
- [] irregularity with number of specs made in log for XCM vs HSN_test
