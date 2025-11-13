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


####
To Do Next SWE GOALS
#### 
[] INHERITANCE IN THE DATACLASS, AND TWO TRAINING SCRIPTS (also inheritance)
[] Classificaiton mixes individuals if folders not serperated, fix this 
[] Breaking apart song for inference context length should be its own script
[] Collate by song length to not do extreme padding
[] Settings to set root? for path 
[] At ability to add song detector to audio2spec.py 
[] a "dataloader" util for birdset iteration 
[] audio2spec needs to be refactored and simplfied 
[] standardize the data fields that are saved into the json files 
[] discrepency in VRAM usage between 