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

- Times are milliseconds; `plot_embedding_bf.py` converts them to 2 ms timebins.
