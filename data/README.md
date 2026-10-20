# data/

Raw data is never committed. Expected layout:

```
data/
  avocado/                      'Hass' Avocado Ripening Photographic Dataset (Mendeley 10.17632/3xd9n945v8.1, CC BY 4.0)
    <labels>.xlsx               one row per photo: filename, sample, storage group, stage 1-5, day, side
    <images>/*.jpg              800x800 JPEGs
  features/
    avocado_resnet18.npz        from `make features` (scripts/extract_features.py)
  stevens/
    manifest.csv                see src/ripen/data/stevens.py for the schema (dataset not yet available)
```

Download the avocado set from https://data.mendeley.com/datasets/3xd9n945v8/1 and unzip it here.
Other candidate datasets and their licenses are listed in `docs/dataset_survey.md`.
