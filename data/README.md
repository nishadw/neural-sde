# data/

Raw data is never committed. Only this file, `manifests/` and the split files are tracked.

```
data/
  avocado/     Mendeley 'Hass' avocado set (10.17632/3xd9n945v8.1, CC BY 4.0): spreadsheet + images/*.jpg
               (python scripts/download_avocado.py, resumable)
  a2sl/        A2SL microwave scans + photos (scripts/ingest_a2sl.py; see docs/data_a2sl.md)
  pbc2/, cmapss/   second domains (docs/data_second_domain.md)
  features/    frozen-encoder features from scripts/extract_features.py
```

Other candidate datasets: `docs/dataset_survey.md`.
