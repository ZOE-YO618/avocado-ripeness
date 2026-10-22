# Data

## Source
'Hass' Avocado Ripening Photographic Dataset  
Pedro Xavier, Pedro Rodrigues, Cristina L. M. Silva (2024). Mendeley Data, Version 1.  
DOI: [10.17632/3xd9n945v8.1](https://doi.org/10.17632/3xd9n945v8.1) — licence **CC BY 4.0**.

- 14,710 photographs (800 × 800 px JPEG) of 478 Hass avocados, two photos per fruit per day (sides `a` and `b`), taken in a studio lightbox.
- Storage groups: `T10` = 10 °C / 85 % RH, `T20` = 20 °C / 85 % RH, `Tam` = ambient conditions.
- Labels: 5-stage Ripening Index — 1 Underripe, 2 Breaking, 3 Ripe (first phase), 4 Ripe (second phase), 5 Overripe.

## Files in this folder
| File | What it is |
|---|---|
| `Avocado Ripening Dataset.xlsx` | Original metadata spreadsheet from the dataset (file name, time stamp, storage group, sample ID, day, ripening index). Redistributed under CC BY 4.0. |
| `subset_files.csv` | The 629 image file names used for the classifier experiments in the report. |
| `subset_samples.json` | The 42 avocado sample IDs in that subset (14 per storage group, side `a` photos only, random seed 42). |

## Getting the images
The photographs are **not** stored in this repository because of their size. Download the dataset from the DOI link above, unzip it, and pass the folder that contains the `.jpg` files to the pipeline:

```bash
python scripts/run_pipeline.py --image-dir "path/to/Avocado Ripening Dataset" --subset data/subset_files.csv
```

Drop `--subset` to use every image found in the folder.

## Why a subset?
The ripening-curve model uses **all 14,722 labels** in the spreadsheet. The image classifier uses a subset of 629 photos (42 avocados × every day, one side) to keep image transfer and run time manageable within the project time frame. The subset was chosen *by avocado* (not by photo) and stratified by storage group, so every ripening stage and every storage condition is represented.
