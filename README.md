# Avocado Ripeness Scanner

**MMA3001 Numerical Methods and Machine Learning, individual project**

This project estimates the ripeness of a Hass avocado from a single photograph, without cutting or squeezing the fruit. It was built for a self-service scanning kiosk in a supermarket's avocado section. A shopper scans the avocado they picked up and the kiosk tells them whether it is ready to eat now, or roughly how many more days it needs at room temperature.

It combines two computational techniques:

| Part | Technique | What it does |
|---|---|---|
| Image classifier | HSV colour feature extraction followed by **k-nearest neighbours (kNN)** | Predicts the ripening stage (1–5) from the photo |
| Baseline | Single-feature **colour-threshold rule** | Reference method the kNN must beat |
| Ripening model | **Logistic curve fitted by nonlinear least squares** (`scipy.optimize.curve_fit`) | Models ripening index against time for each storage temperature, and converts a predicted stage into "days until ready" |

---

## Results at a glance

These results use 629 photos of 42 avocados, evaluated with **grouped 5-fold cross-validation**. Every test avocado was unseen during training.

| Model | Accuracy | Macro F1 | Within ±1 stage | Ready-to-eat yes/no |
|---|---|---|---|---|
| Baseline (hue threshold) | 0.55 | 0.55 | 0.98 | 0.72 |
| kNN, all 10 features | 0.61 | 0.61 | 0.98 | 0.76 |
| **kNN, lighting-invariant (final)** | **0.60** | **0.60** | **0.98** | **0.76** |

- **Robustness:** when test photos are made 30 % darker, the all-feature kNN drops to 0.53 accuracy. The lighting-invariant kNN stays at 0.60, so it was selected for the kiosk.
- **Speed:** about 5 ms per scan (segmentation, features and prediction) at 200 × 200 px.
- **Ripening curves** were fitted to all 14,722 labels in the dataset. They reach stage 3 ("ready") on day **5.5 at 20 °C** and on day **13.4 at 10 °C**. The rate constant gives **Q10 ≈ 2.1**, meaning ripening roughly doubles in speed for every 10 °C rise.

The figures are in [`results/figures/`](results/figures) and every number above is in [`results/metrics.json`](results/metrics.json).

---

## Repository structure

```
avocado-ripeness/
├── README.md                  ← you are here
├── LICENSE                    ← MIT licence (code)
├── requirements.txt           ← Python dependencies
├── pyproject.toml             ← pytest configuration
├── src/avocado_ripeness/      ← source code (importable package)
│   ├── features.py            ← fruit segmentation + HSV colour features
│   ├── baseline.py            ← colour-threshold baseline classifier
│   ├── classifier.py          ← kNN pipeline, k selection, kiosk message
│   └── ripening_model.py      ← logistic curve fitting, time-to-ready, Q10
├── scripts/run_pipeline.py    ← runs the whole experiment end to end
├── tests/                     ← pytest unit tests (19 tests)
├── docs/                      ← HTML source-code documentation (pdoc)
├── data/                      ← dataset metadata + subset lists (see data/README.md)
└── results/                   ← metrics.json, features CSV, figures, test report
```

---

## How to reproduce

### 1. Install

Python 3.10 or newer is required (it was tested on 3.13).

```bash
git clone https://github.com/ZOE-YO618/avocado-ripeness.git
cd avocado-ripeness
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the images

Download the *'Hass' Avocado Ripening Photographic Dataset* (DOI [10.17632/3xd9n945v8.1](https://doi.org/10.17632/3xd9n945v8.1)) and unzip it. See [`data/README.md`](data/README.md) for details.

### 3. Run the experiment

```bash
python scripts/run_pipeline.py --image-dir "path/to/Avocado Ripening Dataset" --subset data/subset_files.csv
```

This takes about 1 minute on a laptop and writes `results/metrics.json`, `results/features_200.csv` and the figures in `results/figures/`. Add `--skip-resolution` to skip the resolution study and make it faster.

### 4. Run the tests

```bash
python -m pytest -v
python -m pytest --html=results/test_report.html --self-contained-html   # HTML test report
```

The tests use small synthetic images and manufactured data, so they **do not need the dataset**. One of them is a verification test: it fits the logistic model to a curve with known `k` and `t0` and checks that it recovers them.

### 5. Rebuild the documentation

```bash
pdoc src/avocado_ripeness -o docs --docformat numpy
```

Then open `docs/index.html` in a browser.

---

## Method summary

1. **Segmentation.** The photo is converted RGB → HSV. Pixels that are both bright and unsaturated are treated as the white lightbox background. Everything else is fruit. A morphological opening is applied and the largest connected region is kept.
2. **Features.** These are computed from the fruit pixels only: the circular mean hue (stored as cos/sin, so that 350° and 10° average to 0° rather than 180°), mean and spread of saturation and brightness, and the fractions of green, purple and very dark pixels.
3. **Baseline.** The unwrapped mean hue is split into 5 stages by 4 cut-points, which are chosen to maximise training accuracy.
4. **kNN.** The features are standardised and then classified by distance-weighted kNN. `k` is chosen by an inner grouped cross-validation on the training folds only. The final model uses only the 7 features that do not depend on brightness.
5. **Validation.** Grouped 5-fold CV splits by **avocado ID**. Consecutive daily photos of the same fruit look almost identical, so a random split would leak information and inflate the scores.
6. **Ripening model.** `R(t) = 1 + 4 / (1 + exp(-k (t - t0)))` is fitted separately for each storage group by Levenberg–Marquardt least squares. It is compared against the 3-parameter form `R_max / (1 + exp(-k (t - t0)))` and validated on 20 % of avocados that were held out from the fit.

## Main limitations

- Stages 4 and 5 are often confused. Stage 5 is defined by mould spots and senescence, which mean colour features capture poorly.
- The photos come from a controlled lightbox. A real kiosk would need a similar enclosure, and the lighting test only simulates uniform brightness changes, not colour casts or shadows.
- The model applies to the Hass variety only, and the classifier was trained on a 629-image subset of the dataset.

---

## Data licence and citation

The dataset is © Xavier, Rodrigues & Silva (2024), used under **CC BY 4.0**:

> Xavier, P., Rodrigues, P., Silva, C. L. M. (2024). *'Hass' Avocado Ripening Photographic Dataset*. Mendeley Data, V1. https://doi.org/10.17632/3xd9n945v8.1

The code in this repository is released under the MIT licence (see [`LICENSE`](LICENSE)).

## AI use

AI assistance (Claude) was used while developing this project. The full disclosure of which tools were used, what for, and how the output was checked is in the *AI use and reflection* section of the written report.
