# AFL Fantasy DNN

A learning project: building deep neural networks (in Keras/TensorFlow) on AFL Fantasy data.

The goal is to *understand* DNNs by building them, working up from fundamentals to an
autoencoder, applied to a real and fun dataset.

## Deliverables

1. **Archetype discovery** — an autoencoder compresses each player's stat profile into a
   small latent vector; players close together in latent space share a playing style.
2. **Form / anomaly detection** — flag players who are statistically unusual, and (in a
   later temporal extension) players whose recent trajectory suggests they're about to
   hit form or fall off.

## Learning arc

1. **Environment** — Python 3.12 + Keras/TensorFlow in an isolated virtual environment.
2. **Data** — source clean per-round AFL player stats into a tidy dataset.
3. **Fundamentals** — a small supervised "predict next-round score" regressor to learn the
   end-to-end mechanics (forward pass, loss, backprop, train/val split, overfitting).
4. **Autoencoder** — archetype discovery (deliverable #1).
5. **Temporal extension** — sequence / variational model for form & anomaly detection
   (deliverable #2).

## Project structure

```
data/
  raw/         # untouched source data (gitignored)
  processed/   # cleaned, model-ready datasets (gitignored)
models/        # saved model artifacts (gitignored)
notebooks/     # interactive exploration & training
src/           # reusable Python modules (data, models, training)
```

## Setup (Windows / PowerShell)

```powershell
# Create the virtual environment with Python 3.12 (TensorFlow has no 3.14 build yet)
py -3.12 -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the pipeline

Scripts (run from the repo root, in order):

```powershell
python src\scrape_footywire.py 2021 2025   # 1. scrape -> data/processed/player_match_stats.csv
python src\explore_data.py                  # 2. data sanity checks + figures
python src\features.py                      # 3. leakage-free feature table
python src\train_regressor.py               # 4. baselines vs linear vs neural net
python src\autoencoder.py                   # 5. archetype discovery (deliverable #1)
python src\sequence_model.py                # 6. LSTM form/anomaly detection (deliverable #2)
python src\predict_upcoming.py              # 7. predict next round (see weekly workflow below)
```

Or read the whole story interactively in **`notebooks/afl_fantasy_dnn.ipynb`** (consumes the
scraped CSV; runs in a couple of minutes).

## Why Python 3.12?

TensorFlow's latest stable release supports Python 3.10–3.13 only — there is no Windows
build for Python 3.14 yet. We pin to 3.12 (the well-tested sweet spot) inside a virtual
environment so the system Python stays untouched and the project stays reproducible.

## Weekly workflow (PC)

Run from the repo root each week before games. Fixtures come from Squiggle automatically;
predictions are written to `web/data/predictions/` and the web index reads `manifest.json`.

```powershell
# 1. Refresh player stats through the current season (include current year)
.\.venv\Scripts\python.exe src\scrape_footywire.py 2021 2026

# 2. Predict the next unplayed round (halts if that round is already done)
.\.venv\Scripts\python.exe src\predict_upcoming.py

# 3. Publish CSVs + manifest to the Raspberry Pi
.\scripts\publish-predictions.ps1
```

Or combine predict + publish:

```powershell
.\.venv\Scripts\python.exe src\predict_upcoming.py --deploy
```

To regenerate a round that was already predicted:

```powershell
.\.venv\Scripts\python.exe src\predict_upcoming.py --force
```

Preview the site locally:

```powershell
.\.venv\Scripts\python.exe -m web
```

Then open http://localhost:8001

RPi setup and deploy are documented in **`RPI_DEPLOY_CHECKLIST.md`**.