# AUTOPOWER AI

A machine-learning powered automotive horsepower predictor, built on a scikit-learn
pipeline (ExtraTrees/XGBoost/Random Forest/Gradient Boosting, whichever scores best)
trained in `FinalTry_2_.ipynb`, and served through a Streamlit app (`app.py`).

## Repo contents

- `FinalTry_2_.ipynb` — data cleaning, EDA, and model training. Running it end-to-end
  regenerates a `models/` folder next to the notebook containing:
  `best_model.joblib`, `trained_models.joblib`, `dataset.joblib`, `results_df.joblib`,
  `target_col.joblib`, `feature_importance.joblib`, and writes `app.py`.
- `app.py` — the Streamlit app. Already extracted and fixed here, so you don't need
  to re-run the notebook cell that generates it unless you change the app code.
- `requirements.txt` — dependencies for local runs and for Streamlit Community Cloud.
- `models/` — **not included** in this deliverable (no dataset was provided). You need
  to run the notebook against your CSV to produce these files before deploying.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app looks for a `models/` folder next to `app.py` (see `CANDIDATE_MODEL_DIRS` in
`app.py`), so make sure that folder (with the `.joblib` files listed above) is committed
alongside `app.py`.

## Deploy on Streamlit Community Cloud (streamlit.io)

1. Push this folder to a GitHub repo, including `app.py`, `requirements.txt`, and the
   `models/` folder with your trained artifacts.
2. Go to https://streamlit.io/cloud (sign in with GitHub) and click "New app".
3. Pick your repo/branch and set the main file path to `app.py`.
4. Deploy — Streamlit Cloud installs everything from `requirements.txt` automatically.
   You do not need to run any of the notebook's install/launch cells; those were only
   for previewing the app inside a Colab runtime and have been removed from the
   cleaned notebook.

## Notes on what was fixed in the notebook

- `app.py`'s source (previously embedded as a string in one notebook cell) had a stray
  leading `...` line before its real docstring — removed.
- The model-saving cell wrote to the hardcoded Colab path `/content/models`; changed to
  a relative `models/` folder so it works in any environment, including a cloned GitHub repo.
- A cell that tried to save an undefined `dataset` variable (`joblib.dump(dataset, ...)`,
  `NameError` on execution) was removed — it was dead code duplicating the working save
  in the previous cell.
- Three overlapping `pip install streamlit ...` cells were merged into one.
- The Colab-only cells that launched Streamlit via `subprocess` and tunneled it through
  `cloudflared` were removed — they're irrelevant (and won't run) outside Colab, and
  are not part of the Streamlit Community Cloud deployment flow.
- All cell outputs were cleared and execution counts reset — the notebook previously
  carried ~4 MB of embedded plot images, which isn't needed for a GitHub upload.
