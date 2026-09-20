# DR Grading — Streamlit App

Streamlit front-end for the **Hybrid Classification-Regression Framework for
Ordinal Diabetic Retinopathy Grading** (EfficientNet-B3, dual classification +
regression heads, QWK-optimized thresholds, Grad-CAM++ explainability).

## 1. Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

### Adding your trained weights
The app looks for a checkpoint at `models/best_hybrid_model.pth` by default.
You have two options:

1. **Bundle it**: export your trained model from the notebook
   (`torch.save(model.state_dict(), "best_hybrid_model.pth")`) and drop the file
   into `models/`.
2. **Upload at runtime**: use the "Upload trained checkpoint" file uploader in
   the sidebar — no restart needed.

Without either, the app runs in **demo mode** (ImageNet-pretrained backbone,
untrained heads) so you can still see the full UI and pipeline working, but
predictions won't be clinically meaningful.

If you saved your optimized Nelder-Mead thresholds, drop them as a JSON list
at `models/thresholds.json`, e.g. `[0.52, 1.48, 2.61, 3.4]`. Otherwise the app
uses the paper's initial thresholds `[0.5, 1.5, 2.5, 3.5]`.

## 2. Deploy (Streamlit Community Cloud — free)

1. Push this folder (`app.py`, `requirements.txt`, and `models/` if you're
   bundling weights) to a **public or private GitHub repo**.
   - ⚠️ `best_hybrid_model.pth` for an EfficientNet-B3 dual-head model will
     likely be 40–50 MB. GitHub's normal push limit is 100 MB per file, so a
     plain commit works — but consider **Git LFS** if you also commit the
     4-fold ensemble checkpoints (6 files could add up).
   - Alternative if you don't want weights in GitHub at all: host the `.pth`
     file somewhere (Hugging Face Hub, Google Drive, S3) and download it at
     startup inside `app.py` with `requests`/`gdown`, or just use the sidebar
     uploader so users supply it per-session.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**, pick the repo/branch and set the main file to
   `app.py`.
3. Deploy. Streamlit Cloud installs `requirements.txt` automatically.
4. Free tier gives 1 CPU / 1 GB RAM — fine for single-image inference with a
   ~11M-parameter EfficientNet-B3, since you're not training here.

### Alternative hosts
- **Hugging Face Spaces** (Streamlit SDK) — same idea, good if your weights
  are already on the HF Hub.
- **Render / Railway** — for more control (Docker), useful if you later add a
  proper REST API alongside the UI.

## 3. Files
- `app.py` — the full Streamlit app
- `requirements.txt` — pinned dependencies
- `models/` — put `best_hybrid_model.pth` (and optionally `thresholds.json`)
  here
