# 🚗 Turbo.az Price Estimator (Multimodal) — CatBoost + SBERT + Stacking

Live demo: **Hugging Face Space**  
https://huggingface.co/spaces/celalibr/turboaz-price-app

A production-ready car price prediction demo for **Turbo.az-style listings**, combining:
- **Tabular signals** (age, mileage, engine, dealer/VIP, currency, etc.)
- **Text semantics** from listing descriptions via **multilingual SBERT**
- A **stacked fusion** (meta Ridge) that blends tabular + text predictions

---

## ✨ Key Features

- **Multimodal modeling**: CatBoost (tabular) + SBERT embeddings (text)
- **Stacking / fusion**: Ridge meta-model to combine predictions
- **Real-time web demo**: Hosted on Hugging Face Spaces (public link)
- **Segment-aware variables**: premium vs non-premium cues (currency, dealer signals, urgency, trust flags)
- **Lightweight inference**: single prediction flow with cached models

---

## 🧠 Modeling Approach (High-level)

### 1) Tabular Model (CatBoostRegressor)
Handles heterogeneous structured features (numeric + categorical) and learns nonlinear interactions effectively.

**Examples of strong predictors**
- Depreciation / age
- Mileage intensity (km per year)
- Engine displacement
- Brand and currency as segmentation signals
- Dealer / VIP / Featured as trust & exposure proxies

### 2) Text Model (SBERT → Ridge)
- Listing description is embedded using  
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- A Ridge model predicts log-price from embeddings

### 3) Fusion (Stacking)
- Meta Ridge takes:
  - `tabular_pred_log`
  - `text_pred_log`
- Outputs final `log(price)` → converted to AZN price

---

## 📦 Repository Contents

- `app.py` — Gradio app (inference UI)
- `requirements.txt` — runtime dependencies
- `tab_model.cbm` — trained CatBoost model (tabular)
- `ridge_sbert.pkl` — Ridge model trained on SBERT embeddings
- `meta_ridge.pkl` — meta Ridge model for fusion/stacking

---

## ▶️ Run Locally

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
