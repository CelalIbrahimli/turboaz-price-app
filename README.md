# 🚗 Turbo.az Price Estimator (Multimodal)  
### CatBoost + SBERT + Stacking

🔗 **Live demo (Hugging Face Space):**  
👉 https://huggingface.co/spaces/celalibr/turboaz-price-app  

A production-ready car price prediction system inspired by Turbo.az-style marketplace listings.  
This project combines structured tabular data and semantic understanding of car descriptions to estimate prices in real time.

It demonstrates a full end-to-end AI workflow:

**data → feature engineering → modeling → multimodal fusion → deployment → public demo**

---

## ✨ Key Features

- Multimodal modeling: **CatBoost (tabular) + SBERT embeddings (text)**
- Stacking / fusion: **Ridge meta-model** combining structured + NLP predictions
- Real-time web demo deployed on **Hugging Face Spaces**
- Segment-aware modeling (**premium vs non-premium**)
- Behavioral + market signals (**dealer, urgency, trust**)
- Lightweight inference pipeline
- Portfolio-level production system

---

## 🧠 Modeling Approach (High-level)

### 🔹 1. Tabular Model (CatBoostRegressor)

The tabular model captures structured marketplace dynamics such as:

- Depreciation and car age  
- Mileage and usage intensity  
- Engine displacement  
- Brand and segment positioning  
- Currency as a premium indicator  
- Dealer / VIP / Featured exposure signals  

**Why CatBoost?**

- Native categorical feature handling  
- Strong nonlinear modeling  
- Robust to noisy real-world marketplace data  
- Minimal preprocessing  

---

### 🔹 2. Text Model (SBERT → Ridge)

Car descriptions contain hidden pricing signals such as:

- Condition  
- Urgency  
- Trust  
- Ownership history  
- Negotiation intent  

We use multilingual SBERT:
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2


This converts raw text into semantic embeddings.

A **Ridge regression model** then predicts log-price from these embeddings.

---

### 🔹 3. Fusion Layer (Stacking)

The final prediction uses stacking:

**Inputs:**
- Tabular log-price  
- Text log-price  

**Meta-model:**
- Ridge regression  

This improves:

- Stability  
- Robustness  
- Generalization  

---

## 📊 Feature Engineering

### Structural
- Car age  
- Mileage  
- Engine size  
- Usage intensity  

### Market segmentation
- Currency  
- Premium vs non-premium  

### Behavioral
- Dealer  
- VIP  
- Urgency  
- Trust  

### Interaction features
- Dealer × urgency  
- Urgency × premium  

### NLP
- Description length  
- Semantic embeddings  

---

## 📦 Repository Contents
app.py # Gradio inference app
requirements.txt # Dependencies
tab_model.cbm # Trained CatBoost model
ridge_sbert.pkl # Text Ridge model
meta_ridge.pkl # Stacking model


---

## ▶️ Run Locally

### 1️⃣ Create environment
python -m venv .venv
source .venv/bin/activate

Windows:

.venv\Scripts\activate
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the app
python app.py

Gradio will generate a local URL.

⚠️ The first prediction may take longer because SBERT downloads (~471MB).

## 🚀 Deployment

The model is deployed on Hugging Face Spaces.

Users can:

Open the demo link

Enter car details

Write a description

Get real-time price estimation

No installation required.

If inactive, the free cloud instance may take a few seconds to wake up.

## ⚠️ Limitations

Marketplace data contains noise and missing values

Currency conversion is simplified

Description quality varies

Some vehicle attributes are unavailable

Educational and portfolio purpose only

## 🔮 Future Improvements

Explainability with SHAP inside UI

Named entity recognition for trims and options

Real-time currency conversion

Fraud and scam detection

API endpoint (FastAPI)

Dealer dashboard

Market intelligence insights

## 💡 Business Potential

Possible real-world applications:

Pricing SaaS for dealers

Fraud detection systems

Market analytics tools

Recommendation engines

## 📈 Why This Project Matters

This project demonstrates:

Real-world AI thinking

Multimodal machine learning

Feature engineering

Model stacking

NLP integration

Deployment and product mindset

This workflow is closer to industry AI pipelines than traditional academic projects.

## 🔐 License

MIT License.

## 🙌 Author

Celal Ibrahimli
AI Engineering | Multimodal ML | End-to-End Systems
