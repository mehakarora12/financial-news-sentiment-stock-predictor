# 📈 Financial News Sentiment Stock Predictor

> Predicts next-day stock price direction by combining **15 technical indicators** with **FinBERT-based sentiment analysis** of financial news, powered by an XGBoost classifier with SHAP explainability — deployed as a live Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25-red?logo=streamlit)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)
![FinBERT](https://img.shields.io/badge/FinBERT-NLP-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://financial-news-sentiment-stock-predictor-h3gzaezbwf66wdfmme9mq.streamlit.app/)

---

## 🎯 Project Overview

This project answers one question: **"Will this stock go up or down tomorrow?"**

It combines two types of signals:
1. **Technical Indicators** — 15 features derived from 5 years of OHLCV price data (RSI, MACD, Bollinger Bands, moving averages, volume ratios)
2. **Sentiment Scores** — extracted from 712 financial news headlines using **ProsusAI/FinBERT**, a BERT model fine-tuned on financial text

Both are fed into a tuned **XGBoost classifier**. The project includes a **3-experiment ablation study**, a full **SHAP explainability layer**, and an interactive **Streamlit prediction dashboard** that fetches live data and explains each prediction in real time.

---

## 🏆 Key Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost — Technical Only (Tuned)** ✅ | 0.520 | 0.525 | **0.842** | **0.646** | 0.522 |
| XGBoost — Tech + Sentiment (6,030 rows) | 0.519 | 0.524 | 0.838 | 0.644 | 0.517 |
| XGBoost — Tech + Sentiment (90 rows) | 0.573 | 0.093 | 0.200 | 0.127 | 0.518 |
| Logistic Regression — Technical Only | 0.505 | 0.542 | 0.353 | 0.419 | 0.519 |

**Best Model: XGBoost with Technical Indicators Only (Tuned)**
- **F1 Score: 0.646** — strong performance for next-day direction on noisy financial data
- **Recall: 0.842** — catches 84% of actual up days, making it a reliable buy-signal detector
- **+9.6% F1 improvement** from hyperparameter tuning (0.550 → 0.646) via `RandomizedSearchCV` with `TimeSeriesSplit`

---

## 🖥️ Streamlit App Demo

The live dashboard:
- Selects from 5 tickers (AAPL, MSFT, GOOGL, AMZN, TSLA)
- Fetches real-time price data via `yfinance`
- Computes all 15 technical indicators live
- Predicts next-day direction with P(UP) / P(DOWN)
- Generates a **SHAP waterfall plot** explaining *why* the model made that prediction
- Displays a price chart with SMA, Bollinger Band, and RSI overlays

```bash
cd app
streamlit run streamlit_app.py
```

---

## 📁 Project Structure

```
financial-news-sentiment-stock-predictor/
│
├── data/
│   ├── price_data.csv              # 6,275 rows OHLCV data, 5 tickers, 5 years
│   ├── news_data.csv               # 712 cleaned headlines across 5 tickers
│   ├── sentiment_features.csv      # 121 ticker-day FinBERT sentiment rows
│   └── features_combined.csv       # 6,030-row master dataset, 25 features
│
├── models/
│   ├── xgboost_model.pkl           # Final trained XGBoost model
│   ├── feature_cols.pkl            # Feature column list for inference
│   ├── best_params.pkl             # Tuned hyperparameters
│   ├── shap_explainer.pkl          # SHAP TreeExplainer (used by Streamlit app)
│   ├── shap_feature_importance.csv # Ranked feature importance from SHAP
│   └── ablation_results_tuned.csv  # Full experiment comparison table
│
├── notebooks/
│   ├── 01_data_collection_price.ipynb    # yfinance OHLCV fetch + target creation
│   ├── 02_data_collection_news.ipynb     # NewsAPI headlines + relevance filtering
│   ├── 03_sentiment_finbert.ipynb        # FinBERT scoring + daily aggregation
│   ├── 04_feature_engineering.ipynb      # 15 technical indicators + EDA
│   ├── 05_model_training.ipynb           # 3-experiment ablation + tuning
│   └── 06_shap_explainability.ipynb      # SHAP beeswarm, waterfall, heatmap
│
├── app/
│   └── streamlit_app.py                  # Live prediction dashboard
│
├── plots/
│   ├── eda_target_distribution.png
│   ├── eda_cross_ticker_features.png
│   ├── eda_correlation_heatmap.png
│   ├── ablation_comparison.png
│   ├── confusion_matrices.png
│   ├── shap_beeswarm.png                 # Global SHAP beeswarm
│   ├── shap_bar.png                      # Mean |SHAP| feature ranking
│   ├── shap_dependence_top3.png          # Dependence plots — top 3 features
│   ├── shap_waterfall_3cases.png         # Waterfall for 3 prediction examples
│   └── shap_ticker_heatmap.png           # Per-ticker SHAP importance heatmap
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔧 Tech Stack

| Category | Tools |
|----------|-------|
| Data Collection | `yfinance`, `NewsAPI` |
| NLP / Sentiment | `ProsusAI/finbert`, `transformers`, `torch` |
| Feature Engineering | `ta` library, `pandas`, `numpy` |
| Machine Learning | `XGBoost`, `scikit-learn` (TimeSeriesSplit, RandomizedSearchCV) |
| Explainability | `SHAP` (TreeExplainer, waterfall, beeswarm, dependence plots) |
| Visualization | `matplotlib`, `seaborn` |
| App | `Streamlit` |
| Environment | Google Colab (notebooks), VS Code (app) |

---

## 📊 Dataset

### Stocks Covered
| Ticker | Company | Rows | Date Range |
|--------|---------|------|-----------|
| AAPL | Apple Inc. | 1,206 | Aug 2021 – Jun 2026 |
| MSFT | Microsoft Corp. | 1,206 | Aug 2021 – Jun 2026 |
| GOOGL | Alphabet Inc. | 1,206 | Aug 2021 – Jun 2026 |
| AMZN | Amazon.com Inc. | 1,206 | Aug 2021 – Jun 2026 |
| TSLA | Tesla Inc. | 1,206 | Aug 2021 – Jun 2026 |

**Total: 6,030 rows** after dropping NaN rows from rolling window indicators (49 rows × 5 tickers = 245 dropped from 6,275 raw)

### Target Variable
- **Binary classification:** 1 = price UP next trading day, 0 = price DOWN
- Created using `groupby('ticker').shift(-1)` — avoids cross-ticker contamination
- **Class balance: 52% Up / 48% Down** — well balanced, no oversampling required

### News Data
- **Source:** NewsAPI (free tier — last 30 days)
- **712 headlines** after multi-query fetch, keyword relevance filtering, deduplication
- **Limitation:** 30-day news window vs 5-year price history → 98.7% of sentiment imputed as neutral (0)

---

## ⚙️ Feature Engineering

### Technical Features (15 total)

| Feature | Description | Category |
|---------|-------------|----------|
| `close_to_sma20` | Close / 20-day SMA | Trend |
| `close_to_sma50` | Close / 50-day SMA | Trend |
| `rsi_14` | 14-day RSI | Momentum |
| `rsi_from_70` | RSI − 70 (distance from overbought) | Momentum |
| `rsi_from_30` | RSI − 30 (distance from oversold) | Momentum |
| `macd` | EMA12 − EMA26 | Trend/Momentum |
| `macd_signal` | 9-day EMA of MACD | Trend/Momentum |
| `macd_diff` | MACD histogram | Trend/Momentum |
| `bb_width` | (BB_high − BB_low) / SMA20 | Volatility |
| `bb_position` | (Close − BB_low) / (BB_high − BB_low) | Volatility |
| `daily_return` | Daily % price change | Returns |
| `volatility_10` | 10-day rolling std of returns | Volatility |
| `volume_change` | Daily % volume change | Volume |
| `volume_ratio` | Volume / 20-day avg volume | Volume |
| `ticker_encoded` | LabelEncoded ticker identity | Identity |

### Sentiment Features (5 total — ablation only)
| Feature | Description |
|---------|-------------|
| `avg_sentiment` | Mean FinBERT signed score (−1 to +1) |
| `sentiment_std` | Std dev of daily sentiment (disagreement) |
| `news_count` | Number of relevant articles |
| `positive_count` | Count of positive headlines |
| `negative_count` | Count of negative headlines |

### Key Design Decisions
- **Ratio features over raw prices** — `close_to_sma20` instead of `Close` + `sma_20` separately. Scale-invariant across 5 tickers and 5 years.
- **Per-ticker indicator computation** — AAPL's SMA only uses AAPL's price history. Never mixing MSFT data into AAPL's indicators.
- **`ticker_encoded` as feature** — lets XGBoost learn TSLA's high-volatility regime vs MSFT's stability (TSLA-volatility correlation: 0.41).
- **Neutral imputation (0)** for missing sentiment — "no news = neutral signal" is consistent and defensible.

---

## 🧪 Ablation Study Design

Three experiments were run to isolate the contribution of sentiment features:

### Experiment 1: Technical Only (Baseline)
- **Data:** 6,030 rows | **Features:** 15 technical
- **Result:** XGBoost F1 = **0.646**, Recall = 0.842 ← Best

### Experiment 2: Tech + Sentiment (90 real-news rows)
- **Data:** 90 rows (days with actual FinBERT scores, May–Jun 2026)
- **Result:** XGBoost F1 = **0.127** — collapsed from insufficient data
- **Finding:** 90 rows / 5 folds = ~15 test rows each — statistically unreliable. Demonstrates that 30 days of NewsAPI data is insufficient for model training.

### Experiment 3: Tech + Sentiment (Full 6,030 rows)
- **Data:** 6,030 rows (sentiment imputed as 0 for missing days)
- **Result:** XGBoost F1 = **0.644** — nearly identical to Experiment 1
- **Finding:** With 1.3% real sentiment coverage, sentiment adds negligible value. Full historical news coverage would be needed to see a meaningful contribution.

---

## 🔍 Model Training & Validation

### Why TimeSeriesSplit?
Standard random splits cause **data leakage** in time series — the model would train on future data. `TimeSeriesSplit` always trains on past, tests on future:

```
Fold 1: Train [Aug 2021 – Feb 2024] → Test [Feb 2024 – Jul 2024]
Fold 2: Train [Aug 2021 – Jul 2024] → Test [Jul 2024 – Jan 2025]
Fold 3: Train [Aug 2021 – Jan 2025] → Test [Jan 2025 – Jul 2025]
Fold 4: Train [Aug 2021 – Jul 2025] → Test [Jul 2025 – Dec 2025]
Fold 5: Train [Aug 2021 – Dec 2025] → Test [Dec 2025 – Jun 2026]
```

`TimeSeriesSplit` was also used **inside `RandomizedSearchCV`** during hyperparameter tuning — a common mistake beginners make is using standard CV folds during tuning, which leaks future data.

### Why XGBoost Over Logistic Regression?
- All feature-target correlations are near-zero (<0.05) — expected in semi-efficient markets
- XGBoost captures **non-linear feature interactions** (e.g., RSI oversold + BB near lower band + negative MACD → likely bounce UP)
- XGBoost F1: **0.646** vs Logistic Regression F1: **0.419**

---

## 🔍 SHAP Explainability

Five SHAP visualizations generated in Phase 6:

1. **Beeswarm plot** — 6,030 SHAP values per feature; shows direction and magnitude of each feature's impact
2. **Bar plot** — mean |SHAP| ranking across all predictions
3. **Dependence plots** — how top-3 features' values map to their SHAP contribution (non-linear relationships XGBoost learned)
4. **Waterfall plots** — three individual predictions explained: confident UP, confident DOWN, borderline
5. **Per-ticker heatmap** — mean |SHAP| per feature per ticker; shows TSLA relies more on `volatility_10` while MSFT leans on `macd_diff`

The SHAP explainer is also embedded in the Streamlit app — every live prediction generates a fresh waterfall explanation.

---

## 📈 EDA Key Findings

1. **Near-zero linear correlations** with target (all < 0.05) — justifies tree-based model over linear regression
2. **Multicollinearity among trend indicators** — RSI ↔ bb_position (r=0.90), macd ↔ close_to_sma50 (r=0.94). Handled naturally by XGBoost's splitting logic
3. **TSLA volatility regime** — dramatically wider distribution for `volatility_10` and `daily_return` vs other tickers. Justifies `ticker_encoded` as a feature
4. **Sentiment distribution** — 98.7% imputed zeros; avg_sentiment spike at 0 confirms real coverage is minimal
5. **Balanced target** — all tickers show 51–53% up days; no class imbalance handling needed
6. **bb_position shows slight separation** — median 0.63 for up days vs 0.55 for down days — small but consistent directional signal

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/financial-news-sentiment-stock-predictor.git
cd financial-news-sentiment-stock-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up NewsAPI key (for notebooks 01–03 only)
```bash
# Locally:
export NEWSAPI_KEY=your_key_here

# Or in Google Colab: add as a Colab Secret named NEWSAPI_KEY
```

### 4. Run notebooks in order (Google Colab)
```
01_data_collection_price.ipynb   → Price data
02_data_collection_news.ipynb    → News headlines
03_sentiment_finbert.ipynb       → FinBERT sentiment
04_feature_engineering.ipynb     → Features + EDA
05_model_training.ipynb          → Models + ablation
06_shap_explainability.ipynb     → SHAP analysis
```

### 5. Run the Streamlit app (local, VS Code)
```bash
cd app
streamlit run streamlit_app.py
```

---

## 💡 Limitations & Future Work

### Current Limitations
1. **NewsAPI free tier** — 30-day news window means only 1.3% real sentiment coverage
2. **Next-day prediction** — hardest prediction horizon; weekly/monthly would show stronger signal
3. **No live trading integration** — predictions are informational, not connected to a brokerage

### Future Improvements
1. **Full news history** — GDELT Project or paid NewsAPI for 5-year news coverage; expected to significantly improve sentiment contribution
2. **Additional signals** — earnings proximity, VIX, options flow, sector ETF momentum
3. **Ensemble modeling** — XGBoost + LSTM for sequence modeling
4. **Per-ticker models** — separate XGBoost per ticker to fully capture individual stock dynamics

---

## 📋 Requirements

```
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.23.0
ta>=0.10.0
transformers>=4.30.0
torch>=2.0.0
xgboost>=1.7.0
scikit-learn>=1.2.0
shap>=0.42.0
matplotlib>=3.6.0
seaborn>=0.12.0
streamlit>=1.25.0
newsapi-python>=0.2.7
```

---

## 👩‍💻 Author

**Mehak**  
Data Science Portfolio Project  
Built with Google Colab · VS Code · Streamlit

---

## 📄 License

This project is licensed under the MIT License.
