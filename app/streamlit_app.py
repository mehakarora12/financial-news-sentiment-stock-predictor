"""
streamlit_app.py — Financial News Sentiment Stock Predictor
Phase 7: Interactive Prediction Dashboard

HOW TO RUN:
    cd app/
    streamlit run streamlit_app.py

REQUIREMENTS (install once):
    pip install streamlit yfinance xgboost scikit-learn shap pandas numpy ta matplotlib seaborn
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import yfinance as yf
from datetime import datetime, timedelta
import ta
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Sentiment Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
TICKER_NAMES = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corp.',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'TSLA': 'Tesla Inc.'
}
TICKER_ENCODED = {'AAPL': 0, 'AMZN': 1, 'GOOGL': 2, 'MSFT': 3, 'TSLA': 4}

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding-top: 1rem; }

    /* Signal cards */
    .signal-strong-up {
        background: linear-gradient(135deg, #0d3321, #0f5132);
        border: 2px solid #28a745;
        border-radius: 16px; padding: 24px 20px; text-align: center;
    }
    .signal-lean-up {
        background: linear-gradient(135deg, #0a2e1a, #155724);
        border: 2px solid #5cb85c;
        border-radius: 16px; padding: 24px 20px; text-align: center;
    }
    .signal-strong-down {
        background: linear-gradient(135deg, #3b0a0a, #5c1010);
        border: 2px solid #dc3545;
        border-radius: 16px; padding: 24px 20px; text-align: center;
    }
    .signal-lean-down {
        background: linear-gradient(135deg, #2e1010, #4a1515);
        border: 2px solid #e07070;
        border-radius: 16px; padding: 24px 20px; text-align: center;
    }

    /* Signal bar */
    .signal-bar-container {
        background: #1e1e1e;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 10px 0;
        border: 1px solid #333;
    }
    .signal-bar-label {
        font-size: 0.8rem;
        color: #aaa;
        margin-bottom: 6px;
    }

    /* Gauge arrow */
    .gauge-wrap {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid #333;
        margin-top: 10px;
    }

    .info-box {
        background: #1a2a3a;
        border-left: 4px solid #2196F3;
        padding: 10px 14px;
        border-radius: 4px;
        margin: 10px 0;
        font-size: 0.85rem;
        color: #ccc;
    }
    .disclaimer {
        background: #2a2000;
        border-left: 4px solid #ffc107;
        padding: 10px 14px;
        border-radius: 4px;
        margin: 10px 0;
        font-size: 0.82rem;
        color: #ddd;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL CLASSIFICATION — KEY FIX
# Uses model's predict() for the direction, then tiers by probability
# for strength label. This avoids the "always uncertain" problem.
# ─────────────────────────────────────────────────────────────────────────────
def classify_signal(prob_up: float, predicted: int):
    """
    Classify prediction into one of 4 signal tiers.
    Direction is from model.predict() (0.5 threshold, optimised during training).
    Strength tiers: Strong (≥55%) vs Lean (50-55%) — realistic for this model.
    
    Returns: (css_class, icon, label, subtitle, color, strength_pct)
    """
    if predicted == 1:  # Model says UP
        if prob_up >= 0.55:
            return ("signal-strong-up", "🟢", "BULLISH SIGNAL",
                    "Strong upward lean", "#4caf50", prob_up)
        else:
            return ("signal-lean-up", "🔼", "MILD BULLISH",
                    "Slight upward lean", "#8bc34a", prob_up)
    else:  # Model says DOWN
        prob_down = 1 - prob_up
        if prob_down >= 0.55:
            return ("signal-strong-down", "🔴", "BEARISH SIGNAL",
                    "Strong downward lean", "#f44336", prob_up)
        else:
            return ("signal-lean-down", "🔽", "MILD BEARISH",
                    "Slight downward lean", "#ff7043", prob_up)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    artifacts = {}
    try:
        with open(os.path.join(MODEL_DIR, 'xgboost_model.pkl'), 'rb') as f:
            artifacts['model'] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'feature_cols.pkl'), 'rb') as f:
            artifacts['feature_cols'] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'shap_explainer.pkl'), 'rb') as f:
            artifacts['explainer'] = pickle.load(f)
        imp_path = os.path.join(MODEL_DIR, 'shap_feature_importance.csv')
        if os.path.exists(imp_path):
            artifacts['importance_df'] = pd.read_csv(imp_path)
        artifacts['loaded'] = True
    except FileNotFoundError as e:
        artifacts['loaded'] = False
        artifacts['error'] = str(e)
    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_price_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index()
        raw.columns = [c.strip() for c in raw.columns]
        raw = raw.rename(columns={'index': 'Date'})
        raw['Date'] = pd.to_datetime(raw['Date'])
        raw = raw.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        raw = raw.sort_values('Date').reset_index(drop=True)
        return raw
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()


def compute_technical_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    d = df.copy()
    close  = d['Close']
    high   = d['High']
    low    = d['Low']
    volume = d['Volume']

    d['sma_20'] = close.rolling(20).mean()
    d['sma_50'] = close.rolling(50).mean()
    d['ema_10'] = close.ewm(span=10, adjust=False).mean()
    d['ema_20'] = close.ewm(span=20, adjust=False).mean()

    d['close_to_sma20'] = close / d['sma_20']
    d['close_to_sma50'] = close / d['sma_50']

    d['rsi_14']      = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    d['rsi_from_70'] = d['rsi_14'] - 70
    d['rsi_from_30'] = d['rsi_14'] - 30

    macd_obj         = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    d['macd']        = macd_obj.macd()
    d['macd_signal'] = macd_obj.macd_signal()
    d['macd_diff']   = macd_obj.macd_diff()

    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    d['bb_high']     = bb.bollinger_hband()
    d['bb_low']      = bb.bollinger_lband()
    d['bb_width']    = (d['bb_high'] - d['bb_low']) / d['sma_20']
    d['bb_position'] = ((close - d['bb_low']) / (d['bb_high'] - d['bb_low'])).clip(0, 1)

    d['daily_return']  = close.pct_change()
    d['volatility_10'] = d['daily_return'].rolling(10).std()
    d['volume_change'] = volume.pct_change()
    d['volume_ratio']  = volume / volume.rolling(20).mean()
    d['ticker_encoded'] = TICKER_ENCODED[ticker]

    d = d.dropna().reset_index(drop=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def plot_probability_gauge(prob_up: float, signal_color: str) -> plt.Figure:
    """
    Horizontal probability bar showing P(UP) vs P(DOWN).
    Green = UP side, Red = DOWN side, dividing line at 50%.
    """
    fig, ax = plt.subplots(figsize=(6, 1.2))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')

    # DOWN bar (left, red)
    ax.barh(0, 1.0, color='#3a0a0a', height=0.6, left=0)
    # UP bar (right, green) — proportional width
    ax.barh(0, prob_up, color='#28a745', height=0.6, left=0, alpha=0.85)
    ax.barh(0, 1 - prob_up, color='#dc3545', height=0.6, left=prob_up, alpha=0.85)

    # Center line
    ax.axvline(0.5, color='white', linewidth=1.5, linestyle='--', alpha=0.6)

    # Labels
    ax.text(0.02, 0, f"DOWN  {1-prob_up:.1%}", va='center', ha='left',
            color='white', fontsize=9, fontweight='bold')
    ax.text(0.98, 0, f"{prob_up:.1%}  UP", va='center', ha='right',
            color='white', fontsize=9, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    plt.tight_layout(pad=0.1)
    return fig


def plot_shap_waterfall(explainer, X_row: pd.DataFrame, feature_cols: list) -> plt.Figure:
    shap_vals = explainer.shap_values(X_row)
    base_val  = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val)
    explanation = shap.Explanation(
        values=shap_vals[0],
        base_values=base_val,
        data=X_row.iloc[0].values,
        feature_names=feature_cols
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.tight_layout()
    return fig


def plot_feature_bar(importance_df: pd.DataFrame) -> plt.Figure:
    top = importance_df.head(10).sort_values('mean_abs_shap')
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#4caf50' if v > 0 else '#f44336'
              for v in top['mean_signed_shap']]
    bars = ax.barh(top['feature'], top['mean_abs_shap'], color=colors, alpha=0.85)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title("Global Feature Importance (Training Data)", fontsize=12, fontweight='bold')
    for bar, val in zip(bars, top['mean_abs_shap']):
        ax.text(val + 0.0001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    ax.set_xlim(0, top['mean_abs_shap'].max() * 1.2)
    plt.tight_layout()
    return fig


def plot_price_chart(df: pd.DataFrame, ticker: str) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6),
                                    gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(df['Date'], df['Close'], label='Close', color='#2196F3', linewidth=1.5)
    if 'sma_20' in df.columns:
        ax1.plot(df['Date'], df['sma_20'], label='SMA 20', color='#FFA726',
                 linewidth=1, linestyle='--', alpha=0.85)
    if 'sma_50' in df.columns:
        ax1.plot(df['Date'], df['sma_50'], label='SMA 50', color='#AB47BC',
                 linewidth=1, linestyle='--', alpha=0.85)
    if 'bb_high' in df.columns:
        ax1.fill_between(df['Date'], df['bb_low'], df['bb_high'],
                         alpha=0.08, color='#90CAF9', label='BB Bands')
        ax1.plot(df['Date'], df['bb_high'], color='#90CAF9', linewidth=0.5, alpha=0.5)
        ax1.plot(df['Date'], df['bb_low'],  color='#90CAF9', linewidth=0.5, alpha=0.5)
    ax1.set_title(f"{ticker} — {TICKER_NAMES[ticker]}  |  1-Year Price Chart",
                  fontweight='bold', fontsize=12)
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.2)

    if 'rsi_14' in df.columns:
        rsi = df['rsi_14']
        ax2.plot(df['Date'], rsi, color='#CE93D8', linewidth=1.2)
        ax2.axhline(70, color='#f44336', linestyle='--', linewidth=0.8, alpha=0.7)
        ax2.axhline(30, color='#4caf50', linestyle='--', linewidth=0.8, alpha=0.7)
        ax2.axhline(50, color='#aaa',    linestyle=':',  linewidth=0.6, alpha=0.5)
        ax2.fill_between(df['Date'], 70, rsi, where=(rsi >= 70), alpha=0.2, color='#f44336')
        ax2.fill_between(df['Date'], 30, rsi, where=(rsi <= 30), alpha=0.2, color='#4caf50')
        ax2.set_ylabel("RSI (14)", fontsize=9)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.15)
        ax2.text(df['Date'].iloc[-1], 73, 'OB', fontsize=7, color='#f44336', ha='right')
        ax2.text(df['Date'].iloc[-1], 24, 'OS', fontsize=7, color='#4caf50', ha='right')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL SIGNAL SUMMARY — contextual interpretation of indicators
# ─────────────────────────────────────────────────────────────────────────────
def indicator_signals(latest) -> list:
    """
    Return a list of (indicator, value, signal_text, is_bullish) tuples
    for the snapshot panel. Gives plain-English reading of each indicator.
    """
    signals = []

    rsi = latest.get('rsi_14', np.nan)
    if not np.isnan(rsi):
        if rsi > 70:
            signals.append(("RSI (14)", f"{rsi:.1f}", "Overbought — caution", False))
        elif rsi < 30:
            signals.append(("RSI (14)", f"{rsi:.1f}", "Oversold — potential bounce", True))
        elif rsi > 55:
            signals.append(("RSI (14)", f"{rsi:.1f}", "Bullish momentum", True))
        elif rsi < 45:
            signals.append(("RSI (14)", f"{rsi:.1f}", "Bearish momentum", False))
        else:
            signals.append(("RSI (14)", f"{rsi:.1f}", "Neutral zone", None))

    macd_diff = latest.get('macd_diff', np.nan)
    if not np.isnan(macd_diff):
        if macd_diff > 0:
            signals.append(("MACD Hist", f"{macd_diff:.4f}", "Bullish crossover", True))
        else:
            signals.append(("MACD Hist", f"{macd_diff:.4f}", "Bearish crossover", False))

    bb_pos = latest.get('bb_position', np.nan)
    if not np.isnan(bb_pos):
        if bb_pos > 0.80:
            signals.append(("BB Position", f"{bb_pos:.2f}", "Near upper band — extended", False))
        elif bb_pos < 0.20:
            signals.append(("BB Position", f"{bb_pos:.2f}", "Near lower band — oversold", True))
        else:
            signals.append(("BB Position", f"{bb_pos:.2f}", "Mid-band range", None))

    ret = latest.get('daily_return', np.nan)
    if not np.isnan(ret):
        signals.append(("Today Return", f"{ret:.2%}", "Up day" if ret > 0 else "Down day", ret > 0))

    vol_ratio = latest.get('volume_ratio', np.nan)
    if not np.isnan(vol_ratio):
        if vol_ratio > 1.5:
            signals.append(("Vol Ratio", f"{vol_ratio:.2f}x", "High volume — strong move", None))
        elif vol_ratio < 0.7:
            signals.append(("Vol Ratio", f"{vol_ratio:.2f}x", "Low volume — weak conviction", None))
        else:
            signals.append(("Vol Ratio", f"{vol_ratio:.2f}x", "Normal volume", None))

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown("# 📈 Financial News Sentiment Stock Predictor")
    st.markdown("*Next-day direction prediction using XGBoost · Technical Indicators · FinBERT Sentiment*")
    st.markdown("---")

    artifacts = load_model_artifacts()

    if not artifacts.get('loaded'):
        st.error(f"""
        ❌ **Model files not found.** Expected at: `{MODEL_DIR}/`

        Required files: `xgboost_model.pkl`, `feature_cols.pkl`, `shap_explainer.pkl`

        **Fix:** Copy the `models/` folder from your Google Drive project to your local project root.
        Error: `{artifacts.get('error', 'Unknown')}`
        """)
        st.stop()

    model         = artifacts['model']
    feature_cols  = artifacts['feature_cols']
    explainer     = artifacts['explainer']
    importance_df = artifacts.get('importance_df', None)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        ticker = st.selectbox(
            "Select Stock",
            TICKERS,
            format_func=lambda x: f"{x} — {TICKER_NAMES[x]}"
        )
        show_shap  = st.toggle("Show SHAP Explanation", value=True)
        show_chart = st.toggle("Show Price Chart", value=True)
        st.markdown("---")
        st.markdown("### 📊 Model Performance")
        st.markdown("""
        | Metric | Score |
        |--------|-------|
        | F1 Score | **0.646** |
        | Recall | **0.842** |
        | Accuracy | 52.0% |
        | ROC-AUC | 0.522 |

        *Recall 0.842 = catches 84% of actual up days. Useful as a buy-signal detector.*
        """)
        st.markdown("---")
        st.caption("Built by Mehak · Data Science Portfolio")

    # ── Fetch & Process ──
    st.markdown(f"### 🔄 Live Data — **{ticker}** · {TICKER_NAMES[ticker]}")

    with st.spinner(f"Downloading 1-year {ticker} price data..."):
        raw_df = fetch_price_data(ticker, period="1y")

    if raw_df.empty:
        st.error(f"Could not fetch data for {ticker}. Check your internet connection.")
        st.stop()

    with st.spinner("Computing 15 technical indicators..."):
        feat_df = compute_technical_features(raw_df, ticker)

    if feat_df.empty or len(feat_df) < 5:
        st.error("Not enough data after computing indicators.")
        st.stop()

    latest      = feat_df.iloc[-1]
    latest_date = latest['Date']
    X_latest    = pd.DataFrame([latest[feature_cols].values], columns=feature_cols)

    # ── Prediction ──
    prob_up   = float(model.predict_proba(X_latest)[0][1])
    prob_down = 1.0 - prob_up
    predicted = int(model.predict(X_latest)[0])   # uses 0.5 threshold from training

    css_class, icon, label, subtitle, sig_color, _ = classify_signal(prob_up, predicted)

    # ── Layout: Prediction | Indicators ──
    col_pred, col_ind = st.columns([1, 2])

    with col_pred:
        st.markdown(f"### Prediction for **{ticker}**")
        st.caption(f"Data through: **{latest_date.date()}**")
        next_day = latest_date + timedelta(days=1)
        # skip to Monday if weekend
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        st.caption(f"Predicting next trading day: **{next_day.strftime('%b %d, %Y')}**")

        # Signal card
        st.markdown(f"""
        <div class="{css_class}">
            <div style="font-size:3.2rem; margin-bottom:4px;">{icon}</div>
            <div style="font-size:1.6rem; font-weight:800; color:{sig_color};
                        letter-spacing:1px;">{label}</div>
            <div style="font-size:0.95rem; color:#ccc; margin-top:6px;">{subtitle}</div>
            <div style="font-size:1.05rem; margin-top:12px; color:#eee;">
                P(UP) = <b style="color:#4caf50;">{prob_up:.1%}</b>
                &nbsp;|&nbsp;
                P(DOWN) = <b style="color:#f44336;">{prob_down:.1%}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Probability gauge bar
        fig_gauge = plot_probability_gauge(prob_up, sig_color)
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close(fig_gauge)

        # Confidence metrics
        c1, c2 = st.columns(2)
        c1.metric("P(UP)",   f"{prob_up:.1%}",   delta=f"{prob_up-0.5:+.1%} vs 50%")
        c2.metric("P(DOWN)", f"{prob_down:.1%}", delta=f"{prob_down-0.5:+.1%} vs 50%",
                  delta_color="inverse")

        st.markdown("""
        <div class="disclaimer">
        ⚠️ <b>Research only — not financial advice.</b><br>
        This model achieves ~52% accuracy. Markets are noisy.
        Never trade based solely on ML predictions.
        </div>
        """, unsafe_allow_html=True)

    with col_ind:
        st.markdown("### 📊 Technical Indicator Snapshot")
        st.caption(f"As of {latest_date.date()}")

        # Indicator signal rows
        sigs = indicator_signals(latest)
        n = len(sigs)
        row1_cols = st.columns(min(n, 3))
        row2_cols = st.columns(min(max(n - 3, 0), 3)) if n > 3 else []

        for i, (ind_name, val, sig_text, is_bull) in enumerate(sigs):
            target_col = row1_cols[i] if i < 3 else row2_cols[i - 3]
            if is_bull is True:
                arrow, color = "↑", "#4caf50"
            elif is_bull is False:
                arrow, color = "↓", "#f44336"
            else:
                arrow, color = "→", "#ffc107"
            with target_col:
                st.markdown(f"""
                <div class="signal-bar-container">
                    <div class="signal-bar-label">{ind_name}</div>
                    <div style="font-size:1.3rem; font-weight:700; color:#fff;">{val}</div>
                    <div style="font-size:0.78rem; color:{color}; margin-top:2px;">
                        {arrow} {sig_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # All 15 features expander
        with st.expander("📋 View All 15 Features Used for Prediction"):
            feat_display = pd.DataFrame({
                'Feature': feature_cols,
                'Value':   [f"{X_latest[c].values[0]:.5f}" for c in feature_cols]
            })
            st.dataframe(feat_display, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Price Chart ──
    if show_chart:
        st.markdown("### 📉 Price Chart · SMA + Bollinger Bands + RSI")
        fig_chart = plot_price_chart(feat_df.tail(252), ticker)
        st.pyplot(fig_chart, use_container_width=True)
        plt.close(fig_chart)

    # ── SHAP ──
    if show_shap:
        st.markdown("---")
        st.markdown("### 🔍 SHAP Explainability — Why This Prediction?")

        tab1, tab2 = st.tabs(["📍 This Prediction (Local)", "🌐 Overall Model (Global)"])

        with tab1:
            st.markdown("""
            The waterfall below explains **this specific prediction**.
            - 🟥 Red bars → feature pushes toward **UP** prediction
            - 🟦 Blue bars → feature pushes toward **DOWN** prediction
            - **E[f(x)]** = average model output across training data (baseline)
            - **f(x)** = final prediction score for today's data
            """)
            with st.spinner("Computing SHAP values..."):
                fig_wf = plot_shap_waterfall(explainer, X_latest, feature_cols)
                st.pyplot(fig_wf, use_container_width=True)
                plt.close(fig_wf)

        with tab2:
            if importance_df is not None:
                st.markdown("""
                Global importance across all **6,030 training rows**.
                Higher = feature has more influence on predictions overall.
                """)
                fig_bar = plot_feature_bar(importance_df)
                st.pyplot(fig_bar, use_container_width=True)
                plt.close(fig_bar)

                display_df = importance_df[['rank','feature','mean_abs_shap','mean_signed_shap']].copy()
                display_df.columns = ['Rank','Feature','Mean |SHAP|','Signed SHAP']
                display_df['Mean |SHAP|']  = display_df['Mean |SHAP|'].map('{:.4f}'.format)
                display_df['Signed SHAP']  = display_df['Signed SHAP'].map('{:+.4f}'.format)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("Run Phase 6 to generate shap_feature_importance.csv.")

    # ── Ablation Table ──
    st.markdown("---")
    st.markdown("### 🧪 Ablation Study Results")
    ablation = pd.DataFrame({
        'Experiment': [
            '✅ XGBoost — Technical Only (Tuned)  ← Best',
            'XGBoost — Tech + Sentiment (6,030 rows)',
            'XGBoost — Tech + Sentiment (90 rows, real news only)',
            'Logistic Regression — Technical Only'
        ],
        'Accuracy':  [0.520, 0.519, 0.573, 0.505],
        'Precision': [0.525, 0.524, 0.093, 0.542],
        'Recall':    [0.842, 0.838, 0.200, 0.353],
        'F1 Score':  [0.646, 0.644, 0.127, 0.419],
        'ROC-AUC':   [0.522, 0.517, 0.518, 0.519]
    })
    st.dataframe(
        ablation.style.highlight_max(subset=['F1 Score', 'Recall'], color='#0f5132'),
        use_container_width=True, hide_index=True
    )
    st.caption("""
    **Key finding:** Technical indicators alone achieve best F1 (0.646).
    Adding sentiment from NewsAPI's 30-day window (1.3% real coverage) adds negligible value.
    In production with full historical news data, sentiment would contribute more significantly.
    """)

    # ── Footer ──
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#666; font-size:0.82rem;">
    Built by <b>Mehak</b> · Data Science Portfolio ·
    XGBoost + FinBERT + SHAP · Google Colab + VS Code + Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()