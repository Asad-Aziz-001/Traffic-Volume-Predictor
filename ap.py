import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.exceptions import InconsistentVersionWarning
import warnings
warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrafficIQ · Volume Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Variables ── */
:root {
  --bg:       #080c14;
  --surface:  #0e1623;
  --card:     #111927;
  --border:   rgba(251,191,36,0.15);
  --amber:    #fbbf24;
  --amber2:   #f59e0b;
  --cyan:     #22d3ee;
  --green:    #4ade80;
  --red:      #f87171;
  --text:     #f0f4ff;
  --muted:    #8899b4;
  --glow:     0 0 24px rgba(251,191,36,0.18);
}

/* ── Body & Background ── */
[data-testid="stAppViewContainer"] {
  background-color: var(--bg) !important;
  background-image:
    radial-gradient(ellipse 80% 50% at 50% -10%, rgba(251,191,36,0.07) 0%, transparent 70%),
    radial-gradient(ellipse 60% 40% at 80% 80%,  rgba(34,211,238,0.05) 0%, transparent 60%),
    repeating-linear-gradient(
      0deg,
      transparent,
      transparent 60px,
      rgba(251,191,36,0.025) 60px,
      rgba(251,191,36,0.025) 61px
    ),
    repeating-linear-gradient(
      90deg,
      transparent,
      transparent 60px,
      rgba(251,191,36,0.025) 60px,
      rgba(251,191,36,0.025) 61px
    );
  font-family: 'JetBrains Mono', monospace;
}

[data-testid="stHeader"], [data-testid="stToolbar"] {
  background: transparent !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a1020 0%, #0c1525 100%) !important;
  border-right: 1px solid rgba(251,191,36,0.12) !important;
}
section[data-testid="stSidebar"] * {
  color: var(--text) !important;
}
section[data-testid="stSidebar"] .stSelectbox label p,
section[data-testid="stSidebar"] label p {
  color: var(--amber) !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
}

/* ── Block Container ── */
.block-container {
  max-width: 1200px;
  padding: 1.5rem 2rem 4rem !important;
}

/* ── Hero Header ── */
.hero {
  padding: 1.8rem 0 1rem;
  border-bottom: 1px solid rgba(251,191,36,0.1);
  margin-bottom: 1.5rem;
  position: relative;
}
.hero-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  color: var(--amber);
  letter-spacing: 0.2em;
  text-transform: uppercase;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.hero-eyebrow::before {
  content: '';
  display: inline-block;
  width: 24px;
  height: 2px;
  background: var(--amber);
}
.hero h1 {
  font-family: 'Syne', sans-serif;
  font-size: 2.6rem;
  font-weight: 800;
  color: var(--text);
  margin: 0;
  line-height: 1.05;
  letter-spacing: -0.02em;
}
.hero h1 span {
  color: var(--amber);
}
.hero-sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  color: var(--muted);
  margin-top: 0.5rem;
  letter-spacing: 0.04em;
}
.hero-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  background: var(--green);
  border-radius: 50%;
  margin-right: 6px;
  animation: pulse 2s ease infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.5; transform: scale(0.7); }
}

/* ── Dark Cards ── */
.dark-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.6rem 1.8rem;
  margin-bottom: 1.2rem;
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s, box-shadow 0.3s;
}
.dark-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--amber), var(--cyan), transparent);
  opacity: 0.6;
}
.dark-card:hover {
  border-color: rgba(251,191,36,0.35);
  box-shadow: var(--glow);
}

/* ── Card Section Title ── */
.card-title {
  font-family: 'Syne', sans-serif;
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--amber);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 1.2rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

/* ── Widget Labels ── */
label[data-testid="stWidgetLabel"] p,
.stSlider label p,
.stSelectbox label p,
.stNumberInput label p {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.72rem !important;
  font-weight: 500 !important;
  color: #8899b4 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}

/* ── Inputs & Selects ── */
div[data-baseweb="select"] > div {
  background: #0a1020 !important;
  border: 1px solid rgba(251,191,36,0.2) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.88rem !important;
}
div[data-baseweb="select"] > div:focus-within {
  border-color: var(--amber) !important;
  box-shadow: 0 0 0 2px rgba(251,191,36,0.15) !important;
}
div[data-baseweb="input"] > div > input {
  background: #0a1020 !important;
  border: 1px solid rgba(251,191,36,0.2) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.88rem !important;
}
div[data-baseweb="input"] > div:focus-within > input {
  border-color: var(--amber) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div > div {
  background: var(--amber) !important;
}
[data-testid="stSlider"] > div > div > div {
  background: rgba(251,191,36,0.15) !important;
}

/* ── Predict Button ── */
div[data-testid="stButton"] > button {
  width: 100%;
  background: linear-gradient(135deg, #d97706 0%, #fbbf24 50%, #f59e0b 100%) !important;
  color: #080c14 !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.9rem 2rem !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 1rem !important;
  font-weight: 800 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  cursor: pointer !important;
  transition: transform 0.2s, box-shadow 0.2s !important;
  box-shadow: 0 4px 24px rgba(251,191,36,0.4) !important;
}
div[data-testid="stButton"] > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(251,191,36,0.55) !important;
}

/* ── Result Card ── */
.result-card {
  background: linear-gradient(135deg, #0f1e10 0%, #0d1a0e 100%);
  border: 1px solid rgba(74,222,128,0.3);
  border-radius: 20px;
  padding: 2rem 2.5rem;
  text-align: center;
  animation: slideUp 0.5s cubic-bezier(0.22, 1, 0.36, 1);
  position: relative;
  overflow: hidden;
  margin-bottom: 1.2rem;
}
.result-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, #4ade80, transparent);
}
@keyframes slideUp {
  from { transform: translateY(20px); opacity: 0; }
  to   { transform: translateY(0);    opacity: 1; }
}
.result-number {
  font-family: 'Syne', sans-serif;
  font-size: 4rem;
  font-weight: 800;
  color: #4ade80;
  letter-spacing: -0.03em;
  line-height: 1;
  text-shadow: 0 0 32px rgba(74,222,128,0.4);
}
.result-unit {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
  color: var(--muted);
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-top: 0.3rem;
}
.result-range {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  color: rgba(74,222,128,0.6);
  margin-top: 0.8rem;
}

/* ── Traffic Level Badge ── */
.traffic-badge {
  display: inline-block;
  padding: 0.3rem 1.2rem;
  border-radius: 100px;
  font-family: 'Syne', sans-serif;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-top: 0.8rem;
}
.badge-low    { background: rgba(74,222,128,0.15);  color: #4ade80;  border: 1px solid rgba(74,222,128,0.3); }
.badge-med    { background: rgba(251,191,36,0.15);  color: #fbbf24;  border: 1px solid rgba(251,191,36,0.3); }
.badge-high   { background: rgba(251,146,60,0.15);  color: #fb923c;  border: 1px solid rgba(251,146,60,0.3); }
.badge-severe { background: rgba(248,113,113,0.15); color: #f87171;  border: 1px solid rgba(248,113,113,0.3); }

/* ── KPI Strip ── */
.kpi-strip { display: flex; gap: 1rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.kpi-box {
  flex: 1;
  min-width: 130px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1rem 1.2rem;
  position: relative;
  overflow: hidden;
}
.kpi-box::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
  background: var(--amber);
  opacity: 0.4;
}
.kpi-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.12em;
}
.kpi-value {
  font-family: 'Syne', sans-serif;
  font-size: 1.5rem;
  font-weight: 800;
  color: var(--amber);
  margin-top: 0.2rem;
  letter-spacing: -0.02em;
}
.kpi-sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--muted);
  margin-top: 0.1rem;
}

/* ── Sidebar Preset Cards ── */
.preset-info {
  background: rgba(251,191,36,0.06);
  border: 1px solid rgba(251,191,36,0.15);
  border-radius: 10px;
  padding: 0.75rem 1rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  color: #c4d0e8;
  margin-bottom: 0.6rem;
  line-height: 1.6;
}
.preset-info strong { color: var(--amber); }

/* ── Divider ── */
hr { border: none; border-top: 1px solid rgba(251,191,36,0.1); margin: 1.2rem 0; }

/* ── Override streamlit text colors ── */
p, .stMarkdown p { color: var(--text) !important; }
h1,h2,h3,h4 { color: var(--text) !important; font-family: 'Syne', sans-serif !important; }
.stSlider [data-testid="stTickBar"] { display: none; }

/* ── Footer ── */
.footer {
  text-align: center;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  color: var(--muted);
  padding: 2rem 0 1rem;
  letter-spacing: 0.08em;
}
.footer span { color: var(--amber); }

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ─────────────────────────────────────────────────────────────
import gdown

file_id = "1SgUb5hJYpMsxSN-9f0IMa1pSBWUFJPh_"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists("model.pkl"):
    with st.spinner("Downloading model..."):
        gdown.download(url, "model.pkl", quiet=False)

model          = joblib.load("model.pkl")
scaler         = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800;
                color:#fbbf24; margin-bottom:0.3rem; letter-spacing:-0.01em;">
      🚦 TrafficIQ
    </div>
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
                color:#8899b4; letter-spacing:0.1em; margin-bottom:1.2rem;">
      METRO INTERSTATE PREDICTOR
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace; font-size:0.72rem; '
        'color:#fbbf24; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.6rem;">'
        '⚙ Traffic Scenario Presets</div>',
        unsafe_allow_html=True
    )

    preset = st.selectbox(
        "Select Scenario",
        ["Custom", "Morning Rush Hour", "Evening Rush Hour", "Night Time", "Weekend"],
        label_visibility="collapsed"
    )

    preset_meta = {
        "Morning Rush Hour":  ("07:00", "Weekday", "High congestion expected"),
        "Evening Rush Hour":  ("17:00", "Weekday", "Peak return traffic"),
        "Night Time":         ("02:00", "Any",     "Low volume, fast flow"),
        "Weekend":            ("12:00", "Weekend", "Moderate leisure traffic"),
        "Custom":             ("--:--", "Custom",  "Set your own parameters"),
    }
    label, day_type, note = preset_meta[preset]
    st.markdown(f"""
    <div class="preset-info">
      <strong>⏰ Time:</strong> {label}<br>
      <strong>📅 Type:</strong> {day_type}<br>
      <strong>📊 Note:</strong> {note}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="preset-info">
      <strong>ℹ️ About</strong><br>
      ML model trained on Metro Interstate I-94 dataset.<br>
      Predicts hourly vehicle volume based on time, weather & holiday factors.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem;
                color:#4a5568; text-align:center; padding-top:0.5rem;">
      Built by ASAD AZIZ 🚀
    </div>
    """, unsafe_allow_html=True)


# ─── Preset Defaults ────────────────────────────────────────────────────────
def preset_values(p):
    if p == "Morning Rush Hour": return 7, 1
    if p == "Evening Rush Hour": return 17, 0
    if p == "Night Time":        return 2, 0
    if p == "Weekend":           return 12, 1
    return 12, 0

hour_default, weekend_default = preset_values(preset)


# ─── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">AI-Powered Traffic Intelligence</div>
  <h1>Metro <span>Traffic</span> Volume Predictor</h1>
  <div class="hero-sub">
    <span class="hero-dot"></span>
    System Online &nbsp;·&nbsp; I-94 Interstate &nbsp;·&nbsp; Real-time ML Inference
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Input Sections ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🕒 Time Parameters</div>', unsafe_allow_html=True)

    hour       = st.slider("Hour of Day", 0, 23, hour_default)
    day        = st.slider("Day of Month", 1, 31, 15)
    month      = st.slider("Month", 1, 12, 6)
    dayofweek  = st.slider("Day of Week  (0 = Monday)", 0, 6, 2)
    is_weekend = st.selectbox("Day Type", [0, 1],
                               index=weekend_default,
                               format_func=lambda x: "🗓 Weekday" if x == 0 else "🏖 Weekend")
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌦 Weather Conditions</div>', unsafe_allow_html=True)

    temp        = st.number_input("Temperature (Kelvin)", 200.0, 350.0, 290.0, step=0.5)
    rain_1h     = st.number_input("Rain — last 1 hour (mm)", 0.0, 50.0, 0.0, step=0.1)
    snow_1h     = st.number_input("Snow — last 1 hour (mm)", 0.0, 50.0, 0.0, step=0.1)
    clouds_all  = st.slider("Cloud Coverage (%)", 0, 100, 50)
    weather_main = st.selectbox("Weather Condition",
                                 ["Clear", "Clouds", "Rain", "Snow", "Mist", "Fog"])
    st.markdown('</div>', unsafe_allow_html=True)

# Holiday — full width
st.markdown('<div class="dark-card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🗓 Holiday & Special Events</div>', unsafe_allow_html=True)
holiday = st.selectbox(
    "Select Holiday",
    ["None", "Christmas Day", "Thanksgiving Day", "New Years Day"],
    format_func=lambda x: f"🎄 {x}" if x == "Christmas Day"
                          else (f"🦃 {x}" if x == "Thanksgiving Day"
                          else (f"🎆 {x}" if x == "New Years Day"
                          else "— No Holiday"))
)
st.markdown('</div>', unsafe_allow_html=True)


# ─── Build Input ────────────────────────────────────────────────────────────
input_data = {
    "temp": temp, "rain_1h": rain_1h, "snow_1h": snow_1h,
    "clouds_all": clouds_all, "hour": hour, "day": day,
    "month": month, "dayofweek": dayofweek, "is_weekend": is_weekend
}
input_df = pd.DataFrame([input_data])
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0
if holiday != "None":
    holiday_col = f"holiday_{holiday}"
    if holiday_col in input_df.columns:
        input_df[holiday_col] = 1
weather_col = f"weather_main_{weather_main}"
if weather_col in input_df.columns:
    input_df[weather_col] = 1
input_df = input_df[feature_columns]


# ─── Predict Button ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🚗  RUN TRAFFIC PREDICTION")


# ─── Results ────────────────────────────────────────────────────────────────
if predict_clicked:
    input_scaled_arr = scaler.transform(input_df)
    input_scaled     = pd.DataFrame(input_scaled_arr, columns=feature_columns)
    prediction       = model.predict(input_scaled)[0]
    conf_low     = int(prediction * 0.90)
    conf_high    = int(prediction * 1.10)
    pred_int     = int(prediction)

    # Traffic level
    if pred_int < 1500:
        level, badge_cls, level_icon = "Low Traffic",    "badge-low",    "🟢"
    elif pred_int < 3000:
        level, badge_cls, level_icon = "Moderate",       "badge-med",    "🟡"
    elif pred_int < 4500:
        level, badge_cls, level_icon = "Heavy Traffic",  "badge-high",   "🟠"
    else:
        level, badge_cls, level_icon = "Severe Congestion", "badge-severe", "🔴"

    temp_c = round(temp - 273.15, 1)

    # KPI strip
    st.markdown(f"""
    <div class="kpi-strip">
      <div class="kpi-box">
        <div class="kpi-label">Predicted Volume</div>
        <div class="kpi-value">{pred_int:,}</div>
        <div class="kpi-sub">vehicles / hour</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">Confidence Band</div>
        <div class="kpi-value">{conf_low:,}–{conf_high:,}</div>
        <div class="kpi-sub">±10% range</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">Temperature</div>
        <div class="kpi-value">{temp_c}°C</div>
        <div class="kpi-sub">{temp}K</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">Cloud Cover</div>
        <div class="kpi-value">{clouds_all}%</div>
        <div class="kpi-sub">{weather_main}</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">Time Slot</div>
        <div class="kpi-value">{hour:02d}:00</div>
        <div class="kpi-sub">{"Weekend" if is_weekend else "Weekday"}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Main result card
    st.markdown(f"""
    <div class="result-card">
      <div class="result-number">{pred_int:,}</div>
      <div class="result-unit">Vehicles per Hour</div>
      <div><span class="traffic-badge {badge_cls}">{level_icon} {level}</span></div>
      <div class="result-range">Confidence: {conf_low:,} — {conf_high:,} vehicles/hr</div>
    </div>
    """, unsafe_allow_html=True)

    # Chart
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Key Input Features</div>', unsafe_allow_html=True)

    features  = ["Hour", "Is Weekend", "Temp (K)", "Clouds %", "Rain (mm)", "Snow (mm)"]
    values    = [hour, is_weekend * 23, temp - 273.15, clouds_all, rain_1h * 10, snow_1h * 10]
    colors    = ["#fbbf24", "#22d3ee", "#f87171", "#a78bfa", "#34d399", "#60a5fa"]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#111927")
    ax.set_facecolor("#0a1020")

    bars = ax.bar(features, values, color=colors, width=0.55,
                  edgecolor="none", zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha='center', va='bottom',
                color='#f0f4ff', fontsize=8.5, fontweight='600',
                fontfamily='monospace')

    ax.set_ylabel("Normalized Value", color="#8899b4", fontsize=9, labelpad=8, fontfamily='monospace')
    ax.tick_params(colors="#8899b4", labelsize=8.5)
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color=(0.98, 0.75, 0.14, 0.07), linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for label in ax.get_xticklabels():
        label.set_fontfamily('monospace')
        label.set_color('#8899b4')

    plt.tight_layout(pad=1.2)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)


# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  TRAFFICIQ &nbsp;·&nbsp; Metro Interstate I-94 &nbsp;·&nbsp;
  Built by <span>ASAD AZIZ</span> 🚀
</div>
""", unsafe_allow_html=True)
