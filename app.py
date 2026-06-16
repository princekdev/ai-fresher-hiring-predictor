"""
╔══════════════════════════════════════════════════════════════╗
║   AI Fresher Hiring Selection Prediction — Streamlit App    ║
║   Author : AI/ML Engineer                                   ║
║   Fix    : Full WCAG-compliant contrast system              ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Hiring Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── WCAG-Compliant CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ══════════════════════════════════════════════════════
   DESIGN TOKENS  — Single source of truth for colours
   ══════════════════════════════════════════════════════ */
:root {
    --primary:    #1D4ED8;
    --primary-lt: #EFF6FF;
    --success:    #15803D;
    --success-lt: #DCFCE7;
    --danger:     #B91C1C;
    --danger-lt:  #FEE2E2;
    --warning:    #B45309;
    --purple:     #6D28D9;
    --teal:       #0F766E;

    /* Text — WCAG AA minimum 4.5:1 on white */
    --text-900:   #0F172A;   /* headings              */
    --text-700:   #1E293B;   /* body text             */
    --text-500:   #475569;   /* secondary / captions  */
    --text-white: #F8FAFC;   /* on dark backgrounds   */

    /* Surfaces */
    --bg-page:    #F1F5F9;
    --bg-card:    #FFFFFF;
    --bg-sidebar: #0F172A;

    --border:     #CBD5E1;
    --radius:     12px;
    --shadow:     0 2px 12px rgba(0,0,0,0.09);
}

/* ══════════════════════════════════════════════════════
   PAGE-LEVEL RESETS  — Stop Streamlit's own dark-theme
   colours bleeding onto a light page
   ══════════════════════════════════════════════════════ */
.stApp,
.stApp * {
    color: var(--text-700);   /* dark text everywhere by default */
}
.stApp { background: var(--bg-page) !important; }

/* Streamlit generated <p>, <span>, <div> text */
.stMarkdown p,
.stMarkdown li,
.stMarkdown td,
.stMarkdown th,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {
    color: var(--text-700) !important;
}

/* Headings in main content */
.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3,
.stMarkdown h4,
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] h4 {
    color: var(--text-900) !important;
}

/* Tables */
.stMarkdown table { color: var(--text-700) !important; }
.stMarkdown thead th {
    background: var(--primary-lt) !important;
    color: var(--text-900) !important;
    font-weight: 700 !important;
}
.stMarkdown tbody tr:nth-child(even) { background: #F8FAFC; }

/* ══════════════════════════════════════════════════════
   SIDEBAR  — dark background → white text, scoped tightly
   ══════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%) !important;
}
/* Every text node inside sidebar gets white */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #E2E8F0 !important;
}
section[data-testid="stSidebar"] .stSlider > label,
section[data-testid="stSidebar"] .stNumberInput > label,
section[data-testid="stSidebar"] .stSlider [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] .stNumberInput [data-testid="stWidgetLabel"] {
    color: #CBD5E1 !important;
}
/* Slider value bubble */
section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
    color: #FFFFFF !important;
}
/* Number input box */
section[data-testid="stSidebar"] input[type="number"] {
    color: var(--text-900) !important;
    background: #FFFFFF !important;
}
/* HR divider in sidebar */
section[data-testid="stSidebar"] hr { border-color: #334155 !important; }

/* ══════════════════════════════════════════════════════
   TABS
   ══════════════════════════════════════════════════════ */
.stTabs [role="tablist"] {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid var(--border);
}
.stTabs [role="tablist"] button {
    font-weight: 600;
    font-size: 0.92rem;
    color: var(--text-700) !important;
    border-radius: 8px;
}
.stTabs [role="tablist"] button[aria-selected="true"] {
    background: var(--primary) !important;
    color: #FFFFFF !important;
}
.stTabs [role="tablist"] button:hover {
    color: var(--primary) !important;
    background: var(--primary-lt) !important;
}
/* Tab panel content */
.stTabs [role="tabpanel"] * {
    color: var(--text-700);
}

/* ══════════════════════════════════════════════════════
   METRICS (st.metric)
   ══════════════════════════════════════════════════════ */
div[data-testid="metric-container"] {
    background: #FFFFFF;
    border-radius: var(--radius);
    padding: 14px 18px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
}
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] [data-testid="stMetricLabel"],
div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: var(--text-500) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"],
div[data-testid="metric-container"] [data-testid="stMetricValue"] div {
    color: var(--text-900) !important;
    font-weight: 800 !important;
    font-size: 1.6rem !important;
}

/* ══════════════════════════════════════════════════════
   DATAFRAME / TABLE
   ══════════════════════════════════════════════════════ */
.stDataFrame, .stDataFrame * { color: var(--text-700) !important; }
div[data-testid="stDataFrameResizable"] { border-radius: var(--radius); }

/* ══════════════════════════════════════════════════════
   EXPANDER
   ══════════════════════════════════════════════════════ */
.streamlit-expanderHeader {
    color: var(--text-900) !important;
    font-weight: 600 !important;
    background: #FFFFFF !important;
}
.streamlit-expanderContent {
    background: #FFFFFF !important;
    color: var(--text-700) !important;
}

/* ══════════════════════════════════════════════════════
   ALERTS  (st.info / st.success / st.warning / st.error)
   ══════════════════════════════════════════════════════ */
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] div {
    color: var(--text-700) !important;
}

/* ══════════════════════════════════════════════════════
   HERO BANNER  — dark blue bg → white text explicitly
   ══════════════════════════════════════════════════════ */
.hero-banner {
    background: linear-gradient(135deg, #1E40AF 0%, #1D4ED8 50%, #0369A1 100%);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 24px;
    box-shadow: 0 6px 30px rgba(29,78,216,0.35);
}
.hero-banner h1 {
    color: #FFFFFF !important;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 8px 0;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3);
}
.hero-banner p {
    color: #BFDBFE !important;
    font-size: 1rem;
    margin: 0;
    line-height: 1.6;
}

/* ══════════════════════════════════════════════════════
   METRIC CARDS  (custom HTML cards in model tab)
   ══════════════════════════════════════════════════════ */
.metric-grid { display: flex; gap: 14px; flex-wrap: wrap; margin: 20px 0 24px; }
.metric-card {
    background: #FFFFFF;
    border-radius: var(--radius);
    padding: 20px 18px;
    flex: 1;
    min-width: 130px;
    box-shadow: var(--shadow);
    border-top: 4px solid var(--primary);
    text-align: center;
    border: 1px solid var(--border);
    border-top: 4px solid var(--primary);
}
.metric-card .val {
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--primary);      /* explicit — never inherits bad value */
    display: block;
}
.metric-card .lbl {
    font-size: 0.75rem;
    color: var(--text-500);     /* explicit dark grey */
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-top: 5px;
    display: block;
    font-weight: 600;
}

/* ══════════════════════════════════════════════════════
   RESULT BOXES
   ══════════════════════════════════════════════════════ */
.result-selected {
    background: linear-gradient(135deg, #DCFCE7, #BBF7D0);
    border: 2px solid #15803D;
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
    margin: 16px 0;
}
.result-selected h2  { color: #14532D !important; font-size: 2rem; margin: 0; }
.result-selected p   { color: #15803D !important; font-size: 1.05rem; margin: 8px 0 0; }

.result-not-selected {
    background: linear-gradient(135deg, #FEE2E2, #FECACA);
    border: 2px solid #B91C1C;
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
    margin: 16px 0;
}
.result-not-selected h2 { color: #7F1D1D !important; font-size: 2rem; margin: 0; }
.result-not-selected p  { color: #B91C1C !important; font-size: 1.05rem; margin: 8px 0 0; }

/* ══════════════════════════════════════════════════════
   PROBABILITY BAR
   ══════════════════════════════════════════════════════ */
.prob-label {
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--text-900) !important;
    margin-bottom: 6px;
}
.prob-bar-wrap {
    background: #E2E8F0;
    border-radius: 999px;
    height: 24px;
    overflow: hidden;
    border: 1px solid var(--border);
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width .6s ease;
}

/* ══════════════════════════════════════════════════════
   SECTION DIVIDERS & LABELS
   ══════════════════════════════════════════════════════ */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--primary) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 20px 0 4px;
}

/* ══════════════════════════════════════════════════════
   FOOTER
   ══════════════════════════════════════════════════════ */
.footer {
    text-align: center;
    color: var(--text-500) !important;
    font-size: 0.8rem;
    margin-top: 40px;
    padding: 20px 0 10px;
    border-top: 1px solid var(--border);
}
.footer strong { color: var(--text-700) !important; }

/* ══════════════════════════════════════════════════════
   BUTTON
   ══════════════════════════════════════════════════════ */
.stButton > button[kind="primary"] {
    background: var(--primary) !important;
    color: #FFFFFF !important;
    font-weight: 700;
    border-radius: 8px;
    border: none;
}
.stButton > button[kind="primary"]:hover {
    background: #1E40AF !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# DATA LOADING & MODEL TRAINING  (cached)
# ══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    url = ("https://github.com/YBIFoundation/ProjectDataSet"
           "/raw/main/Fresher%20Hiring%20Selection.csv")
    df = pd.read_csv(url)
    return df.drop(columns=['Unnamed: 0'], errors='ignore')


@st.cache_resource(show_spinner=False)
def train_model(df):
    FEATURES = ['Age', 'CGPA', 'Aptitude_Score', 'Programming_Score',
                'SQL_Score', 'Communication_Score', 'Projects',
                'Certifications', 'Internship_Months',
                'Hackathon_Participation', 'Attendance']
    X = df[FEATURES]
    y = df['Selected']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, solver='lbfgs',
                               random_state=42, C=1.0,
                               class_weight='balanced')
    model.fit(X_train_s, y_train)
    y_pred       = model.predict(X_test_s)
    y_pred_proba = model.predict_proba(X_test_s)[:, 1]
    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'auc':       roc_auc_score(y_test, y_pred_proba),
        'cm':        confusion_matrix(y_test, y_pred),
        'fpr':       roc_curve(y_test, y_pred_proba)[0],
        'tpr':       roc_curve(y_test, y_pred_proba)[1],
    }
    coeff_df = pd.DataFrame({
        'Feature':     FEATURES,
        'Coefficient': model.coef_[0],
    }).sort_values('Coefficient', ascending=False)
    return model, scaler, FEATURES, metrics, coeff_df, X_test, y_test


with st.spinner("🔄 Loading dataset & training model…"):
    df = load_data()
    model, scaler, FEATURES, metrics, coeff_df, X_test, y_test = train_model(df)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR — INPUT FORM
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎯 Candidate Profile")
    st.markdown("---")
    st.markdown("### 📋 Academic Details")
    age  = st.slider("Age", 18, 30, 22, help="Candidate's current age")
    cgpa = st.slider("CGPA (out of 100)", 40, 100, 70,
                     help="Cumulative GPA scaled to 100")
    st.markdown("### 📊 Assessment Scores")
    apt  = st.slider("Aptitude Score",      30, 100, 70)
    prog = st.slider("Programming Score",   20, 100, 65)
    sql  = st.slider("SQL Score",           20, 100, 60)
    comm = st.slider("Communication Score", 20, 100, 70)
    st.markdown("### 🏆 Experience & Activities")
    projects   = st.number_input("Number of Projects",        0, 15, 2, step=1)
    certs      = st.number_input("Certifications Earned",     0, 20, 3, step=1)
    internship = st.number_input("Internship Months",         0, 24, 3, step=1)
    hackathons = st.number_input("Hackathon Participations",  0, 20, 2, step=1)
    attendance = st.slider("Attendance (%)", 50, 100, 85)
    st.markdown("---")
    predict_btn = st.button("🚀 Predict Selection",
                            use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-banner">
  <h1>🎯 AI Fresher Hiring Selection Predictor</h1>
  <p>Machine Learning–powered candidate screening &nbsp;·&nbsp;
     Logistic Regression &nbsp;·&nbsp;
     2,000 training records &nbsp;·&nbsp;
     97.5% accuracy</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════

tab_predict, tab_eda, tab_model, tab_about = st.tabs([
    "🔮 Prediction", "📊 Data Analysis",
    "⚙️ Model Performance", "📖 About"
])


# ─────────────────────────────────────────────
# TAB 1 — PREDICTION
# ─────────────────────────────────────────────

with tab_predict:
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.markdown("### 📋 Submitted Candidate Profile")
        profile_df = pd.DataFrame({
            "Feature": ["Age", "CGPA", "Aptitude Score", "Programming Score",
                        "SQL Score", "Communication Score", "Projects",
                        "Certifications", "Internship Months",
                        "Hackathon Participations", "Attendance (%)"],
            "Your Score": [age, cgpa, apt, prog, sql, comm,
                           projects, certs, internship, hackathons, attendance]
        })
        st.dataframe(profile_df, use_container_width=True,
                     hide_index=True, height=420)

    with col_right:
        st.markdown("### 🔮 Prediction Result")

        if predict_btn:
            input_arr    = np.array([[age, cgpa, apt, prog, sql, comm,
                                      projects, certs, internship,
                                      hackathons, attendance]])
            input_scaled = scaler.transform(input_arr)
            prediction   = model.predict(input_scaled)[0]
            probability  = model.predict_proba(input_scaled)[0][1]

            if prediction == 1:
                st.markdown("""
                <div class="result-selected">
                  <h2>✅ SELECTED</h2>
                  <p>This candidate is likely to be hired!</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-not-selected">
                  <h2>❌ NOT SELECTED</h2>
                  <p>This candidate is unlikely to be hired.</p>
                </div>""", unsafe_allow_html=True)

            # Probability bar
            pct      = int(probability * 100)
            bar_col  = "#15803D" if prediction == 1 else "#B91C1C"
            st.markdown(f'<p class="prob-label">Selection Probability: {pct}%</p>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="prob-bar-wrap">
              <div class="prob-bar-fill"
                   style="width:{pct}%; background:{bar_col};"></div>
            </div>""", unsafe_allow_html=True)

            st.markdown("---")

            conf_label = ("🟢 High Confidence"    if probability >= 0.85 else
                          "🟡 Moderate Confidence" if probability >= 0.60 else
                          "🔴 Low Confidence")

            c1, c2 = st.columns(2)
            c1.metric("Confidence Level",       conf_label)
            c2.metric("Rejection Probability",  f"{100 - pct}%")

            st.markdown("---")
            st.markdown("#### 💡 AI Recommendation")
            if prediction == 1 and probability >= 0.85:
                st.success("**Strong candidate — recommend for interview round.**")
            elif prediction == 1:
                st.info("**Decent candidate — consider for interview with caution.**")
            elif probability >= 0.35:
                st.warning("**Borderline candidate — may benefit from upskilling.**")
            else:
                st.error("**Profile does not meet current hiring criteria.**")

            if prediction == 0:
                st.markdown("#### 📈 Top Areas to Improve")
                merged = pd.DataFrame({
                    'Feature':     FEATURES,
                    'YourScore':   input_arr[0],
                    'Coefficient': model.coef_[0],
                })
                for _, row in merged.nlargest(3, 'Coefficient').iterrows():
                    st.markdown(f"- **{row['Feature'].replace('_',' ')}** "
                                f"(your score: {row['YourScore']:.0f})")
        else:
            st.info("👈 Fill in the candidate details in the sidebar "
                    "and click **Predict Selection**.")
            st.markdown("#### 📁 Dataset Preview")
            st.dataframe(df.head(8), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# TAB 2 — EDA
# ─────────────────────────────────────────────

with tab_eda:
    st.markdown("### 📊 Exploratory Data Analysis")

    total      = len(df)
    n_selected = int(df['Selected'].sum())
    n_not      = total - n_selected

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Candidates", f"{total:,}")
    col2.metric("✅ Selected",      f"{n_selected:,}  ({n_selected/total*100:.1f}%)")
    col3.metric("❌ Not Selected",  f"{n_not:,}  ({n_not/total*100:.1f}%)")

    st.markdown("---")

    with st.expander("📋 Descriptive Statistics", expanded=False):
        st.dataframe(df.describe().round(2), use_container_width=True)

    # ── Histograms ─────────────────────────────────────────
    st.markdown("#### 📈 Score Distributions")
    hist_cols  = ['CGPA', 'Aptitude_Score', 'Programming_Score',
                  'SQL_Score', 'Communication_Score']
    h_colors   = ['#1D4ED8', '#6D28D9', '#0F766E', '#B45309', '#B91C1C']

    fig_h, axes = plt.subplots(1, len(hist_cols), figsize=(22, 4.5))
    fig_h.patch.set_facecolor('#F8FAFC')
    for ax, feat, col in zip(axes, hist_cols, h_colors):
        data = df[feat]
        ax.hist(data, bins=20, color=col, alpha=0.85, edgecolor='white')
        ax.axvline(data.mean(), color='#111827', lw=2, ls='--',
                   label=f'Mean={data.mean():.1f}')
        ax.set_title(feat.replace('_', '\n'), fontsize=10,
                     fontweight='bold', color='#0F172A')
        ax.set_xlabel("Score", fontsize=9, color='#1E293B')
        ax.set_ylabel("Count", fontsize=9, color='#1E293B')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_facecolor('#FFFFFF')
        ax.tick_params(colors='#1E293B')
        for spine in ax.spines.values():
            spine.set_edgecolor('#CBD5E1')
    fig_h.tight_layout()
    st.pyplot(fig_h, use_container_width=True)
    plt.close()

    # ── Boxplots ───────────────────────────────────────────
    st.markdown("#### 📦 Selected vs Not Selected — Feature Comparison")
    box_feats = ['CGPA', 'Programming_Score', 'Communication_Score',
                 'Aptitude_Score', 'Attendance']

    fig_b, axes = plt.subplots(1, len(box_feats), figsize=(22, 5))
    fig_b.patch.set_facecolor('#F8FAFC')
    for ax, feat in zip(axes, box_feats):
        d0 = df[df['Selected'] == 0][feat]
        d1 = df[df['Selected'] == 1][feat]
        bp = ax.boxplot(
    [d0, d1],
    patch_artist=True,
    tick_labels=['Not\nSelected', 'Selected'],
    medianprops=dict(color='white', lw=2.5)
)
        bp['boxes'][0].set_facecolor('#DC2626')
        bp['boxes'][0].set_alpha(0.80)
        bp['boxes'][1].set_facecolor('#15803D')
        bp['boxes'][1].set_alpha(0.80)
        ax.set_title(feat.replace('_', '\n'), fontsize=10,
                     fontweight='bold', color='#0F172A')
        ax.tick_params(colors='#1E293B')
        ax.grid(axis='y', alpha=0.3)
        ax.set_facecolor('#FFFFFF')
        for spine in ax.spines.values():
            spine.set_edgecolor('#CBD5E1')
    legend_els = [
        mpatches.Patch(color='#DC2626', alpha=0.80, label='Not Selected'),
        mpatches.Patch(color='#15803D', alpha=0.80, label='Selected'),
    ]
    fig_b.legend(handles=legend_els, loc='upper right', fontsize=10,
                 facecolor='white', edgecolor='#CBD5E1')
    fig_b.tight_layout()
    st.pyplot(fig_b, use_container_width=True)
    plt.close()

    # ── Correlation Heatmap ────────────────────────────────
    st.markdown("#### 🌡️ Correlation Heatmap")
    fig_c, ax = plt.subplots(figsize=(12, 9))
    fig_c.patch.set_facecolor('#F8FAFC')
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor='#E2E8F0',
                ax=ax, annot_kws={"size": 8, "color": "#0F172A"})
    ax.set_title("Feature Correlation Matrix",
                 fontsize=13, fontweight='bold', pad=12, color='#0F172A')
    ax.tick_params(colors='#1E293B', labelsize=9)
    ax.set_facecolor('#FFFFFF')
    fig_c.tight_layout()
    st.pyplot(fig_c, use_container_width=True)
    plt.close()


# ─────────────────────────────────────────────
# TAB 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────

with tab_model:
    st.markdown("### ⚙️ Logistic Regression — Model Performance")

    m = metrics
    # Custom metric cards — explicit colours so no CSS inheritance surprises
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card" style="border-top-color:#1D4ED8;">
        <span class="val" style="color:#1D4ED8;">{m['accuracy']*100:.2f}%</span>
        <span class="lbl">Accuracy</span>
      </div>
      <div class="metric-card" style="border-top-color:#15803D;">
        <span class="val" style="color:#15803D;">{m['precision']*100:.2f}%</span>
        <span class="lbl">Precision</span>
      </div>
      <div class="metric-card" style="border-top-color:#6D28D9;">
        <span class="val" style="color:#6D28D9;">{m['recall']*100:.2f}%</span>
        <span class="lbl">Recall</span>
      </div>
      <div class="metric-card" style="border-top-color:#B45309;">
        <span class="val" style="color:#B45309;">{m['f1']*100:.2f}%</span>
        <span class="lbl">F1 Score</span>
      </div>
      <div class="metric-card" style="border-top-color:#0F766E;">
        <span class="val" style="color:#0F766E;">{m['auc']:.4f}</span>
        <span class="lbl">ROC-AUC</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_cm, col_roc = st.columns(2, gap="large")

    with col_cm:
        st.markdown("#### 🟦 Confusion Matrix")
        fig_cm, ax = plt.subplots(figsize=(6, 5))
        fig_cm.patch.set_facecolor('#F8FAFC')
        sns.heatmap(m['cm'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Selected', 'Selected'],
                    yticklabels=['Not Selected', 'Selected'],
                    linewidths=1, linecolor='white',
                    annot_kws={"size": 16, "weight": "bold",
                               "color": "#0F172A"}, ax=ax)
        ax.set_xlabel("Predicted", fontsize=11, color='#1E293B')
        ax.set_ylabel("Actual",    fontsize=11, color='#1E293B')
        ax.set_title("Confusion Matrix", fontsize=13,
                     fontweight='bold', color='#0F172A')
        ax.tick_params(colors='#1E293B')
        fig_cm.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)
        plt.close()

        tn, fp, fn, tp = m['cm'].ravel()
        st.markdown(f"""
| Label | Count |
|---|---|
| ✅ True Positive (correctly hired)   | **{tp}** |
| ✅ True Negative (correctly rejected) | **{tn}** |
| ⚠️ False Positive (wrongly hired)    | **{fp}** |
| ⚠️ False Negative (missed talent)    | **{fn}** |
        """)

    with col_roc:
        st.markdown("#### 📈 ROC Curve")
        fig_roc, ax = plt.subplots(figsize=(6, 5))
        fig_roc.patch.set_facecolor('#F8FAFC')
        ax.plot(m['fpr'], m['tpr'], color='#1D4ED8', lw=2.5,
                label=f"AUC = {m['auc']:.4f}")
        ax.fill_between(m['fpr'], m['tpr'], alpha=0.10, color='#1D4ED8')
        ax.plot([0, 1], [0, 1], color='#94A3B8', lw=1.5, ls='--',
                label='Random (AUC = 0.5)')
        ax.set_xlabel("False Positive Rate", fontsize=11, color='#1E293B')
        ax.set_ylabel("True Positive Rate",  fontsize=11, color='#1E293B')
        ax.set_title("ROC Curve", fontsize=13,
                     fontweight='bold', color='#0F172A')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#FFFFFF')
        ax.tick_params(colors='#1E293B')
        for spine in ax.spines.values():
            spine.set_edgecolor('#CBD5E1')
        fig_roc.tight_layout()
        st.pyplot(fig_roc, use_container_width=True)
        plt.close()

        st.info(f"**ROC-AUC = {m['auc']:.4f}** — Excellent discriminative ability. "
                "A perfect model = 1.0. Random guessing = 0.5. "
                "Our model is near-perfect!")

    # Feature Importance
    st.markdown("#### 🏆 Feature Importance (Logistic Regression Coefficients)")
    fig_fi, ax = plt.subplots(figsize=(11, 6))
    fig_fi.patch.set_facecolor('#F8FAFC')
    colors_fi = ['#15803D' if c > 0 else '#B91C1C'
                 for c in coeff_df['Coefficient']]
    bars = ax.barh(coeff_df['Feature'], coeff_df['Coefficient'],
                   color=colors_fi, alpha=0.88, edgecolor='white', height=0.65)
    ax.axvline(0, color='#334155', lw=1.2)
    for bar, val in zip(bars, coeff_df['Coefficient']):
        x_pos = val + 0.06 if val >= 0 else val - 0.06
        ha    = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{val:+.3f}', va='center', ha=ha,
                fontsize=9, color='#0F172A', fontweight='600')
    ax.set_xlabel("Coefficient Value", fontsize=11, color='#1E293B')
    ax.set_title("Feature Importance — Logistic Regression",
                 fontsize=13, fontweight='bold', color='#0F172A')
    ax.tick_params(colors='#1E293B')
    legend_els = [
        mpatches.Patch(color='#15803D', alpha=0.88,
                       label='Positive (+) = Boosts Selection'),
        mpatches.Patch(color='#B91C1C', alpha=0.88,
                       label='Negative (−) = Reduces Selection'),
    ]
    ax.legend(handles=legend_els, fontsize=9,
              facecolor='white', edgecolor='#CBD5E1')
    ax.grid(axis='x', alpha=0.3)
    ax.set_facecolor('#FFFFFF')
    for spine in ax.spines.values():
        spine.set_edgecolor('#CBD5E1')
    fig_fi.tight_layout()
    st.pyplot(fig_fi, use_container_width=True)
    plt.close()


# ─────────────────────────────────────────────
# TAB 4 — ABOUT
# ─────────────────────────────────────────────

with tab_about:
    st.markdown("### 📖 About This Project")

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("""
#### 🎯 Project Overview
This application uses **Logistic Regression** to predict whether an
engineering fresher candidate will be **selected** (1) or
**not selected** (0) during hiring.

#### 📁 Dataset
- **Source**: YBI Foundation Open Dataset
- **Records**: 2,000 candidate profiles
- **Features**: 11 academic & experiential attributes
- **Target**: Binary — Selected (1) or Not Selected (0)

#### 🛠️ Tech Stack
| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Library | Scikit-Learn |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |
        """)

    with col_r:
        st.markdown("""
#### ⚡ Model Performance Summary
| Metric | Score |
|---|---|
| Accuracy  | **97.50%** |
| Precision | **100.00%** |
| Recall    | **97.18%** |
| F1 Score  | **98.57%** |
| AUC       | **0.9996** |

#### ⚠️ Ethical Disclaimer
This tool is for **educational and demonstration purposes only**.

- AI predictions should **never replace** human judgment in hiring.
- Always conduct **bias audits** on AI hiring tools.
- Ensure **transparency** with candidates about AI usage.
- Provide candidates with the **right to appeal** AI decisions.
- Comply with **local labor and data protection laws**.
        """)

    st.markdown("---")
    st.markdown("""
#### 📐 Features Explained
| Feature | Description |
|---|---|
| Age | Candidate's age (18–30 range) |
| CGPA | Academic performance score (0–100 scale) |
| Aptitude Score | Quantitative & logical reasoning test score |
| Programming Score | Coding assessment score (any language) |
| SQL Score | Database query proficiency score |
| Communication Score | Verbal & written communication assessment |
| Projects | Number of self/academic projects completed |
| Certifications | Number of professional certifications earned |
| Internship Months | Duration of internship experience in months |
| Hackathon Participation | Number of hackathons participated in |
| Attendance | Academic attendance percentage |
    """)


# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
  🎯 <strong>AI Fresher Hiring Prediction</strong> &nbsp;|&nbsp;
  Built with Python, Scikit-Learn &amp; Streamlit &nbsp;|&nbsp;
  Model Accuracy: <strong>97.50%</strong> &nbsp;|&nbsp;
  ⚠️ For educational use only — not a substitute for human judgment
</div>
""", unsafe_allow_html=True)
