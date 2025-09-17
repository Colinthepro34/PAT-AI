"""
Streamlit Data Assistant
Chat-like interface to upload a dataset and run natural-language-ish prompts.
"""

from typing import Optional, Dict, Any, List
import re
import io
import os

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import zipfile
import math
import plotly.io as pio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.impute import SimpleImputer

# ---------------------- Session State ----------------------
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'chat_started' not in st.session_state:
    st.session_state['chat_started'] = False


# ---------------------- Prompt Parsing ----------------------
ACTION_MAP = {
    'mean': ['mean', 'average', 'avg'],
    'median': ['median'],
    'mode': ['mode'],
    'describe': ['describe', 'summary', 'summary statistics'],
    'head': ['head', 'show head', 'show first', 'first rows'],
    'tail': ['tail', 'last rows'],
    'dropna': ['dropna', 'drop na', 'drop missing', 'remove missing'],
    'fillna': ['fillna', 'fill missing', 'impute'],
    'histogram': ['histogram', 'hist', 'distribution','Distribution of numerical variables '],
    'barchart': ['bar chart','bar','Frequency counts for categorical variables','Frequency counts for numerical variables'],
    'heatmap': ['Correlation between numerical variables','heatmap'],
    'scatter': ['scatter', 'scatter plot'],
    'count': ['count', 'value counts'],
    'corr': ['correlation', 'corr', 'correlations'],
    "rows": ["rows", "number of rows", "row count"],
    "columns": ["columns", "number of columns", "col count"],
    "dtypes": ["data types", "dtypes", "types", "column types"],
    "data_quality": ["data quality", "check quality", "missing values", "duplicates", "outliers", "clean data"],
    "feature_types": ["categorical","numerical"],
    "target_relationships":["target","relationship"],
    "categorical_counts":["categorical","count"],
    "distribution":["distribution"]
}

INVERSE_ACTION = {}
for k, vs in ACTION_MAP.items():
    for v in vs:
        INVERSE_ACTION[v] = k


def detect_actions(text: str) -> List[str]:
    """Return list of actions detected in user text."""
    text_low = text.lower()
    actions = []

    for phrase, action in INVERSE_ACTION.items():
        if phrase in text_low and action not in actions:
            actions.append(action)

    for k in ACTION_MAP.keys():
        if k in text_low and k not in actions:
            actions.append(k)

    return actions


def extract_column_names(text: str, df: pd.DataFrame) -> List[str]:
    if df is None:
        return []
    cols = list(df.columns.astype(str))
    found = []
    for col in cols:
        pattern = re.compile(rf"\b{re.escape(col)}\b", flags=re.IGNORECASE)
        if pattern.search(text):
            found.append(col)
    if found:
        return found
    m = re.findall(r"(?:of|for)\s+([A-Za-z0-9_\-]+)", text)
    if m:
        for token in m:
            for col in cols:
                if (
                    token.lower() == col.lower()
                    or token.lower() in col.lower()
                    or col.lower() in token.lower()
                ):
                    found.append(col)
    return list(dict.fromkeys(found))

# ---------------------- Safe Export ----------------------
def safe_export_fig(fig, filename: str, fmt: str = "png", scale: int = 2):
    """
    Safely export a Plotly figure. Falls back to HTML if Kaleido/system deps are missing.
    """
    try:
        if fmt == "png":
            return fig.to_image(format="png", scale=scale), filename
        elif fmt == "html":
            return fig.to_html(full_html=False, include_plotlyjs="cdn").encode("utf-8"), filename
    except Exception:
        # Fallback → export as HTML instead of PNG
        html_name = filename.replace(".png", ".html")
        return fig.to_html(full_html=False, include_plotlyjs="cdn").encode("utf-8"), html_name

# ---------------------- Column extraction helper ----------------------
def extract_column_names(text: str, df: pd.DataFrame) -> List[str]:
    """Try to find explicit column names in text, or tokens after 'of'/'for'."""
    if df is None:
        return []
    cols = list(df.columns.astype(str))
    found = []
    # exact column match (case-insensitive)
    for col in cols:
        pattern = re.compile(rf"\b{re.escape(col)}\b", flags=re.IGNORECASE)
        if pattern.search(text):
            found.append(col)
    if found:
        return list(dict.fromkeys(found))
    # heuristic: tokens after "of" or "for"
    m = re.findall(r"(?:of|for)\s+([A-Za-z0-9_\-]+)", text)
    if m:
        for token in m:
            for col in cols:
                if token.lower() == col.lower() or token.lower() in col.lower() or col.lower() in token.lower():
                    found.append(col)
    return list(dict.fromkeys(found))


# ---------------------- Single action handler (always returns list of dicts) ----------------------
def run_action(action: str, text: str, df: pd.DataFrame, cols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Execute a single action and return list of result blocks.
    Each block is a dict: {'type': 'text'|'table'|'plotly'|'matplotlib'|'data_quality'|'download', 'content': ...}
    """
    if df is None:
        return [{"type": "text", "content": "No dataset loaded. Please upload a CSV or Excel file first."}]

    results: List[Dict[str, Any]] = []
    if cols is None:
        cols = extract_column_names(text, df)

    try:
        # ---------------- Data quality ----------------
        if action == "data_quality":
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            dup_count = int(df.duplicated().sum())
            numeric = df.select_dtypes(include=[np.number])
            outlier_info = {}
            for c in numeric.columns:
                if numeric[c].std(ddof=0) == 0 or numeric[c].isna().all():
                    continue
                z = np.abs((numeric[c] - numeric[c].mean()) / numeric[c].std(ddof=0))
                out_cnt = int((z > 3).sum())
                if out_cnt > 0:
                    outlier_info[c] = out_cnt

            if missing.empty and dup_count == 0 and not outlier_info:
                results.append({"type": "text", "content": "✅ No missing values, duplicates, or outliers found."})
            else:
                results.append({
                    "type": "data_quality",
                    "content": {
                        "missing": missing.to_dict(),
                        "duplicates": dup_count,
                        "outliers": outlier_info
                    }
                })

            return results

        # ---------------- Dataset info ----------------
        if action == "rows":
            results.append({"type": "text", "content": f"The dataset has **{df.shape[0]} rows**."})
            return results
        if action == "columns":
            results.append({"type": "text", "content": f"The dataset has **{df.shape[1]} columns**."})
            return results
        if action == "dtypes":
            results.append({"type": "table", "content": df.dtypes.to_frame("dtype")})
            return results

        # ---------------- Statistics ----------------
        if action == "mean":
            numeric = df.select_dtypes(include=[np.number])
            results.append({"type": "table", "content": numeric.mean().to_frame("mean")})
            return results
        if action == "median":
            numeric = df.select_dtypes(include=[np.number])
            results.append({"type": "table", "content": numeric.median().to_frame("median")})
            return results
        if action == "mode":
            if cols:
                res = {c: df[c].mode().tolist() for c in cols}
                results.append({"type": "text", "content": f"Mode:\n{res}"})
            else:
                results.append({"type": "text", "content": "Please specify a column for mode."})
            return results
        if action == "describe":
            results.append({"type": "table", "content": df.describe(include="all")})
            return results

        # ---------------- Head/Tail ----------------
        if action == "head":
            n = 5
            m = re.search(r"head\s*(\d+)", text.lower())
            if m: n = int(m.group(1))
            results.append({"type": "table", "content": df.head(n)})
            return results
        if action == "tail":
            n = 5
            m = re.search(r"tail\s*(\d+)", text.lower())
            if m: n = int(m.group(1))
            results.append({"type": "table", "content": df.tail(n)})
            return results

        # ---------------- NA handling ----------------
        if action == "dropna":
            before = df.shape
            new_df = df.dropna()
            st.session_state["df"] = new_df
            after = new_df.shape
            results.append({"type": "text", "content": f"Dropped NA rows. Before: {before}, After: {after}"})
            return results
        if action == "fillna":
            imputer = SimpleImputer(strategy="mean")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            # apply only if any numeric cols exist
            if len(numeric_cols) > 0:
                df[numeric_cols] = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)
            for c in df.select_dtypes(exclude=[np.number]).columns:
                if not df[c].mode().empty:
                    df[c] = df[c].fillna(df[c].mode().iloc[0])
                else:
                    df[c] = df[c].fillna("")
            st.session_state["df"] = df
            results.append({"type": "text", "content": "Filled missing values: numeric → mean, non-numeric → mode."})
            return results

        # ---------------- Feature types ----------------
        if action == "feature_types":
            numerical = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
            result_df = pd.DataFrame({
                "Feature Type": ["Numerical", "Categorical"],
                "Columns": [", ".join(numerical) if numerical else "None",
                            ", ".join(categorical) if categorical else "None"]
            })
            results.append({"type": "table", "content": result_df})
            return results

        # ---------------- Distribution analysis ----------------
        if action == "distribution":
            figs: List[go.Figure] = []
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if "id" not in c.lower()]
            if not numeric_cols:
                results.append({"type": "text", "content": "No numeric columns found for distribution analysis."})
                return results

            skew_info = {}
            narrative_parts = []
            for c in numeric_cols:
                series = df[c].dropna()
                if series.empty:
                    skew_val = 0.0
                    kurt = 0.0
                else:
                    skew_val = float(series.skew())
                    kurt = float(series.kurtosis())

                if skew_val < -0.5:
                    skewness = "Left-skewed (most values high, with some low outliers)"
                elif skew_val > 0.5:
                    skewness = "Right-skewed (most values low, with some high outliers)"
                else:
                    skewness = "Approximately symmetric"

                if kurt > 3:
                    tail_note = "with fat tails (occasional extreme values)"
                elif kurt < 3:
                    tail_note = "with light tails"
                else:
                    tail_note = ""

                skew_info[c] = {"skewness": skewness, "skew_value": skew_val, "kurtosis": kurt}
                narrative_parts.append(f"{c} → {skewness}" + (f", {tail_note}" if tail_note else ""))

                fig = px.histogram(df, x=c, title=f"Distribution: {c}")
                fig.update_traces(marker=dict(line=dict(width=1, color="black")))
                figs.append(fig)
                results.append({"type": "plotly", "content": fig})

            narrative_text = "### 📊 Distribution Analysis\n\n"
            narrative_text += "Here are the histograms for all numerical variables:\n\n"
            narrative_text += "\n\n".join(narrative_parts)
            results.append({"type": "text", "content": narrative_text})

            skew_df = pd.DataFrame(skew_info).T
            results.append({"type": "table", "content": skew_df})
            return results

        # ---------------- Target relationships ----------------
        if action == "target_relationships":
            # user may specify target column via extract_column_names or fallback to last column
            target_cols = cols if cols else []
            target = target_cols[0] if target_cols else df.columns[-1]
            if target not in df.columns:
                results.append({"type": "text", "content": f"Target column '{target}' not found."})
                return results

            plots = []
            summary_lines = [f"### 🎯 Relationships with Target: **{target}**\n"]
            if pd.api.types.is_numeric_dtype(df[target]):
                numeric_preds = [c for c in df.select_dtypes(include=[np.number]).columns if c != target and "id" not in c.lower()]
                if not numeric_preds:
                    results.append({"type": "text", "content": "No numeric predictors available for target analysis."})
                    return results

                for pred in numeric_preds:
                    fig = px.scatter(df, x=pred, y=target, trendline="ols", title=f"{pred} vs {target}")
                    fig.update_traces(marker=dict(line=dict(width=1, color="black")))
                    plots.append(fig)
                    results.append({"type": "plotly", "content": fig})

                corrs = df[numeric_preds + [target]].corr()[target].drop(target)
                strong = corrs[abs(corrs) > 0.3].sort_values(key=lambda s: -abs(s))
                if not strong.empty:
                    summary_lines.append("Strong linear relationships with target:\n")
                    for col, val in strong.items():
                        direction = "positive" if val > 0 else "negative"
                        summary_lines.append(f"- **{col}** → {direction} correlation (r = {val:.2f})")
                else:
                    summary_lines.append("No strong linear correlations found with the numeric predictors.")
            else:
                # categorical target: numeric predictors -> boxplots; categorical predictors -> grouped bars
                numeric_preds = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
                cat_preds = [c for c in df.select_dtypes(exclude=[np.number]).columns if c != target]
                for pred in numeric_preds:
                    fig = px.box(df, x=target, y=pred, points="all", title=f"{pred} distribution by {target}")
                    fig.update_traces(marker=dict(line=dict(width=1, color="black")))
                    plots.append(fig)
                    results.append({"type": "plotly", "content": fig})
                for pred in cat_preds:
                    cross = pd.crosstab(df[pred], df[target])
                    fig = px.bar(cross, barmode="group", title=f"{pred} vs {target}")
                    fig.update_traces(marker=dict(line=dict(width=1, color="black")))
                    plots.append(fig)
                    results.append({"type": "plotly", "content": fig})
                summary_lines.append("Boxplots and bar charts show distribution differences across target classes.")

            results.append({"type": "text", "content": "\n".join(summary_lines)})
            return results

        # ---------------- Correlation / Heatmap ----------------
        if action in ("correlation", "corr"):
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if "id" not in c.lower()]
            if len(numeric_cols) < 2:
                results.append({"type": "text", "content": "Not enough numeric columns for correlation analysis."})
                return results

            corr = df[numeric_cols].corr()
            # heatmap via px.imshow (do not use marker for heatmap)
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation Heatmap")
            fig.update_traces(colorbar=dict(title="Correlation"), selector=dict(type="heatmap"))
            fig.update_layout(margin=dict(l=60, r=30, t=60, b=60), template="plotly_white")
            results.append({"type": "plotly", "content": fig})

            narrative = ["### 🔗 Correlation Analysis\n"]
            high_corrs = []
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    val = float(corr.iloc[i, j])
                    if abs(val) > 0.7:
                        high_corrs.append((corr.columns[i], corr.columns[j], val))

            if high_corrs:
                narrative.append("Strong correlations detected:")
                for c1, c2, val in high_corrs:
                    direction = "positive" if val > 0 else "negative"
                    narrative.append(f"- **{c1}** and **{c2}** → {direction} correlation (r = {val:.2f})")
            else:
                narrative.append("No strong correlations (|r| > 0.7) found among numeric variables.")

            results.append({"type": "text", "content": "\n".join(narrative)})
            return results

        # ---------------- Categorical counts ----------------
        if action == "categorical_counts":
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if not cat_cols:
                results.append({"type": "text", "content": "No categorical variables found in the dataset."})
                return results

            plots = []
            summary_lines = ["### 📊 Frequency Counts for Categorical Variables\n"]
            for c in cat_cols:
                counts = df[c].value_counts().reset_index()
                counts.columns = [c, "count"]
                top_preview = counts.head(6).to_dict(orient="records")
                summary_lines.append(f"**{c}**: {top_preview} ...")
                fig = px.bar(counts, x=c, y="count", title=f"Frequency of {c}")
                fig.update_traces(marker=dict(line=dict(width=1, color="black")))
                plots.append(fig)
                results.append({"type": "plotly", "content": fig})

            results.append({"type": "text", "content": "\n".join(summary_lines)})
            return results

        # ---------------- Generic plotting block (histogram/bar/line/scatter/heatmap multi) ----------------
        if action in ("histogram", "bar", "line", "scatter", "heatmap"):
            figs = []
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if "id" not in c.lower()]
            target_cols = cols if cols else numeric_cols

            if action == "scatter":
                if len(target_cols) >= 2:
                    for i in range(len(target_cols) - 1):
                        x, y = target_cols[i], target_cols[i + 1]
                        fig = px.scatter(df, x=x, y=y, title=f"Scatter: {x} vs {y}")
                        fig.update_traces(marker=dict(line=dict(width=1, color="black")))
                        figs.append(fig)
                        results.append({"type": "plotly", "content": fig})
                else:
                    results.append({"type": "text", "content": "Scatter requires at least 2 numerical columns."})
                    return results

            elif action == "line":
                if len(target_cols) >= 2:
                    for i in range(len(target_cols) - 1):
                        x, y = target_cols[i], target_cols[i + 1]
                        fig = px.line(df, x=x, y=y, title=f"Line: {x} vs {y}")
                        fig.update_traces(line=dict(width=2), marker=dict(line=dict(width=1, color="black")))
                        figs.append(fig)
                        results.append({"type": "plotly", "content": fig})
                else:
                    results.append({"type": "text", "content": "Line requires at least 2 numerical columns."})
                    return results

            elif action == "heatmap":
                numeric = df.select_dtypes(include=[np.number])
                if numeric.shape[1] < 2:
                    results.append({"type": "text", "content": "Not enough numeric columns for heatmap."})
                    return results
                corr = numeric.corr()
                fig = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap")
                fig.update_traces(colorbar=dict(title="Correlation"), selector=dict(type="heatmap"))
                fig.update_layout(template="plotly_white")
                figs.append(fig)
                results.append({"type": "plotly", "content": fig})

            else:  # histogram or bar
                for c in target_cols:
                    if action == "bar" or (df[c].dtype == object and action == "histogram"):
                        counts = df[c].value_counts().reset_index()
                        counts.columns = [c, "count"]
                        fig = px.bar(counts, x=c, y="count", title=f"Bar: {c}")
                        fig.update_traces(marker=dict(line=dict(width=1, color="black")))
                    else:
                        if pd.api.types.is_numeric_dtype(df[c]):
                            fig = px.histogram(df, x=c, title=f"Histogram: {c}")
                            fig.update_traces(marker=dict(line=dict(width=1, color="black")))
                        else:
                            counts = df[c].value_counts().reset_index()
                            counts.columns = [c, "count"]
                            fig = px.bar(counts, x=c, y="count", title=f"Bar: {c}")
                            fig.update_traces(marker=dict(line=dict(width=1, color="black")))
                    figs.append(fig)
                    results.append({"type": "plotly", "content": fig})
                    return results

        # ---------------- Unique counts ----------------
        if action == "count":
            if cols:
                res = {c: int(df[c].nunique()) for c in cols}
                results.append({"type": "text", "content": f"Unique counts:\n{res}"})
            else:
                results.append({"type": "text", "content": "Please specify columns to count unique values."})
            return results

        # ---------------- Legacy corr (matplotlib) ----------------
        if action == "corr":
            corr = df.select_dtypes(include=[np.number]).corr()
            fig, ax = plt.subplots(figsize=(6, 6))
            cax = ax.matshow(corr)
            fig.colorbar(cax)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)
            results.append({"type": "matplotlib", "content": fig})
            return results

        # ---------------- Unknown ----------------
        results.append({"type": "text", "content": "Sorry — I did not understand the request."})
        return results

    except Exception as e:
        return [{"type": "text", "content": f"Error: {e}"}]


# ---------------------- Multi-action wrapper ----------------------
def run_actions(actions: List[str], text: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Call run_action for each action and flatten results.
    Defensive: if a run_action returns a dict it will be wrapped into list.
    """
    all_results: List[Dict[str, Any]] = []
    for action in actions:
        try:
            action_results = run_action(action, text, df)
            if isinstance(action_results, dict):
                all_results.append(action_results)
            elif isinstance(action_results, list):
                all_results.extend(action_results)
            else:
                all_results.append({"type": "text", "content": f"Unexpected result type from {action}"})
        except Exception as e:
            all_results.append({"type": "text", "content": f"Error in {action}: {e}"})
    return all_results
# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title='PAT', layout='wide')

# Global CSS
# Global CSS for a ChatGPT-style input area
st.markdown("""
<style>
body {
    font-family: "Inter", sans-serif;
    background-color: #0f172a;
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1e293b;
    color: white;
}

/* Ensure sidebar text stays white */
[data-testid="stSidebar"] .stSidebar-header, 
[data-testid="stSidebar"] .stSidebar-content, 
[data-testid="stSidebar"] .stText, 
[data-testid="stSidebar"] .stMarkdown {
    color: white !important;
}

/* Custom styling for the file uploader button */
[data-testid="stFileUploader"] button {
    background-color: #e11d48 !important;
    color: white !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 5px !important;
    font-size: 16px !important;
}

/* Chat input area (like ChatGPT) */
div[data-testid="stChatInput"] div[role="textbox"],
div[data-testid="stTextArea"] textarea,
textarea[aria-label="Chat input"],
div[role="textbox"] {
    font-size: 18px !important;
    line-height: 1.4 !important;
    padding: 12px 20px !important;
    background-color: #2a3747 !important;  /* Dark background */
    color: white !important;  /* White text */
    border: 2px solid #4b6b8d !important;  /* Subtle border */
    border-radius: 20px !important;  /* Rounded corners */
    width: 100% !important;
    box-sizing: border-box;
    min-height: 80px !important;
    max-height: 180px !important;
    resize: none !important;
}

/* Focus effect */
div[data-testid="stChatInput"] div[role="textbox"]:focus {
    outline: none !important;
    border: 2px solid #2563eb !important;  /* Blue outline on focus */
}

/* Placeholder text style */
div[data-testid="stChatInput"] div[role="textbox"]::placeholder {
    color: #a0aec0 !important;  /* Light gray */
    opacity: 1 !important;
}

/* Chat bubbles */
.stChatMessage {
    border-radius: 12px;
    padding: 8px 12px;
    margin: 4px 0;
}
.stChatMessage[data-testid="stChatMessage-user"] {
    background-color: #2563eb;
    color: white;
}
.stChatMessage[data-testid="stChatMessage-assistant"] {
    background-color: #334155;
    color: white;
}

/* Send button */
button[data-testid="stChatSendButton"],
button[aria-label="Send"],
button[title="Send"] {
    padding: 8px 14px !important;
    font-size: 15px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.header('📂 Upload data')
    uploaded = st.file_uploader('Upload CSV or Excel', type=['csv', 'xlsx', 'xls'])
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state['df'] = df
            st.success(f'Loaded {uploaded.name} — {df.shape[0]} rows, {df.shape[1]} cols')
        except Exception as e:
            st.error(f'Could not load file: {e}')

    st.markdown('---')
    st.header('⚙️ Settings')
    show_index = st.checkbox('Show dataframe index in tables', value=False)
    st.caption('Developed by: Colin')


# ---------------------- Main Chat ----------------------
chat_col, right_col = st.columns([3, 1])

with chat_col:
    if not st.session_state["chat_started"]:
        st.markdown("## 👋 Hi, I'm **PAT**, your Data Analyst.")
        st.markdown("I can refine your data, analyze it, and create visuals.")
        st.image("/workspaces/PAT-AI/Made with insMind-IMG_20250915_063635 (1).png", width=600)
    else:
        for msg in st.session_state['chat_history']:
            if msg['role'] == 'user':
                st.chat_message('user').write(msg['content'])
            else:
                with st.chat_message('assistant'):
                    if msg['type'] == 'text':
                        st.markdown(msg['content'])
                    elif msg['type'] == 'table':
                        st.dataframe(msg['content'])
                    elif msg['type'] == 'plotly':
                        st.plotly_chart(msg['content'], use_container_width=True)
                    elif msg['type'] == 'matplotlib':
                        st.pyplot(msg['content'])
                    elif msg['type'] == 'data_quality':
                        issues = msg['content']
                        st.markdown("### 🧹 Data Quality Report")

                        # Missing values
                        if issues["missing"]:
                            st.markdown(f"⚠️ Missing values found: {issues['missing']}")
                            if st.button("Remove Missing Values", key=f"remove-missing-{len(st.session_state['chat_history'])}"):
                                st.session_state["df"] = st.session_state["df"].dropna()
                                st.success("Removed missing values!")

                        # Duplicates
                        if issues["duplicates"] > 0:
                            st.markdown(f"⚠️ Found {issues['duplicates']} duplicate rows")
                            if st.button("Remove Duplicates", key=f"remove-duplicates-{len(st.session_state['chat_history'])}"):
                                st.session_state["df"] = st.session_state["df"].drop_duplicates()
                                st.success("Removed duplicates!")

                        # Outliers
                        if issues["outliers"]:
                            st.markdown(f"⚠️ Outliers detected: {issues['outliers']}")
                            if st.button("Remove Outliers", key=f"remove-outliers-{len(st.session_state['chat_history'])}"):
                                cleaned = st.session_state["df"].copy()
                                for col in issues["outliers"].keys():
                                    z_scores = np.abs((cleaned[col] - cleaned[col].mean()) / cleaned[col].std())
                                    cleaned = cleaned[z_scores <= 3]
                                st.session_state["df"] = cleaned
                                st.success("Removed outliers!")

                        # Download cleaned dataset
                        if st.session_state["df"] is not None:
                            csv = st.session_state["df"].to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "⬇️ Download Cleaned CSV",
                                csv,
                                "cleaned_dataset.csv",
                                "text/csv",
                                key=f"download-cleaned-{len(st.session_state['chat_history'])}",
                            )
                    else:
                        st.write(msg.get('content'))

    # Chat input always at the bottom
    user_input = st.chat_input("Ask me to analyze your data...")
if user_input:
    st.session_state["chat_started"] = True
    st.session_state["chat_history"].append(
        {"role": "user", "content": user_input}
    )
    
    actions = detect_actions(user_input)

    # Run all detected actions
    results = []
    for action in actions:
        action_results = run_action(action, user_input, st.session_state["df"])
        if isinstance(action_results, dict):
            results.append(action_results)
        elif isinstance(action_results, list):
            results.extend(action_results)

    # Append assistant results
    for result in results:
        st.session_state["chat_history"].append(
            {"role": "assistant", "type": result["type"], "content": result["content"]}
        )
    
    st.rerun()

with right_col:
    st.header('📊 Current dataset')
    if st.session_state['df'] is None:
        st.write('No dataset loaded')
    else:
        st.write(f"{st.session_state['df'].shape[0]} rows × {st.session_state['df'].shape[1]} cols")
        if st.button('Show dataframe'):
            st.dataframe(st.session_state['df'])
