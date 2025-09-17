"""
Streamlit Data Assistant
Chat-like interface to upload a dataset and run natural-language-ish prompts.
"""

from typing import Optional, Dict, Any, List
import re
import io
import os

import streamlit as st
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
    'heatmap': ['Correlation between numerical variables'],
    'scatter': ['scatter', 'scatter plot'],
    'count': ['count', 'value counts'],
    'corr': ['correlation', 'corr', 'correlations'],
    "rows": ["rows", "number of rows", "row count"],
    "columns": ["columns", "number of columns", "col count"],
    "dtypes": ["data types", "dtypes", "types", "column types"],
    "data_quality": ["data quality", "check quality", "missing values", "duplicates", "outliers", "clean data"],
    "feature_types": ["categorical","numerical"]
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


# ---------------------- Action Execution ----------------------
def run_action(action: str, text: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None:
        return [{
            "type": "text",
            "content": "No dataset loaded. Please upload a CSV or Excel file first.",
        }]

    results: List[Dict[str, Any]] = []
    cols = extract_column_names(text, df)

    try:
        if action == "data_quality":
            issues = []
            
            # Missing values
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if not missing.empty:
                issues.append(f"⚠️ Missing values found:\n{missing.to_dict()}")

            # Duplicates
            dup_count = df.duplicated().sum()
            if dup_count > 0:
                issues.append(f"⚠️ Found {dup_count} duplicate rows")

            # Outliers (Z-score > 3)
            numeric = df.select_dtypes(include=[np.number])
            outlier_info = {}
            for col in numeric.columns:
                z_scores = np.abs((numeric[col] - numeric[col].mean()) / numeric[col].std())
                outlier_count = (z_scores > 3).sum()
                if outlier_count > 0:
                    outlier_info[col] = int(outlier_count)
            if outlier_info:
                issues.append(f"⚠️ Outliers detected:\n{outlier_info}")

            if not issues:
                results.append({"type": "text", "content": "✅ No missing values, duplicates, or outliers found."})
            else:
                results.append({
                    "type": "data_quality",
                    "content": {
                        "missing": missing.to_dict(),
                        "duplicates": int(dup_count),
                        "outliers": outlier_info
                    }
                })

        elif action == "rows":
            results.append({"type": "text", "content": f"The dataset has **{df.shape[0]} rows**."})

        elif action == "columns":
            results.append({"type": "text", "content": f"The dataset has **{df.shape[1]} columns**."})

        elif action == "dtypes":
            results.append({"type": "table", "content": df.dtypes.to_frame("dtype")})

        elif action == "mean":
            numeric = df.select_dtypes(include=[np.number])
            results.append({"type": "table", "content": numeric.mean().to_frame("mean")})

        elif action == "median":
            if cols:
                res = {c: float(df[c].dropna().median()) for c in cols}
                results.append({"type": "text", "content": f"Median:\n{res}"})
            else:
                numeric = df.select_dtypes(include=[np.number])
                results.append({"type": "table", "content": pd.DataFrame(numeric.median().to_dict(), index=["median"]).T})

        elif action == "mode":
            if cols:
                res = {c: df[c].mode().tolist() for c in cols}
                results.append({"type": "text", "content": f"Mode:\n{res}"})
            else:
                results.append({"type": "text", "content": "Please specify a column for mode."})

        elif action == "describe":
            results.append({"type": "table", "content": df.describe(include="all")})

        elif action == "head":
            n = 5
            m = re.search(r"head\s*(\d+)", text.lower())
            if m: n = int(m.group(1))
            results.append({"type": "table", "content": df.head(n)})

        elif action == "tail":
            n = 5
            m = re.search(r"tail\s*(\d+)", text.lower())
            if m: n = int(m.group(1))
            results.append({"type": "table", "content": df.tail(n)})

        elif action == "dropna":
            before = df.shape
            new_df = df.dropna()
            # st.session_state["df"] = new_df
            after = new_df.shape
            results.append({"type": "text", "content": f"Dropped NA rows. Before: {before}, After: {after}"})

        elif action == "fillna":
            imputer = SimpleImputer(strategy="mean")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)
            for col in df.select_dtypes(exclude=[np.number]).columns:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")
            # st.session_state["df"] = df
            results.append({"type": "text", "content": "Filled missing values: numeric -> mean, non-numeric -> mode."})
        
        elif action == "feature_types":
            numerical = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()

            result_df = pd.DataFrame({
                "Feature Type": ["Numerical", "Categorical"],
                "Columns": [", ".join(numerical) if numerical else "None",
                            ", ".join(categorical) if categorical else "None"]
            })

            results.append({"type": "table", "content": result_df})

        elif action == "distribution":
            figs = []
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            # Exclude ID-like columns
            exclude_ids = [c for c in df.columns if "id" in c.lower()]
            numeric_cols = [c for c in numeric_cols if c not in exclude_ids]

            if not numeric_cols:
                results.append({"type": "text", "content": "No numeric columns found for distribution analysis."})
            else:
                skew_info = {}
                narrative_parts = []

                for col in numeric_cols:
                    # Skewness & kurtosis
                    skew_val = df[col].skew()
                    kurt_val = df[col].kurtosis()

                    if skew_val < -0.5:
                        skewness = "Left-skewed (most values high, with a few low outliers)"
                    elif skew_val > 0.5:
                        skewness = "Right-skewed (most values low, with a few high outliers)"
                    else:
                        skewness = "Approximately symmetric"

                    if kurt_val > 3:
                        tail_note = "with fat tails (occasional extreme values)"
                    elif kurt_val < 3:
                        tail_note = "with light tails (values more concentrated)"
                    else:
                        tail_note = ""

                    skew_info[col] = {
                        "skewness": skewness,
                        "skew_value": float(skew_val),
                        "kurtosis": float(kurt_val)
                    }

                    # Narrative text for each column
                    narrative_parts.append(f"**{col}** → {skewness} {tail_note}".strip())

                    # Histogram with black edges
                    fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                    fig.update_traces(marker_line_color="black", marker_line_width=1)
                    figs.append(fig)
                    results.append({"type": "plotly", "content": fig})

                # Narrative summary
                narrative_text = "### 📊 Distribution Analysis\n\n"
                narrative_text += "Here are the histograms for all numerical variables in your dataset:\n\n"
                narrative_text += "\n\n".join(narrative_parts)

                results.append({"type": "text", "content": narrative_text})

                # Skewness & kurtosis summary table
                skew_df = pd.DataFrame(skew_info).T
                results.append({"type": "table", "content": skew_df})

                # Download all histograms as ZIP
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for i, fig in enumerate(figs, start=1):
                        img_bytes = fig.to_image(format="png", scale=2)
                        zf.writestr(f"distribution_{i}.png", img_bytes)

                results.append({
                    "type": "download",
                    "content": {
                        "file": zip_buffer.getvalue(),
                        "filename": "distribution_histograms.zip",
                        "label": "📥 Download all distribution histograms"
                    }
                })

        elif action == "target_relationships":
            # Extract target column
            cols = extract_column_names(text, df)
            target = cols[0] if cols else df.columns[-1]  # default: last column

            if target not in df.columns:
                results.append({"type": "text", "content": f"Target column '{target}' not found in dataset."})
            else:
                target_dtype = df[target].dtype

                plots = []
                summary = [f"### 🎯 Relationships with Target: **{target}**\n"]

                if pd.api.types.is_numeric_dtype(target_dtype):
                    # Numeric target
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    numeric_cols = [c for c in numeric_cols if c != target and "id" not in c.lower()]

                    for col in numeric_cols:
                        fig = px.scatter(df, x=col, y=target, trendline="ols",
                                         title=f"{col} vs {target}")
                        fig.update_traces(marker=dict(line=dict(width=1, color="black")))
                        plots.append(fig)

                    # Correlation summary
                    corrs = df[numeric_cols + [target]].corr()[target].drop(target)
                    top_corrs = corrs[abs(corrs) > 0.3].sort_values(ascending=False)
                    if not top_corrs.empty:
                        summary.append("Strong correlations with target:\n")
                        for col, val in top_corrs.items():
                            direction = "positive" if val > 0 else "negative"
                            summary.append(f"- **{col}** → {direction} correlation (r = {val:.2f})")
                    else:
                        summary.append("No strong linear correlations found with target.")

                else:
                    # Categorical target
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
                    cat_cols = [c for c in cat_cols if c != target]

                    # Boxplots for numeric vs target
                    for col in numeric_cols:
                        fig = px.box(df, x=target, y=col, points="all",
                                     title=f"{col} distribution by {target}")
                        plots.append(fig)

                    # Bar plots for categorical vs target
                    for col in cat_cols:
                        cross = pd.crosstab(df[col], df[target])
                        fig = px.bar(cross, barmode="group",
                                     title=f"{col} vs {target}")
                        plots.append(fig)

                    summary.append("Boxplots and bar plots show distribution differences across target classes.")

                # Add plots to results
                for fig in plots:
                    results.append({"type": "plotly", "content": fig})

                # Add summary
                results.append({"type": "text", "content": "\n".join(summary)})

                # Download all plots as images
                if plots:
                    images = []
                    for i, fig in enumerate(plots):
                        images.append(fig.to_image(format="png", scale=2))
                    results.append({
                        "type": "download",
                        "content": {
                            "file": b"".join(images),
                            "filename": f"{target}_relationships.png",
                            "label": f"📥 Download {target} Relationship Plots"
                        }
                    })

        elif action == "correlation":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_ids = [c for c in df.columns if "id" in c.lower()]
            numeric_cols = [c for c in numeric_cols if c not in exclude_ids]

            if len(numeric_cols) < 2:
                results.append({"type": "text", "content": "Not enough numeric columns for correlation analysis."})
            else:
                corr = df[numeric_cols].corr()

                # Heatmap with black edges
                fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap of Numerical Features",
                )
                fig.update_traces(marker_line_color="black", selector=dict(type="heatmap"))
                results.append({"type": "plotly", "content": fig})

                # Narrative summary of high correlations
                narrative = ["### 🔗 Correlation Analysis\n"]
                high_corrs = []
                for i in range(len(corr.columns)):
                    for j in range(i + 1, len(corr.columns)):
                        val = corr.iloc[i, j]
                        if abs(val) > 0.7:
                            high_corrs.append((corr.columns[i], corr.columns[j], val))

                if high_corrs:
                    narrative.append("Strong correlations detected:\n")
                    for c1, c2, val in high_corrs:
                        direction = "positive" if val > 0 else "negative"
                        narrative.append(f"- **{c1}** and **{c2}** → {direction} correlation (r = {val:.2f})")
                else:
                    narrative.append("No strong correlations (|r| > 0.7) found among numeric variables.")

                results.append({"type": "text", "content": "\n".join(narrative)})

                # Download heatmap
                img_bytes = fig.to_image(format="png", scale=2)
                results.append({
                    "type": "download",
                    "content": {
                        "file": img_bytes,
                        "filename": "correlation_heatmap.png",
                        "label": "📥 Download Correlation Heatmap"
                    }
                })

        elif action == "categorical_counts":
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

            if not cat_cols:
                results.append({"type": "text", "content": "No categorical variables found in the dataset."})
            else:
                summary = ["### 📊 Frequency Counts for Categorical Variables\n"]
                plots = []

                for col in cat_cols:
                    counts = df[col].value_counts().reset_index()
                    counts.columns = [col, "count"]

                    # Add text summary
                    summary.append(f"**{col}**:\n{counts.head(5).to_dict(orient='records')} ...")

                    # Add bar plot
                    fig = px.bar(counts, x=col, y="count", title=f"Frequency of {col}")
                    fig.update_traces(marker=dict(line=dict(width=1, color="black")))
                    plots.append(fig)

                # Add plots to results
                for fig in plots:
                    results.append({"type": "plotly", "content": fig})

                # Add summary text
                results.append({"type": "text", "content": "\n".join(summary)})

                # Download all frequency plots (optional, like dashboard)
                if plots:
                    images = []
                    for fig in plots:
                        images.append(fig.to_image(format="png", scale=2))
                    results.append({
                        "type": "download",
                        "content": {
                            "file": b"".join(images),
                            "filename": "categorical_counts.png",
                            "label": "📥 Download All Frequency Plots"
                        }
                    })

        elif action in ("histogram", "bar", "line", "scatter", "heatmap"):
            figs = []  # collect figures for dashboard + download

            # Detect numeric and categorical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

            # Exclude ID-like columns
            exclude_ids = [c for c in df.columns if "id" in c.lower()]
            numeric_cols = [c for c in numeric_cols if c not in exclude_ids]

            # If user specified columns, use them; otherwise auto-detect
            target_cols = cols if cols else numeric_cols

            # Scatter (needs 2 cols)
            if action == "scatter":
                if len(target_cols) >= 2:
                    for i in range(len(target_cols) - 1):
                        x, y = target_cols[i], target_cols[i + 1]
                        fig = px.scatter(df, x=x, y=y, title=f"Scatter: {x} vs {y}")
                        fig.update_traces(marker=dict(line=dict(color="black", width=1)))
                        figs.append(fig)
                        results.append({"type": "plotly", "content": fig})
                else:
                    results.append({"type": "text", "content": "Scatter requires at least 2 numerical columns."})

            # Line (needs 2 cols)
            elif action == "line":
                if len(target_cols) >= 2:
                    for i in range(len(target_cols) - 1):
                        x, y = target_cols[i], target_cols[i + 1]
                        fig = px.line(df, x=x, y=y, title=f"Line: {x} vs {y}")
                        fig.update_traces(line=dict(color="blue", width=2),
                                          marker=dict(line=dict(color="black", width=1)))
                        figs.append(fig)
                        results.append({"type": "plotly", "content": fig})
                else:
                    results.append({"type": "text", "content": "Line requires at least 2 numerical columns."})

            # Heatmap (correlation of all numeric cols)
            elif action == "heatmap":
                if numeric_cols:
                    corr = df[numeric_cols].corr()
                    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                    fig.update_traces(marker_line_color="black", marker_line_width=1)
                    figs.append(fig)
                    results.append({"type": "plotly", "content": fig})
                else:
                    results.append({"type": "text", "content": "No numeric columns for heatmap."})

            # Histogram / Bar
            else:
                for col in target_cols:
                    if action == "bar" or (df[col].dtype == object and action == "histogram"):
                        counts = df[col].value_counts().reset_index()
                        counts.columns = [col, "count"]
                        fig = px.bar(counts, x=col, y="count", title=f"Bar: {col}")
                        fig.update_traces(marker_line_color="black", marker_line_width=1)
                        figs.append(fig)
                        results.append({"type": "plotly", "content": fig})
                    else:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            fig = px.histogram(df, x=col, title=f"Histogram: {col}")
                            fig.update_traces(marker_line_color="black", marker_line_width=1)
                            figs.append(fig)
                            results.append({"type": "plotly", "content": fig})
                        else:
                            counts = df[col].value_counts().reset_index()
                            counts.columns = [col, "count"]
                            fig = px.bar(counts, x=col, y="count", title=f"Bar: {col}")
                            fig.update_traces(marker_line_color="black", marker_line_width=1)
                            figs.append(fig)
                            results.append({"type": "plotly", "content": fig})

            # ✅ Dashboard download (all figs into ZIP)
            if figs:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for i, fig in enumerate(figs, start=1):
                        img_bytes = fig.to_image(format="png", scale=2)
                        zf.writestr(f"{action}_{i}.png", img_bytes)

                results.append({
                    "type": "download",
                    "content": {
                        "file": zip_buffer.getvalue(),
                        "filename": f"{action}_plots_dashboard.zip",
                        "label": f"📥 Download all {action} plots (ZIP)"
                    }
                })

        elif action == "count":
            if cols:
                res = {c: int(df[c].nunique()) for c in cols}
                results.append({"type": "text", "content": f"Unique counts:\n{res}"})
            else:
                results.append({"type": "text", "content": "Please specify columns to count unique values."})

        elif action == "corr":
            corr = df.select_dtypes(include=[np.number]).corr()
            fig, ax = plt.subplots()
            cax = ax.matshow(corr)
            fig.colorbar(cax)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)
            results.append({"type": "matplotlib", "content": fig})

        else:
            results.append({"type": "text", "content": "Sorry — I did not understand the request."})

    except Exception as e:
        results.append({"type": "text", "content": f"Error: {e}"})

    return results

def run_actions(actions: List[str], text: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
    all_results: List[Dict[str, Any]] = []
    for action in actions:
        try:
            action_results = run_action(action, text, df)
            if isinstance(action_results, dict):
                all_results.append(action_results)
            elif isinstance(action_results, list):
                all_results.extend(action_results)
            # This logic seems misplaced, it will add 'feature_types' to actions in every loop iteration if the condition is met
            # I will remove it for now as it's likely an error.
            # if re.search(r"(categorical|numerical|feature types?)", text.lower()):
            #     actions.append("feature_types")

        except Exception as e:
            all_results.append({"type": "text", "content": f"Error in {action}: {e}"})
    return all_results

def run_actions(actions: List[str], text: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
    all_results: List[Dict[str, Any]] = []
    for action in actions:
        try:
            action_results = run_action(action, text, df)
            if isinstance(action_results, dict):
                all_results.append(action_results)
            elif isinstance(action_results, list):
                all_results.extend(action_results)
            if re.search(r"(categorical|numerical|feature types?)", text.lower()):
                actions.append("feature_types")

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
        results = run_actions(actions, user_input, st.session_state["df"])
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
