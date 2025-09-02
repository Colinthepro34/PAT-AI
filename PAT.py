"""
Streamlit Data Assistant
Single-file Streamlit app that provides a chat-like interface to upload a dataset
and run natural-language-ish prompts such as "mean of age", "plot histogram of salary",
"show head", "find mode of column X" etc. Conversation history persists and results
are rendered as chat messages (text, tables, matplotlib/plotly charts).

How to run:
1. Save this file as `streamlit_data_assistant.py`
2. Install requirements: `pip install -r requirements.txt` (see requirements below)
3. Run: `streamlit run streamlit_data_assistant.py`

Requirements (example):
streamlit
pandas
numpy
matplotlib
plotly
scikit-learn
openpyxl  # if you want to load xlsx

Notes:
- Prompt parsing is intentionally simple (keyword mapping + regex for column names).
- The app stores `st.session_state['df']` and `st.session_state['chat_history']` so results
  accumulate and do not disappear when you ask follow-up questions.
- You can extend the `ACTION_MAP` and `parse_prompt()` for more advanced NLU (spaCy etc.).
"""

from typing import Optional, Dict, Any, List
import re
import io

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.impute import SimpleImputer

# ---------------------- Helper: session state init ----------------------
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'chat_history' not in st.session_state:
    # each entry is dict: {'role': 'user'|'assistant', 'type':'text'|'table'|'plot', 'content': ...}
    st.session_state['chat_history'] = []

# ---------------------- Prompt parsing & action mapping ----------------------
# Basic mapping of synonyms to canonical actions
ACTION_MAP = {
    'mean': ['mean', 'average', 'avg'],
    'median': ['median'],
    'mode': ['mode'],
    'describe': ['describe', 'summary', 'summary statistics'],
    'head': ['head', 'show head', 'show first', 'first rows'],
    'tail': ['tail', 'last rows'],
    'dropna': ['dropna', 'drop na', 'drop missing', 'remove missing'],
    'fillna': ['fillna', 'fill missing', 'impute'],
    'histogram': ['histogram', 'hist', 'distribution'],
    'bar': ['bar chart', 'bar'],
    'scatter': ['scatter', 'scatter plot'],
    'count': ['count', 'value counts'],
    'corr': ['correlation', 'corr', 'correlations'],
}

# Inverse lookup for quick detection
INVERSE_ACTION = {}
for k, vs in ACTION_MAP.items():
    for v in vs:
        INVERSE_ACTION[v] = k


def detect_action(text: str) -> Optional[str]:
    text_low = text.lower()
    # look for known verbs/phrases
    for phrase, action in INVERSE_ACTION.items():
        if phrase in text_low:
            return action
    # fallback simple keywords
    for k in ACTION_MAP.keys():
        if k in text_low:
            return k
    return None


def extract_column_names(text: str, df: pd.DataFrame) -> List[str]:
    # naive approach: try to find exact column names mentioned, or words preceded by 'of' or 'for'
    if df is None:
        return []
    cols = list(df.columns.astype(str))
    found = []
    # check for exact column mentions (case-insensitive)
    for col in cols:
        pattern = re.compile(rf"\b{re.escape(col)}\b", flags=re.IGNORECASE)
        if pattern.search(text):
            found.append(col)
    if found:
        return found
    # heuristic: look for "of <word>" or "for <word>"
    m = re.findall(r"(?:of|for)\s+([A-Za-z0-9_\-]+)", text)
    if m:
        # map to closest column by lowercase match
        for token in m:
            for col in cols:
                if token.lower() == col.lower() or token.lower() in col.lower() or col.lower() in token.lower():
                    found.append(col)
    return list(dict.fromkeys(found))  # unique preserve order


# ---------------------- Action execution ----------------------

def run_action(action: str, text: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Return a dict with keys: 'type' ('text'|'table'|'plot'), and content.
    For text: content is str, for table: content is a DataFrame, for plot: content is a matplotlib Figure or plotly Figure
    """
    if df is None:
        return {'type': 'text', 'content': 'No dataset loaded. Please upload a CSV or Excel file first.'}

    cols = extract_column_names(text, df)

    try:
        if action == 'mean':
            if cols:
                res = {c: float(df[c].dropna().mean()) for c in cols}
                return {'type': 'text', 'content': f"Mean:\n{res}"}
            else:
                # mean for all numeric columns
                numeric = df.select_dtypes(include=[np.number])
                res = numeric.mean().to_dict()
                return {'type': 'table', 'content': pd.DataFrame(res, index=['mean']).T}

        if action == 'median':
            if cols:
                res = {c: float(df[c].dropna().median()) for c in cols}
                return {'type': 'text', 'content': f"Median:\n{res}"}
            else:
                numeric = df.select_dtypes(include=[np.number])
                res = numeric.median().to_dict()
                return {'type': 'table', 'content': pd.DataFrame(res, index=['median']).T}

        if action == 'mode':
            if cols:
                res = {c: df[c].mode().tolist() for c in cols}
                return {'type': 'text', 'content': f"Mode:\n{res}"}
            else:
                return {'type': 'text', 'content': 'Please specify a column for mode (modes may be multiple values).'}

        if action == 'describe':
            return {'type': 'table', 'content': df.describe(include='all')}

        if action == 'head':
            n = 5
            m = re.search(r"head\s*(\d+)", text.lower())
            if m:
                n = int(m.group(1))
            return {'type': 'table', 'content': df.head(n)}

        if action == 'tail':
            n = 5
            m = re.search(r"tail\s*(\d+)", text.lower())
            if m:
                n = int(m.group(1))
            return {'type': 'table', 'content': df.tail(n)}

        if action == 'dropna':
            before = df.shape
            new_df = df.dropna()
            st.session_state['df'] = new_df
            after = new_df.shape
            return {'type': 'text', 'content': f'Dropped NA rows. Before: {before}, After: {after}'}

        if action == 'fillna':
            # simple fill with mean for numeric, mode for non-numeric
            imputer = SimpleImputer(strategy='mean')
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)
            for col in df.select_dtypes(exclude=[np.number]).columns:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else '')
            st.session_state['df'] = df
            return {'type': 'text', 'content': 'Filled missing values: numeric -> mean, non-numeric -> mode (if available).'}

        if action in ('histogram', 'bar', 'scatter'):
            # need a column (or x and y for scatter)
            if action == 'scatter':
                # try to extract two columns
                cols = extract_column_names(text, df)
                if len(cols) >= 2:
                    x, y = cols[0], cols[1]
                    fig = px.scatter(df, x=x, y=y, title=f'Scatter: {x} vs {y}')
                    return {'type': 'plotly', 'content': fig}
                else:
                    return {'type': 'text', 'content': 'Scatter requires two columns. Example: "scatter of col1 and col2"'}

            # histogram or bar
            if not cols:
                return {'type': 'text', 'content': 'Please specify a column to plot (e.g. "histogram of salary").'}
            col = cols[0]
            # if categorical and asked for bar, show counts
            if action == 'bar' or (df[col].dtype == object and action == 'histogram'):
                counts = df[col].value_counts().reset_index()
                counts.columns = [col, 'count']
                fig = px.bar(counts, x=col, y='count', title=f'Bar: {col}')
                return {'type': 'plotly', 'content': fig}
            else:
                # histogram: numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    fig = px.histogram(df, x=col, title=f'Histogram: {col}')
                    return {'type': 'plotly', 'content': fig}
                else:
                    counts = df[col].value_counts().reset_index()
                    counts.columns = [col, 'count']
                    fig = px.bar(counts, x=col, y='count', title=f'Bar: {col}')
                    return {'type': 'plotly', 'content': fig}

        if action == 'count':
            if cols:
                res = {c: int(df[c].nunique()) for c in cols}
                return {'type': 'text', 'content': f'Unique counts:\n{res}'}
            else:
                return {'type': 'text', 'content': 'Please specify one or more columns to count unique values.'}

        if action == 'corr':
            corr = df.select_dtypes(include=[np.number]).corr()
            fig, ax = plt.subplots()
            cax = ax.matshow(corr)
            fig.colorbar(cax)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)
            return {'type': 'matplotlib', 'content': fig}

        return {'type': 'text', 'content': 'Sorry — I did not understand the request. Try: mean, median, mode, histogram, scatter, describe, head, dropna, fillna, corr.'}

    except Exception as e:
        return {'type': 'text', 'content': f'Error while executing action: {e}'}


# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title='PAT', layout='wide')

st.title('Upload the Data - Let us analyze for you')

# sidebar: upload and settings
with st.sidebar:
    st.header('Upload data')
    uploaded = st.file_uploader('Upload CSV or Excel', type=['csv', 'xlsx', 'xls'], accept_multiple_files=False)
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
    st.header('Settings')
    show_index = st.checkbox('Show dataframe index in tables', value=False)
    st.write('Tip: ask things like "mean of salary", "plot histogram of age", "show head 10"')
    st.write('Developed by: Colin')

# main chat area
chat_col, right_col = st.columns([3, 1])

with chat_col:
    # render history as chat messages
    for msg in st.session_state['chat_history']:
        if msg['role'] == 'user':
            st.chat_message('user').write(msg['content'])
        else:
            # assistant message: may contain text, table, plot
            with st.chat_message('assistant'):
                if msg['type'] == 'text':
                    st.markdown(msg['content'])
                elif msg['type'] == 'table':
                    st.dataframe(msg['content'])
                elif msg['type'] == 'plotly':
                    st.plotly_chart(msg['content'], use_container_width=True)
                elif msg['type'] == 'matplotlib':
                    st.pyplot(msg['content'])
                else:
                    st.write(msg.get('content'))

    # input area
    user_input = st.chat_input('Ask me to analyze your data (e.g. "mean of age")')
    if user_input:
        # append user message
        st.session_state['chat_history'].append({'role': 'user', 'content': user_input})
        # detect action
        action = detect_action(user_input)
        result = run_action(action, user_input, st.session_state['df'])
        # append assistant reply
        st.session_state['chat_history'].append({'role': 'assistant', 'type': result['type'], 'content': result['content']})
        # rerun to show new messages
        st.rerun()

with right_col:
    st.header('Current dataset')
    if st.session_state['df'] is None:
        st.write('No dataset loaded')
    else:
        st.write(f"{st.session_state['df'].shape[0]} rows × {st.session_state['df'].shape[1]} cols")
        if st.button('Show dataframe'):
            st.dataframe(st.session_state['df'])

# ---------------------- End ----------------------
