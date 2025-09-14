"""
Streamlit Data Assistant
Chat-like interface to upload a dataset and run natural-language-ish prompts.
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
    'histogram': ['histogram', 'hist', 'distribution'],
    'bar': ['bar chart', 'bar'],
    'scatter': ['scatter', 'scatter plot'],
    'count': ['count', 'value counts'],
    'corr': ['correlation', 'corr', 'correlations'],
}

INVERSE_ACTION = {}
for k, vs in ACTION_MAP.items():
    for v in vs:
        INVERSE_ACTION[v] = k


def detect_action(text: str) -> Optional[str]:
    text_low = text.lower()
    for phrase, action in INVERSE_ACTION.items():
        if phrase in text_low:
            return action
    for k in ACTION_MAP.keys():
        if k in text_low:
            return k
    return None


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
                if token.lower() == col.lower() or token.lower() in col.lower() or col.lower() in token.lower():
                    found.append(col)
    return list(dict.fromkeys(found))


# ---------------------- Action Execution ----------------------
def run_action(action: str, text: str, df: pd.DataFrame) -> Dict[str, Any]:
    if df is None:
        return {'type': 'text', 'content': 'No dataset loaded. Please upload a CSV or Excel file first.'}

    cols = extract_column_names(text, df)

    try:
        if action == 'mean':
            if cols:
                res = {c: float(df[c].dropna().mean()) for c in cols}
                return {'type': 'text', 'content': f"Mean:\n{res}"}
            else:
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
                return {'type': 'text', 'content': 'Please specify a column for mode.'}

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
            imputer = SimpleImputer(strategy='mean')
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)
            for col in df.select_dtypes(exclude=[np.number]).columns:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else '')
            st.session_state['df'] = df
            return {'type': 'text', 'content': 'Filled missing values: numeric -> mean, non-numeric -> mode.'}

        if action in ('histogram', 'bar', 'scatter'):
            if action == 'scatter':
                cols = extract_column_names(text, df)
                if len(cols) >= 2:
                    x, y = cols[0], cols[1]
                    fig = px.scatter(df, x=x, y=y, title=f'Scatter: {x} vs {y}')
                    return {'type': 'plotly', 'content': fig}
                else:
                    return {'type': 'text', 'content': 'Scatter requires two columns.'}

            if not cols:
                return {'type': 'text', 'content': 'Please specify a column to plot.'}
            col = cols[0]
            if action == 'bar' or (df[col].dtype == object and action == 'histogram'):
                counts = df[col].value_counts().reset_index()
                counts.columns = [col, 'count']
                fig = px.bar(counts, x=col, y='count', title=f'Bar: {col}')
                return {'type': 'plotly', 'content': fig}
            else:
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
                return {'type': 'text', 'content': 'Please specify columns to count unique values.'}

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

        return {'type': 'text', 'content': 'Sorry — I did not understand the request.'}

    except Exception as e:
        return {'type': 'text', 'content': f'Error: {e}'}


# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title='PAT', layout='wide')

# Global CSS
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
/* Chat input box */
div[data-testid="stChatInput"] div[role="textbox"],
div[data-testid="stTextArea"] textarea,
textarea[aria-label="Chat input"],
div[role="textbox"] {
    min-height: 80px !important;
    max-height: 240px !important;
    font-size: 16px !important;
    padding: 12px !important;
    border-radius: 10px !important;
    line-height: 1.3 !important;
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
        st.image("ChatGPT Image Aug 18, 2025, 07_38_23 AM.png", width=600)
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
                    else:
                        st.write(msg.get('content'))

    user_input = st.chat_input("Ask me to analyze your data...")
    if user_input:
        st.session_state["chat_started"] = True
        st.session_state['chat_history'].append({'role': 'user', 'content': user_input})
        action = detect_action(user_input)
        result = run_action(action, user_input, st.session_state['df'])
        st.session_state['chat_history'].append({
            'role': 'assistant',
            'type': result['type'],
            'content': result['content']
        })
        st.rerun()


with right_col:
    st.header('📊 Current dataset')
    if st.session_state['df'] is None:
        st.write('No dataset loaded')
    else:
        st.write(f"{st.session_state['df'].shape[0]} rows × {st.session_state['df'].shape[1]} cols")
        if st.button('Show dataframe'):
            st.dataframe(st.session_state['df'])
