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
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# ---------------------- Session State ----------------------
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'chat_started' not in st.session_state:
    st.session_state['chat_started'] = False

# ---------------------- Churn Detection Class ----------------------
class ChurnDetector:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        
    def analyze_data(self, df):
        """Analyze the dataset for churn patterns"""
        print("\n" + "="*50)
        print("DATA ANALYSIS RESULTS")
        print("="*50)
        
        # Basic statistics
        print(f"Dataset shape: {df.shape}")
        if 'churn' in df.columns:
            print(f"Churn rate: {df['churn'].mean():.2%}")
            print(f"Total churned customers: {df['churn'].sum()}")
            print(f"Total retained customers: {(df['churn'] == 0).sum()}")
        
        # Missing values analysis
        print("\nMissing Values Analysis:")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            print(missing_data[missing_data > 0])
        else:
            print("No missing values found")
        
        # Feature distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        print(f"\nNumeric features: {len(numeric_cols)}")
        print(f"Categorical features: {len(categorical_cols)}")
        
        return {
            'shape': df.shape,
            'churn_rate': df['churn'].mean() if 'churn' in df.columns else None,
            'missing_values': missing_data,
            'numeric_features': numeric_cols.tolist(),
            'categorical_features': categorical_cols.tolist()
        }
    
    def predict_churn(self, customer_data, use_rules=True):
        """Predict churn using rule-based approach or loaded models"""
        if use_rules:
            return self.predict_churn_rules(customer_data)
        else:
            if not self.is_trained:
                raise ValueError("Models not loaded yet. Please load models first or use rule-based prediction.")
            
            # This would be used with actual pretrained models
            print("⚠️  Using placeholder prediction logic")
            # Model prediction logic would go here
            
        return self.predict_churn_rules(customer_data)
    
    def predict_churn_rules(self, customer_data):
        """Rule-based churn prediction"""
        print("Using rule-based churn prediction...")
        
        # Initialize churn probability
        churn_prob = np.full(len(customer_data), 0.15)  # base probability
        
        # Apply business rules to adjust probability
        if 'contract_type' in customer_data.columns:
            churn_prob += (customer_data['contract_type'] == 'Month-to-month') * 0.35
            
        if 'payment_method' in customer_data.columns:
            churn_prob += (customer_data['payment_method'] == 'Electronic check') * 0.25
            
        if 'tenure' in customer_data.columns:
            churn_prob += (customer_data['tenure'] < 12) * 0.40
            churn_prob += ((customer_data['tenure'] >= 12) & (customer_data['tenure'] < 24)) * 0.20
            
        if 'monthly_charges' in customer_data.columns:
            churn_prob += (customer_data['monthly_charges'] > 80) * 0.20
            
        if 'tech_support' in customer_data.columns:
            churn_prob += (customer_data['tech_support'] == 'No') * 0.15
            
        if 'online_security' in customer_data.columns:
            churn_prob += (customer_data['online_security'] == 'No') * 0.10
            
        if 'senior_citizen' in customer_data.columns:
            churn_prob += (customer_data['senior_citizen'] == 1) * 0.15
            
        if 'internet_service' in customer_data.columns:
            churn_prob += (customer_data['internet_service'] == 'Fiber optic') * 0.10
        
        # Cap probability at 1.0
        churn_prob = np.minimum(churn_prob, 1.0)
        
        results = pd.DataFrame({
            'customer_id': customer_data['customer_id'] if 'customer_id' in customer_data else range(len(customer_data)),
            'churn_probability': churn_prob,
            'risk_level': pd.cut(churn_prob, 
                               bins=[0, 0.3, 0.7, 1.0], 
                               labels=['Low', 'Medium', 'High'])
        })
        
        return results
    
    def get_churn_insights(self, df):
        """Generate business insights from churn analysis"""
        if 'churn' not in df.columns:
            print("No churn column found for insights generation")
            return {}
        
        insights = {}
        
        # Contract type analysis
        if 'contract_type' in df.columns:
            contract_churn = df.groupby('contract_type')['churn'].agg(['count', 'sum', 'mean'])
            contract_churn.columns = ['total_customers', 'churned_customers', 'churn_rate']
            insights['contract_analysis'] = contract_churn
        
        # Tenure analysis
        if 'tenure' in df.columns:
            df['tenure_group'] = pd.cut(df['tenure'], 
                                      bins=[0, 12, 24, 36, 72], 
                                      labels=['0-12 months', '12-24 months', '24-36 months', '36+ months'])
            tenure_churn = df.groupby('tenure_group')['churn'].agg(['count', 'sum', 'mean'])
            tenure_churn.columns = ['total_customers', 'churned_customers', 'churn_rate']
            insights['tenure_analysis'] = tenure_churn
        
        # Payment method analysis
        if 'payment_method' in df.columns:
            payment_churn = df.groupby('payment_method')['churn'].agg(['count', 'sum', 'mean'])
            payment_churn.columns = ['total_customers', 'churned_customers', 'churn_rate']
            insights['payment_analysis'] = payment_churn
        
        return insights

# ---------------------- Prompt Parsing ----------------------
ACTION_MAP = {
    'mean': ['mean', 'average', 'avg', 'What is the average of','Statistical Analysis'],
    'median': ['median', 'midpoint', 'middle value','Statistical Analysis'],
    'mode': ['mode', 'most frequent', 'common value'],
    'describe': ['describe', 'summary', 'summary statistics', 'dataset summary', 'tell me about the data','Statistical Analysis'],
    'head': ['head', 'show head', 'show first', 'first rows', 'top rows', 'first few rows', 'preview the data'],
    'tail': ['tail', 'last rows', 'bottom rows', 'last few rows'],
    'dropna': ['dropna', 'drop na', 'drop missing', 'remove missing', 'remove rows with missing values'],
    'fillna': ['fillna', 'fill missing', 'impute', 'handle missing', 'handle missing values'],
    'histogram': ['histogram', 'hist', 'distribution', 'numerical distribution', 'distribution of numerical variables'],
    'barchart': ['bar chart', 'bar', 'frequency counts', 'frequency counts for categorical variables'],
    'heatmap': ['heatmap', 'correlation heatmap', 'correlation between numerical variables'],
    'pie':['pie'],
    'scatter': ['scatter', 'scatter plot', 'relationship between numerical variables'],
    'count': ['count', 'value counts'],
    'corr': ['correlation', 'corr', 'correlations', 'correlation matrix'],
    'rows': ['rows','row', 'number of rows', 'row count', 'how many rows','How many rows are in the dataset','give me the structure of the dataset'],
    'columns': ['columns','column', 'number of columns', 'col count', 'how many columns','give me the structure of the dataset'],
    'dtypes': ['datatypes','datatype', 'dtypes', 'types', 'column types', 'what are the data types','structure'],
    'data_quality': ['data quality', 'check quality', 'missing values', 'duplicates', 'outliers', 'clean data', 'check for missing values'],
    'feature_types': ['categorical', 'numerical', 'feature types', 'what are the feature types'],
    'target_relationships': ['target', 'relationship with target', 'analyze target'],
    'distribution': ['distribution', 'how is the data distributed'],
    'line': ['line', 'line chart', 'line plot', 'time series plot', 'plot over time'],
    'insights': ['insights', 'key insights', 'summarize the data', 'what are the key takeaways', 'analyze the dataset and give me some insights', 'tell me about the dataset and its key features', 'I need a summary of the data quality and business insights'],
    'hello':['hello','how are you','Heyy!'],
    'prediction':['future','prediction'],
    'churn': ['churn', 'customer churn', 'churn analysis', 'churn prediction', 'predict churn', 'churn detection']
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

# ---------------------- Single action handler (always returns list of dicts) ----------------------
def run_action(action: str, text: str, df: pd.DataFrame, cols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Execute a single action and return a list of result blocks.
    Each block is a dict: {'type': 'text'|'table'|'plotly'|'matplotlib'|'data_quality'|'download', 'content': ...}
    """
    results: List[Dict[str, Any]] = []

    if df is None:
        return [{"type": "text", "content": "No dataset loaded. Please upload a CSV or Excel file first."}]

    if cols is None:
        cols = extract_column_names(text, df)

    try: 
        if action == "churn":
            # Initialize churn detector
            detector = ChurnDetector()
            
            # Check if we have required columns for churn analysis
            required_cols = ['customer_id', 'tenure', 'monthly_charges', 'contract_type']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                # If missing required columns, provide guidance
                results.append({
                    "type": "text", 
                    "content": f"⚠️ Missing required columns for churn analysis: {missing_cols}\n\n"
                                 f"Available columns: {list(df.columns)}\n\n"
                                 f"For optimal churn analysis, your dataset should include:\n"
                                 f"- customer_id (unique identifier)\n"
                                 f"- tenure (months as customer)\n" 
                                 f"- monthly_charges (monthly fee)\n"
                                 f"- contract_type (Month-to-month, One year, Two year)\n"
                                 f"- payment_method (payment type)\n"
                                 f"- churn (target variable: 1 for churned, 0 for retained)\n\n"
                                 f"Will proceed with available data..."
                })
            
            # Perform data analysis
            analysis_results = detector.analyze_data(df)
            
            # Add analysis summary
            analysis_text = f"📊 **Dataset Analysis Summary**\n\n"
            analysis_text += f"• Dataset shape: {analysis_results['shape']}\n"
            if analysis_results['churn_rate']:
                analysis_text += f"• Churn rate: {analysis_results['churn_rate']:.2%}\n"
            analysis_text += f"• Numeric features: {len(analysis_results['numeric_features'])}\n"
            analysis_text += f"• Categorical features: {len(analysis_results['categorical_features'])}\n"
            
            if analysis_results['missing_values'].sum() > 0:
                analysis_text += f"• Missing values detected in {(analysis_results['missing_values'] > 0).sum()} columns\n"
            else:
                analysis_text += f"• No missing values found\n"
            
            results.append({"type": "text", "content": analysis_text})
            
            # Generate business insights if churn column exists
            if 'churn' in df.columns:
                insights = detector.get_churn_insights(df)
                
                for insight_name, insight_data in insights.items():
                    if not insight_data.empty:
                        insight_title = insight_name.replace('_', ' ').title()
                        results.append({
                            "type": "table", 
                            "content": {
                                "title": f"📈 {insight_title}",
                                "data": insight_data.round(3)
                            }
                        })
            
            # Predict churn for all customers using rule-based approach
            predictions = detector.predict_churn(df, use_rules=True)
            
            # Add prediction summary
            risk_summary = predictions['risk_level'].value_counts()
            summary_text = f"🎯 **Churn Risk Assessment**\n\n"
            
            for level in ['High', 'Medium', 'Low']:
                if level in risk_summary.index:
                    count = risk_summary[level]
                    pct = (count / len(predictions)) * 100
                    if level == 'High':
                        summary_text += f"🔴 **{level} Risk**: {count} customers ({pct:.1f}%) - Immediate action needed\n"
                    elif level == 'Medium':
                        summary_text += f"🟡 **{level} Risk**: {count} customers ({pct:.1f}%) - Monitor closely\n"
                    else:
                        summary_text += f"🟢 **{level} Risk**: {count} customers ({pct:.1f}%) - Stable customers\n"
            
            results.append({"type": "text", "content": summary_text})
            
            # Show top high-risk customers
            high_risk = predictions[predictions['risk_level'] == 'High'].head(10)
            if len(high_risk) > 0:
                # Merge with original data to show customer details
                detailed_high_risk = high_risk.merge(df, on='customer_id', how='left')
                
                # Select relevant columns for display
                display_cols = ['customer_id', 'churn_probability', 'risk_level']
                for col in ['tenure', 'monthly_charges', 'contract_type', 'payment_method']:
                    if col in detailed_high_risk.columns:
                        display_cols.append(col)
                
                results.append({
                    "type": "table",
                    "content": {
                        "title": "🚨 Top 10 High-Risk Customers",
                        "data": detailed_high_risk[display_cols].round(3)
                    }
                })
            
            # Business recommendations
            recommendations = f"💡 **Recommended Actions**\n\n"
            
            if 'High' in risk_summary.index and risk_summary['High'] > 0:
                recommendations += f"**For {risk_summary['High']} High-Risk Customers:**\n"
                recommendations += f"• Immediate personal outreach within 48 hours\n"
                recommendations += f"• Offer retention incentives (discounts, upgrades)\n"
                recommendations += f"• Schedule customer satisfaction calls\n"
                recommendations += f"• Consider contract upgrade offers\n\n"
            
            if 'Medium' in risk_summary.index and risk_summary['Medium'] > 0:
                recommendations += f"**For {risk_summary['Medium']} Medium-Risk Customers:**\n"
                recommendations += f"• Proactive engagement via email/SMS\n"
                recommendations += f"• Enroll in loyalty programs\n" 
                recommendations += f"• Conduct service quality surveys\n"
                recommendations += f"• Monitor usage patterns\n\n"
            
            recommendations += f"**General Strategy:**\n"
            recommendations += f"• Focus on month-to-month contract customers\n"
            recommendations += f"• Improve payment experience (reduce electronic check friction)\n"
            recommendations += f"• Enhance customer support for new customers (tenure < 12 months)\n"
            recommendations += f"• Review pricing strategy for high monthly charges\n"
            
            results.append({"type": "text", "content": recommendations})
            
            # Return full predictions as downloadable data
            results.append({
                "type": "download",
                "content": {
                    "filename": "churn_predictions.csv",
                    "data": predictions.merge(df, on='customer_id', how='left')
                }
            })
            return results

        if action == "prediction":
            # Ensure Date column is parsed as datetime
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            # Pick default time and value columns if auto-detect fails
            time_cols = [col for col in df.columns if df[col].dtype == "datetime64[ns]"]
            value_cols = [col for col in df.columns if df[col].dtype in ["float64", "int64"]]

            # Override defaults if still missing
            if not time_cols and "Date" in df.columns:
                time_cols = ["Date"]

            if not value_cols:
                if "close" in df.columns:
                    value_cols = ["close"]
                elif "ltp" in df.columns:
                    value_cols = ["ltp"]
                elif "OPEN" in df.columns:
                    value_cols = ["OPEN"]

            # Use first available
            if time_cols:
                 TIME_COL = time_cols[0]
            if value_cols:
                 VALUE_COL = value_cols[0]

            # Check if we have time series data
            time_cols = []
            value_cols = []
            
            # Look for time columns
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
                    time_cols.append(col)
                elif df[col].dtype in ['float64', 'int64'] and col.lower() not in ['id', 'customer_id']:
                    value_cols.append(col)
            
            if not time_cols:
                # Try to convert columns that might be dates
                for col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        time_cols.append(col)
                        break
                    except:
                        continue
            
            if not time_cols or not value_cols:
                results.append({
                    "type": "text", 
                    "content": "⚠️ Time series prediction requires:\n"
                                 "- A date/time column\n"
                                 "- A numeric value column (e.g., Close, Price, Sales)\n\n"
                                 f"Available columns: {list(df.columns)}\n"
                                 "Please ensure your data has proper date and numeric columns."
                })
                return results
            
            # Use first available time and value columns (Re-defining for safety and clarity)
            TIME_COL = time_cols[0]
            VALUE_COL = value_cols[0]
            
            # Parameters
            MA_SHORT = 10
            MA_LONG = 20
            RSI_WINDOW = 14
            BB_WINDOW = 20
            BB_STD = 2
            ROC_PERIOD = 10
            FLAT_THRESHOLD = 0.33

            # Prepare data
            df_pred = df.copy()
            df_pred[TIME_COL] = pd.to_datetime(df_pred[TIME_COL])
            df_pred = df_pred.sort_values(TIME_COL).reset_index(drop=True)
            price_col = VALUE_COL

            # 1. MOVING AVERAGES (TREND)
            df_pred['MA_short'] = df_pred[price_col].rolling(MA_SHORT, min_periods=1).mean()
            df_pred['MA_long'] = df_pred[price_col].rolling(MA_LONG, min_periods=1).mean()
            df_pred['MA_signal'] = np.where(df_pred['MA_short'] > df_pred['MA_long'], 1,
                                         np.where(df_pred['MA_short'] < df_pred['MA_long'], -1, 0))

            # 2. RSI (MOMENTUM)
            df_pred['change'] = df_pred[price_col].diff()
            df_pred['gain'] = df_pred['change'].where(df_pred['change'] > 0, 0)
            df_pred['loss'] = -df_pred['change'].where(df_pred['change'] < 0, 0)
            df_pred['avg_gain'] = df_pred['gain'].rolling(RSI_WINDOW, min_periods=1).mean()
            df_pred['avg_loss'] = df_pred['loss'].rolling(RSI_WINDOW, min_periods=1).mean()
            df_pred['RS'] = df_pred['avg_gain'] / df_pred['avg_loss'].replace(0, np.nan)
            df_pred['RSI'] = 100 - (100 / (1 + df_pred['RS']))
            df_pred['RSI'] = df_pred['RSI'].fillna(50)  # Fill NaN with neutral RSI
            df_pred['RSI_signal'] = np.where(df_pred['RSI'] < 30, 1,
                                            np.where(df_pred['RSI'] > 70, -1, 0))

            # 3. BOLLINGER BANDS (VOLATILITY)
            df_pred['BB_middle'] = df_pred[price_col].rolling(BB_WINDOW, min_periods=1).mean()
            df_pred['BB_std'] = df_pred[price_col].rolling(BB_WINDOW, min_periods=1).std()
            df_pred['BB_upper'] = df_pred['BB_middle'] + (BB_STD * df_pred['BB_std'])
            df_pred['BB_lower'] = df_pred['BB_middle'] - (BB_STD * df_pred['BB_std'])

            df_pred['BB_signal'] = 0
            df_pred.loc[df_pred[price_col] >= df_pred['BB_upper'] * 0.98, 'BB_signal'] = -1
            df_pred.loc[df_pred[price_col] <= df_pred['BB_lower'] * 1.02, 'BB_signal'] = 1

            # 4. SUPPORT & RESISTANCE — FIBONACCI LEVELS
            absolute_high = df_pred[price_col].max()
            absolute_low = df_pred[price_col].min()
            price_range = absolute_high - absolute_low

            FIB_RATIOS = {'0.000': 0.000, '0.382': 0.382, '0.618': 0.618, '1.000': 1.000}
            fib_levels = {label: absolute_low + ratio * price_range for label, ratio in FIB_RATIOS.items()}

            support_prices = [fib_levels['0.000'], fib_levels['0.382'], fib_levels['0.618']]
            resistance_prices = [fib_levels['0.382'], fib_levels['0.618'], fib_levels['1.000']]

            def get_nearest_support(price, supports):
                candidates = [s for s in supports if s <= price]
                return max(candidates) if candidates else min(supports)

            def get_nearest_resistance(price, resistances):
                candidates = [r for r in resistances if r >= price]
                return min(candidates) if candidates else max(resistances)

            df_pred['nearest_support'] = df_pred[price_col].apply(lambda p: get_nearest_support(p, support_prices))
            df_pred['nearest_resistance'] = df_pred[price_col].apply(lambda p: get_nearest_resistance(p, resistance_prices))
            df_pred['zone_range'] = df_pred['nearest_resistance'] - df_pred['nearest_support']
            df_pred['distance_to_support'] = df_pred[price_col] - df_pred['nearest_support']
            df_pred['distance_to_resistance'] = df_pred['nearest_resistance'] - df_pred[price_col]
            df_pred['pct_to_support'] = df_pred['distance_to_support'] / df_pred['zone_range']
            df_pred['pct_to_resistance'] = df_pred['distance_to_resistance'] / df_pred['zone_range']
            df_pred['SR_signal'] = 0
            df_pred.loc[df_pred['pct_to_support'] <= 0.30, 'SR_signal'] = 1
            df_pred.loc[df_pred['pct_to_resistance'] <= 0.30, 'SR_signal'] = -1

            # 5. MOMENTUM (ROC)
            df_pred['ROC'] = ((df_pred[price_col] / df_pred[price_col].shift(ROC_PERIOD)) - 1) * 100
            df_pred['ROC'] = df_pred['ROC'].fillna(0)  # Fill NaN with 0
            df_pred['MOM_signal'] = np.where(df_pred['ROC'] > 2, 1,
                                            np.where(df_pred['ROC'] < -2, -1, 0))

            # 6. FINAL PREDICTION
            weights = {'MA': 1.5, 'RSI': 1.0, 'BB': 1.2, 'MOM': 1.0, 'SR': 2.0}
            MAX_SIGNAL = sum(weights.values())

            df_pred['total_signal'] = (
                df_pred['MA_signal'] * weights['MA'] +
                df_pred['RSI_signal'] * weights['RSI'] +
                df_pred['BB_signal'] * weights['BB'] +
                df_pred['MOM_signal'] * weights['MOM'] +
                df_pred['SR_signal'] * weights['SR']
            )
            df_pred['normalized_signal'] = df_pred['total_signal'] / MAX_SIGNAL
            df_pred['prediction'] = 'SIDEWAYS'
            df_pred.loc[df_pred['normalized_signal'] > FLAT_THRESHOLD, 'prediction'] = 'UP'
            df_pred.loc[df_pred['normalized_signal'] < -FLAT_THRESHOLD, 'prediction'] = 'DOWN'
            df_pred['conviction_score'] = (df_pred['normalized_signal'].abs() * 100).clip(upper=85)

            # PLOTTING
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            
            # Price plot with indicators
            ax1.plot(df_pred[TIME_COL], df_pred[price_col], label=f'{VALUE_COL} Price', color='black', linewidth=1.8, zorder=3)
            ax1.plot(df_pred[TIME_COL], df_pred['BB_upper'], label='BB Upper', color='blue', linestyle='--', alpha=0.7)
            ax1.plot(df_pred[TIME_COL], df_pred['BB_middle'], label='BB Middle (MA20)', color='grey', linestyle='-', alpha=0.7)
            ax1.plot(df_pred[TIME_COL], df_pred['BB_lower'], label='BB Lower', color='blue', linestyle='--', alpha=0.7)
            ax1.fill_between(df_pred[TIME_COL], df_pred['BB_lower'], df_pred['BB_upper'], color='blue', alpha=0.08)
            ax1.plot(df_pred[TIME_COL], df_pred['MA_short'], label=f'MA {MA_SHORT}', color='orange', linewidth=1.5)
            ax1.plot(df_pred[TIME_COL], df_pred['MA_long'], label=f'MA {MA_LONG}', color='red', linewidth=1.5)
            
            # Support and resistance levels
            support_levels = [('Fib 0.000', fib_levels['0.000'], 'darkgreen'),
                              ('Fib 0.382', fib_levels['0.382'], 'green'),
                              ('Fib 0.618', fib_levels['0.618'], 'lightgreen')]
            resistance_levels = [('Fib 0.618', fib_levels['0.618'], 'orangered'),
                                 ('Fib 1.000', fib_levels['1.000'], 'darkred')]
            
            for label, level, color in support_levels:
                ax1.axhline(y=level, color=color, linestyle='-', linewidth=2, alpha=0.85, label=label)
            for label, level, color in resistance_levels:
                ax1.axhline(y=level, color=color, linestyle='-', linewidth=2, alpha=0.85, label=label)

            # Prediction markers
            up_mask = df_pred['prediction'] == 'UP'
            down_mask = df_pred['prediction'] == 'DOWN'
            ax1.scatter(df_pred[TIME_COL][up_mask], df_pred[price_col][up_mask],
                        marker='^', color='green', s=100, edgecolors='black', linewidth=0.5, label='Predicted UP', zorder=5)
            ax1.scatter(df_pred[TIME_COL][down_mask], df_pred[price_col][down_mask],
                        marker='v', color='red', s=100, edgecolors='black', linewidth=0.5, label='Predicted DOWN', zorder=5)

            ax1.set_title(f'Technical Analysis - {VALUE_COL} Prediction', fontsize=14, fontweight='bold')
            ax1.set_ylabel(f'{VALUE_COL}', fontsize=12)
            ax1.legend(loc='upper center', fontsize=9, ncol=2)
            ax1.grid(True, alpha=0.3)

            # RSI subplot
            ax2.plot(df_pred[TIME_COL], df_pred['RSI'], label='RSI', color='purple', linewidth=1.5)
            ax2.axhline(70, color='red', linestyle='--', alpha=0.6, label='Overbought (70)')
            ax2.axhline(30, color='green', linestyle='--', alpha=0.6, label='Oversold (30)')
            ax2.axhline(50, color='grey', linestyle='-', alpha=0.5)
            ax2.set_ylabel('RSI', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.legend(loc='upper left', fontsize=9)
            ax2.grid(True, alpha=0.3)
            
            # Format dates
            fig.autofmt_xdate()
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.tight_layout()

            # Get latest prediction
            last_idx = len(df_pred) - 1
            last_pred = df_pred.iloc[last_idx]['prediction']
            last_conviction = df_pred.iloc[last_idx]['conviction_score']
            last_norm_sig = df_pred.iloc[last_idx]['normalized_signal']
            last_support = df_pred.iloc[last_idx]['nearest_support']
            last_resistance = df_pred.iloc[last_idx]['nearest_resistance']
            last_rsi = df_pred.iloc[last_idx]['RSI']

            breakdown_text = f"📈 **TECHNICAL ANALYSIS PREDICTION**\n\n"
            breakdown_text += f"**Final Prediction: {last_pred}**\n"
            breakdown_text += f"Conviction Score: {last_conviction:.1f}%\n"
            breakdown_text += f"Normalized Signal: {last_norm_sig:+.3f}\n\n"
            
            breakdown_text += f"**Key Levels:**\n"
            breakdown_text += f"• Nearest Support: {last_support:.2f}\n"
            breakdown_text += f"• Nearest Resistance: {last_resistance:.2f}\n"
            breakdown_text += f"• Current RSI: {last_rsi:.1f}\n\n"
            
            breakdown_text += f"**Signal Breakdown:**\n"
            breakdown_text += f"• MA Signal (×1.5): {df_pred.iloc[last_idx]['MA_signal'] * weights['MA']:+.1f}\n"
            breakdown_text += f"• RSI Signal (×1.0): {df_pred.iloc[last_idx]['RSI_signal'] * weights['RSI']:+.1f}\n"
            breakdown_text += f"• BB Signal (×1.2): {df_pred.iloc[last_idx]['BB_signal'] * weights['BB']:+.1f}\n"
            breakdown_text += f"• Momentum Signal (×1.0): {df_pred.iloc[last_idx]['MOM_signal'] * weights['MOM']:+.1f}\n"
            breakdown_text += f"• S/R Signal (×2.0): {df_pred.iloc[last_idx]['SR_signal'] * weights['SR']:+.1f}\n"
            breakdown_text += f"• **Total Signal: {df_pred.iloc[last_idx]['total_signal']:.2f} / {MAX_SIGNAL}**"

            results.append({"type": "matplotlib", "content": fig})
            results.append({"type": "text", "content": breakdown_text})
            
            # Add prediction data as download
            prediction_summary = df_pred[[TIME_COL, VALUE_COL, 'prediction', 'conviction_score', 'RSI', 'MA_short', 'MA_long']].tail(20)
            results.append({
                "type": "download",
                "content": {
                    "filename": "technical_analysis_predictions.csv",
                    "data": prediction_summary
                }
            })
            return results

        if action == "hello":
            return [{"type": "text", "content": "Heyy!! How can I help you today? Upload the dataset and let's start the action!"}]
        # ---------------- Dataset Info ----------------
        if action in ("structure", "rows"):
            results.append({"type": "text", "content": f"The dataset has **{df.shape[0]} rows**."})
            return results

        if action in ("structure", "columns"):
            results.append({"type": "text", "content": f"The dataset has **{df.shape[1]} columns**."})
            return results

        if action == "dtypes":
            results.append({"type": "table", "content": df.dtypes.to_frame("dtype")})
            return results

        # ---------------- Describe ----------------
        if action == "describe":
            desc = df.describe(include='all').T  # transpose so columns are rows
            results.append({"type": "table", "content": desc})
            return results

        # ---------------- Mean / Median / Mode ----------------
        if action == "mean":
            numeric = df.select_dtypes(include=[np.number])
            results.append({"type": "table", "content": numeric.mean().to_frame("mean")})
            return results

        if action == "median":
            numeric = df.select_dtypes(include=[np.number])
            results.append({"type": "table", "content": numeric.median().to_frame("median")})
            return results

        if action == "mode":
            if not cols:
                results.append({"type": "text", "content": "Please specify columns for mode calculation."})
                return results
            mode_res = {}
            for c in cols:
                if c in df.columns:
                    mode_res[c] = df[c].mode().tolist()
            results.append({"type": "text", "content": f"Mode:\n{mode_res}"})
            return results

        # ---------------- Insights ----------------
        if action == "insights":
            business_text = ["### 💡 Business / Practical Insights\n"]
            
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if "id" not in c.lower()]
            time_cols = [c for c in df.columns if "date" in c.lower() or "year" in c.lower() or "time" in c.lower()]
            
            # Trends
            trends = []
            if time_cols and numeric_cols:
                time_col = time_cols[0]
                df_sorted = df.sort_values(by=time_col)
                for col in numeric_cols:
                    trend = df_sorted[col].diff().mean()
                    if trend > 0:
                        trends.append(f"📈 **{col}** shows an increasing trend over {time_col}.")
                    elif trend < 0:
                        trends.append(f"📉 **{col}** shows a decreasing trend over {time_col}.")
            
            if trends:
                business_text.extend(trends)
            
            # Correlations
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                strong_pos = []
                strong_neg = []
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        val = corr.iloc[i, j]
                        if val > 0.7:
                            strong_pos.append(f"{corr.columns[i]} & {corr.columns[j]}")
                        elif val < -0.7:
                            strong_neg.append(f"{corr.columns[i]} & {corr.columns[j]}")
                if strong_pos:
                    business_text.append(f"🔗 Strong positive correlations: {', '.join(strong_pos)}")
                if strong_neg:
                    business_text.append(f"🔗 Strong negative correlations: {', '.join(strong_neg)}")
            
            # Anomalies
            anomalies = {}
            for col in numeric_cols:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > 3]
                if not outliers.empty:
                    anomalies[col] = len(outliers)
            if anomalies:
                anomaly_summary = ", ".join([f"{col} ({count} outliers)" for col, count in anomalies.items()])
                business_text.append(f"🚨 Anomalies detected in: {anomaly_summary}")

            # Segmentation opportunities
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if any(df[c].nunique() > 1 and df[c].nunique() <= 10 for c in cat_cols):
                business_text.append("🧩 Segmentation opportunities detected in categorical features.")

            # Fallback
            if len(business_text) == 1:
                business_text.append("No strong patterns detected. Review correlations, trends, and anomalies manually.")

            results.append({"type": "text", "content": "\n".join(business_text)})
            return results

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

        # ---------------- Head/Tail ----------------
        if action == "head":
            n = 5
            m = re.search(r"head\s*(\d+)", text.lower())
            if m:
                n = int(m.group(1))
            results.append({"type": "table", "content": df.head(n)})
            return results
        if action == "tail":
            n = 5
            m = re.search(r"tail\s*(\d+)", text.lower())
            if m:
                n = int(m.group(1))
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
        if action in ("histogram", "bar", "line", "scatter", "heatmap","pie"):
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
                figs = []
                time_cols = [
                    c for c in df.columns
                    if "date" in c.lower() or "year" in c.lower() or "time" in c.lower()
                ]
                if not time_cols:
                    results.append({"type": "text", "content": "No timeline (date/year) column found for line plots."})
                else:
                    time_col = time_cols[0]  # take the first match
                    numeric_cols = [
                        c for c in df.select_dtypes(include=[np.number]).columns
                        if "id" not in c.lower()
                    ]
                    if not numeric_cols:
                        results.append({"type": "text", "content": "No numeric columns available for line plots."})
                    else:
                        for col in numeric_cols:
                            fig = px.line(df, x=time_col, y=col, title=f"{col} over {time_col}")
                            fig.update_traces(
                                line=dict(width=2),
                                marker=dict(line=dict(width=1, color="black"))
                            )
                            results.append({"type": "plotly", "content": fig})
                            figs.append(fig)
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

            elif action == "histogram":
                num_cols = [
                    c for c in df.select_dtypes(include=[np.number]).columns
                    if "id" not in c.lower()
                ]
                figs = []
                for col in num_cols:
                    fig = px.histogram(
                        df, x=col, title=f"Histogram: {col}"
                    )
                    fig.update_traces(marker=dict(line=dict(width=1, color="black")))
                    results.append({"type": "plotly", "content": fig})
                    figs.append(fig)
                return results

            elif action == "bar":
                cat_cols = df.select_dtypes(exclude=[np.number]).columns
                for col in cat_cols:
                    counts = df[col].value_counts().reset_index()
                    counts.columns = [col, "count"]
                    fig = px.bar(counts, x=col, y="count", title=f"Bar Chart: {col}")
                    fig.update_traces(marker=dict(line=dict(width=1, color="black")))
                    results.append({"type": "plotly", "content": fig})
                    return results
            elif action == "pie":
                cat_cols = df.select_dtypes(exclude=[np.number]).columns
                if not cat_cols.any():
                    results.append({"type": "text", "content": "No categorical columns available for pie chart."})
                    return results
                else:
                    for col in cat_cols:
                        counts = df[col].value_counts().reset_index()
                        counts.columns = [col, "count"]
                        fig = px.pie(counts, names=col, values="count", title=f"Pie Chart: {col}")
                        fig.update_traces(textinfo="percent+label", pull=[0.05]*len(counts))
                        results.append({"type": "plotly", "content": fig})
                    return results

            # ---------------- Dynamic Business/Practical Insights ----------------
            business_text = ["### 💡 Business / Practical Insights\n\n"]

            # Insight 1: Trends
            if trends:
                positive_trends = [t for t in trends if "increasing" in t]
                negative_trends = [t for t in trends if "decreasing" in t]
                
                if positive_trends:
                    business_text.append("✅ Leverage positive trends. Variables like " + ", ".join([t.split('**')[1] for t in positive_trends]) + " are growing, which could signal business success or a positive market shift.")
                if negative_trends:
                    business_text.append("🛑 Address negative trends. The decline in " + ", ".join([t.split('**')[1] for t in negative_trends]) + " may require strategic intervention or further investigation.")
            
            # Insight 2: Correlations
            numeric_cols_for_corr = [c for c in df.select_dtypes(include=[np.number]).columns if "id" not in c.lower()]
            if len(numeric_cols_for_corr) > 1:
                corr = df[numeric_cols_for_corr].corr()
                strong_pos_corrs = []
                strong_neg_corrs = []
                
                for i in range(len(corr.columns)):
                    for j in range(i + 1, len(corr.columns)):
                        val = float(corr.iloc[i, j])
                        if val > 0.7:
                            strong_pos_corrs.append((corr.columns[i], corr.columns[j]))
                        elif val < -0.7:
                            strong_neg_corrs.append((corr.columns[i], corr.columns[j]))
                
                if strong_pos_corrs:
                    business_text.append(f"🔗 **Strong positive correlations** detected: " + ", ".join([f"{c1} & {c2}" for c1, c2 in strong_pos_corrs]) + ". Consider how these variables influence each other to optimize your strategy.")
                if strong_neg_corrs:
                    business_text.append(f"🔗 **Strong negative correlations** detected: " + ", ".join([f"{c1} & {c2}" for c1, c2 in strong_neg_corrs]) + ". The inverse relationship may present opportunities for targeted adjustments.")

            # Insight 3: Anomalies
            if anomalies:
                anomaly_summary = ", ".join([f"{col} ({count} outliers)" for col, count in anomalies.items()])
                business_text.append(f"🚨 **Anomalies** found in {anomaly_summary}. These data points could be errors or valuable signals of a unique event that warrants further investigation.")

            # Insight 4: Segmentation
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if any(df[c].nunique() > 1 and df[c].nunique() <= 10 for c in cat_cols):
                business_text.append("🧩 **Segmentation** opportunities are present. Consider how customer or product segments based on categorical features might reveal different behaviors or trends.")
                
            # Fallback to general insights if no specifics found
            if len(business_text) <= 1: # Only contains the heading
                business_text.append("No specific patterns or strong trends were found. Consider these general tips:\n"
                                     "- Look for correlations between your key metrics.\n"
                                     "- Investigate any extreme values for potential anomalies.")

            results.append({"type": "text", "content": "\n\n".join(business_text)})
            
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

    except Exception as e:
        results.append({"type": "text", "content": "Sorry — I did not understand the request."})
        return results
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
    st.header('PAT.ai')
    st.caption('Perform | Analyze | Transform')
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

    st.caption('Developed by: Colin, Nisarg, Dona')

# ---------------------- Main Chat ----------------------
chat_col, right_col = st.columns([3, 1])

with chat_col:
    if not st.session_state["chat_started"]:
        st.markdown("## 👋 Hi, I'm **PAT**, your Data Analyst.")
        st.markdown("I can refine your data, analyze it, and create visuals.")
        st.image("Made with insMind-IMG_20250915_063635 (1).png", width=600)
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
                        st.plotly_chart(msg['content'], use_container_width=True, key=f"plotly_{id(msg)}")
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

    # --- Predefined Queries---
    predefined_queries = {
    1: "Summary of the dataset",
    2: "Want a Data Quality Report?",
    3: "what are the feature types?",
    4: "Numeric Distribution of dataset",
    5: "The Target Relationships of dataset",
    6: "Want a Pie chart?",
    7: "Relationship between numerical variables?",
    8: "Explore Correlation Heatmap?",
    9: "Lets analyze the Key Insights",
    10: "Lets Predict the Dataset",
    11: "Lets detect a churn from dataset",
    12: "Heyy!"
}
    # --- MOVED and CORRECTED INDENTATION for Predefined Queries ---
    if "remaining_queries" not in st.session_state:
        st.session_state["remaining_queries"] = [1, 2, 3]  
    valid_remaining = [q for q in st.session_state["remaining_queries"] if q in predefined_queries]
    st.session_state["remaining_queries"] = valid_remaining

    st.markdown("<div class='suggestions'>", unsafe_allow_html=True)
    cols = st.columns(len(st.session_state["remaining_queries"]))
    for i, q in enumerate(st.session_state["remaining_queries"]):
        label = predefined_queries.get(q)
        if label:
            if cols[i].button(label, key=f"predef-{q}"):
                query_text = label
                actions = detect_actions(query_text)
                results = run_actions(actions, query_text, st.session_state["df"])

                for result in results:
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "type": result["type"], "content": result["content"]}
                    )

                st.session_state["remaining_queries"].remove(q)
                next_q = max(st.session_state["remaining_queries"], default=0) + 1
                if next_q in predefined_queries:
                    st.session_state["remaining_queries"].append(next_q)

                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # --- MOVED and CORRECTED INDENTATION for CSS ---
    st.markdown("""
        <style>
div[data-testid="stHorizontalBlock"] button {
        background-color: #2d2f38;
        color: white;
        border-radius: 20px;
        padding: 6px 16px;
        border: none;
        margin: 0 5px; /* Add spacing between buttons */
    }
div[data-testid="stHorizontalBlock"] button:hover {
        background-color: #444654;
        color: white;
    }
        </style>
    """, unsafe_allow_html=True)

    # --- MOVED and CORRECTED INDENTATION for Handle manual input ---
    if user_input:
        st.session_state["chat_started"] = True
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_input}
        )

        actions = detect_actions(user_input)
        results = []
        for action in actions:
            action_results = run_action(action, user_input, st.session_state["df"])
            if isinstance(action_results, dict):
                results.append(action_results)
            elif isinstance(action_results, list):
                results.extend(action_results)

        for result in results:
            st.session_state["chat_history"].append(
                {"role": "assistant", "type": result["type"], "content": result["content"]}
            )

        st.rerun()

# ---------------------- Right Column ----------------------
with right_col:
    st.header('📊 Current dataset')
    if st.session_state['df'] is None:
        st.write('No dataset loaded')
    else:
        st.write(f"{st.session_state['df'].shape[0]} rows × {st.session_state['df'].shape[1]} cols")
        if st.button('Show dataframe'):
            st.dataframe(st.session_state['df'])