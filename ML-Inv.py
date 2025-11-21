# ML_Inv_windows.py
"""
Windows-ready ML pipeline for invoice data (no Sales Employee logic).

Save as ML_Inv_windows.py and run from PowerShell/Command Prompt:
    python ML_Inv_windows.py --input "C:\path\to\INVOICE DATA- MLR-DT-LR.xlsx" --output_dir "C:\path\to\out"

Features:
- Reads .xlsx/.xls/.csv
- Monthly timeseries + forecasting (Prophet -> SARIMAX -> naive)
- RFM segmentation (KMeans)
- Churn labeling (simple)
- Anomaly detection (IsolationForest)
- Invoice value prediction: Linear Regression, Decision Tree (GridSearch), Random Forest
- Saves CSVs, PNG plots, joblib models, and a JSON report
"""
import os
import sys
import argparse
import warnings
from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# Make script safe when launched from Jupyter (clears injected argv)
if 'ipykernel' in sys.modules:
    sys.argv = [sys.argv[0]]

# Optional forecasting libs
HAS_PROPHET = False
HAS_STATS = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    try:
        from fbprophet import Prophet
        HAS_PROPHET = True
    except Exception:
        HAS_PROPHET = False

try:
    import statsmodels.api as sm
    HAS_STATS = True
except Exception:
    HAS_STATS = False

# ---------- Utilities ----------
def read_data(path, date_col='Date'):
    path = str(path)
    if path.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found. Columns: {df.columns.tolist()}")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).str.strip()
    return df

def basic_clean(df):
    # Document Total detection
    if 'Document Total' not in df.columns:
        cand = [c for c in df.columns if 'document' in c.lower() and 'total' in c.lower()]
        if cand:
            df = df.rename(columns={cand[0]: 'Document Total'})
        else:
            raise ValueError("Document Total column not found.")
    df['Document Total'] = pd.to_numeric(df['Document Total'], errors='coerce').fillna(0.0)
    if 'INV No' not in df.columns:
        df['INV No'] = pd.Series(np.arange(1, len(df) + 1)).astype(str)
    df['INV No'] = df['INV No'].astype(str)
    if 'Customer Name' not in df.columns:
        df['Customer Name'] = 'UNKNOWN'
    df = df.dropna(subset=['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    return df

def save_fig(fig, path):
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)

# ---------- Time-series ----------
def aggregate_time_series(df, freq='M'):
    ts = df.set_index('Date').resample(freq)['Document Total'].sum().reset_index()
    ts = ts.rename(columns={'Date': 'ds', 'Document Total': 'y'})
    return ts

def forecast(ts, periods=12, prefer='prophet'):
    if prefer == 'prophet' and HAS_PROPHET:
        m = Prophet()
        m.fit(ts)
        future = m.make_future_dataframe(periods=periods, freq='M')
        forecast_df = m.predict(future)
        return 'prophet', m, forecast_df
    if HAS_STATS:
        try:
            ts2 = ts.set_index('ds').asfreq('M')
            ts2['y'] = ts2['y'].fillna(0)
            model = sm.tsa.statespace.SARIMAX(ts2['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                                             enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            pred = res.get_forecast(steps=periods)
            pred_index = pd.date_range(start=ts2.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq='M')
            forecast_df = pd.DataFrame({'ds': pred_index, 'yhat': pred.predicted_mean.values})
            return 'sarimax', res, forecast_df
        except Exception:
            pass
    # naive moving average
    ts = ts.copy()
    ts['ma3'] = ts['y'].rolling(3, min_periods=1).mean()
    last = ts['ma3'].iloc[-1]
    future_idx = pd.date_range(start=ts['ds'].iloc[-1] + pd.offsets.MonthBegin(1), periods=periods, freq='M')
    forecast_df = pd.DataFrame({'ds': future_idx, 'yhat': np.repeat(last, periods)})
    return 'naive', None, forecast_df

# ---------- RFM ----------
def rfm_features(df, snapshot_date=None):
    if snapshot_date is None:
        snapshot_date = df['Date'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('Customer Name').agg({
        'Date': lambda x: (snapshot_date - x.max()).days,
        'INV No': 'count',
        'Document Total': 'sum'
    }).rename(columns={'Date': 'Recency', 'INV No': 'Frequency', 'Document Total': 'Monetary'}).reset_index()
    return rfm

def customer_segmentation(rfm, n_clusters=4):
    rfm2 = rfm.copy()
    rfm2['Monetary'] = np.log1p(rfm2['Monetary'])
    X = rfm2[['Recency', 'Frequency', 'Monetary']].fillna(0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    rfm2['Cluster'] = kmeans.fit_predict(X)
    return rfm2, kmeans

# ---------- Churn ----------
def create_churn_label(df, days_window=90):
    last_date = df['Date'].max()
    cutoff = last_date - pd.Timedelta(days=days_window)
    recent = set(df[df['Date'] > cutoff]['Customer Name'].unique())
    customers = df['Customer Name'].unique()
    labels = pd.DataFrame({'Customer Name': customers})
    labels['churn'] = (~labels['Customer Name'].isin(recent)).astype(int)
    return labels

# ---------- Anomaly detection ----------
def anomaly_detection(df):
    df2 = df.copy()
    cust_avg = df.groupby('Customer Name')['Document Total'].mean().rename('cust_mean')
    df2 = df2.join(cust_avg, on='Customer Name')
    df2['ratio'] = df2['Document Total'] / (df2['cust_mean'] + 1e-6)
    iso = IsolationForest(contamination=0.01, random_state=RANDOM_STATE)
    feats = ['Document Total', 'ratio']
    df2['anomaly_score'] = iso.fit_predict(df2[feats].fillna(0))
    df2['is_anomaly'] = (df2['anomaly_score'] == -1).astype(int)
    return df2[['INV No', 'Date', 'Customer Name', 'Document Total', 'is_anomaly']]

# ---------- Regression features & models ----------
def prepare_regression_features(df):
    d = df.copy()
    cust_avg = df.groupby('Customer Name')['Document Total'].agg(['mean', 'count']).rename(columns={'mean': 'cust_avg', 'count': 'cust_count'})
    d = d.join(cust_avg, on='Customer Name')
    d['month'] = d['Date'].dt.month
    d['dayofweek'] = d['Date'].dt.dayofweek
    d['is_month_start'] = d['Date'].dt.is_month_start.astype(int)
    d['is_month_end'] = d['Date'].dt.is_month_end.astype(int)
    le = LabelEncoder()
    d['Customer_enc'] = le.fit_transform(d['Customer Name'].astype(str))
    features = ['cust_avg', 'cust_count', 'month', 'dayofweek', 'is_month_start', 'is_month_end', 'Customer_enc']
    X = d[features].fillna(0)
    y = d['Document Total']
    return X, y, d, le

def train_and_compare_models(X, y, outdir):
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    results['LR'] = {'model': lr, 'rmse': mean_squared_error(y_test, pred_lr, squared=False), 'r2': r2_score(y_test, pred_lr)}
    # Decision Tree + basic grid
    dt = DecisionTreeRegressor(random_state=RANDOM_STATE)
    param_grid = {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10]}
    gs = GridSearchCV(dt, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    gs.fit(X_train, y_train)
    best_dt = gs.best_estimator_
    pred_dt = best_dt.predict(X_test)
    results['DT'] = {'model': best_dt, 'rmse': mean_squared_error(y_test, pred_dt, squared=False), 'r2': r2_score(y_test, pred_dt)}
    # Random Forest benchmark
    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results['RF'] = {'model': rf, 'rmse': mean_squared_error(y_test, pred_rf, squared=False), 'r2': r2_score(y_test, pred_rf)}
    # Save metrics and models
    metrics = []
    for name, info in results.items():
        metrics.append({'model': name, 'rmse': float(info['rmse']), 'r2': float(info['r2'])})
        joblib.dump(info['model'], os.path.join(outdir, f"model_{name}.joblib"))
    pd.DataFrame(metrics).to_csv(os.path.join(outdir, "model_comparison_metrics.csv"), index=False)
    return results, X_test, y_test

# ---------- Main ----------
def main(args):
    input_path = args.input
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Reading data from:", input_path)
    df = read_data(input_path, date_col=args.date_col)
    df = basic_clean(df)
    print(f"Rows: {len(df)}; date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    # Monthly timeseries
    ts_monthly = aggregate_time_series(df, freq='M')
    ts_monthly.to_csv(outdir / "monthly_timeseries.csv", index=False)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(ts_monthly['ds'], ts_monthly['y'], marker='o')
    ax.set_title('Monthly Sales (Document Total)')
    ax.set_xlabel('Date'); ax.set_ylabel('Sales')
    save_fig(fig, outdir / "monthly_sales.png")
    print("Saved monthly_sales.png")

    # Forecast
    method, model_obj, forecast_df = forecast(ts_monthly, periods=args.forecast_periods, prefer='prophet')
    if forecast_df is not None:
        if 'yhat' not in forecast_df.columns and 'y' in forecast_df.columns:
            forecast_df = forecast_df.rename(columns={'y': 'yhat'})
        forecast_df.to_csv(outdir / f"{method}_forecast.csv", index=False)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(ts_monthly['ds'], ts_monthly['y'], label='historic')
        if 'yhat' in forecast_df.columns:
            ax.plot(forecast_df['ds'], forecast_df['yhat'], label='forecast')
        else:
            ax.plot(forecast_df['ds'], forecast_df.iloc[:,1], label='forecast')
        ax.legend()
        save_fig(fig, outdir / f"{method}_forecast_plot.png")
        print(f"Forecast saved ({method}).")

    # RFM & segmentation
    rfm = rfm_features(df)
    rfm.to_csv(outdir / "rfm.csv", index=False)
    rfm2, kmeans = customer_segmentation(rfm, n_clusters=args.n_clusters)
    rfm2.to_csv(outdir / "rfm_segmented.csv", index=False)
    joblib.dump(kmeans, outdir / "customer_kmeans.joblib")
    print("RFM and segmentation saved.")

    # Churn labels & simple churn model
    labels = create_churn_label(df, days_window=args.churn_days_window)
    labels.to_csv(outdir / "churn_labels.csv", index=False)
    churn_df = rfm.merge(labels, on='Customer Name', how='left').fillna(0)
    Xc = churn_df[['Recency', 'Frequency', 'Monetary']].fillna(0)
    yc = churn_df['churn']
    clf = None
    if len(churn_df) >= 10 and yc.nunique() > 1:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xc, yc)
        churn_df['churn_prob'] = clf.predict_proba(Xc)[:,1]
        churn_df.to_csv(outdir / "churn_predictions.csv", index=False)
        joblib.dump(clf, outdir / "churn_model.joblib")
        print("Trained churn model.")
    else:
        print("Not enough churn-labeled customers to train a reliable model; skipped.")

    # Anomaly detection
    anomalies = anomaly_detection(df)
    anomalies.to_csv(outdir / "anomalies.csv", index=False)
    print("Anomaly detection saved.")

    # Regression predictions
    X, y, df_feats, cust_le = prepare_regression_features(df)
    results, X_test, y_test = train_and_compare_models(X, y, outdir)
    best_model_name = min(results.keys(), key=lambda k: results[k]['rmse'])
    best_model = results[best_model_name]['model']
    df_feats[f'predicted_invoice_value_{best_model_name}'] = best_model.predict(X)
    df_feats[['INV No', 'Date', 'Customer Name', 'Document Total', f'predicted_invoice_value_{best_model_name}']].to_csv(outdir / "invoice_predictions.csv", index=False)
    joblib.dump(cust_le, outdir / "customer_labelencoder.joblib")
    print("Regression models trained & predictions saved.")

    # Report
    report = {
        'rows': int(len(df)),
        'date_start': str(df['Date'].min().date()),
        'date_end': str(df['Date'].max().date()),
        'total_revenue': float(df['Document Total'].sum()),
        'best_model': best_model_name,
        'model_metrics': {k: {'rmse': float(v['rmse']), 'r2': float(v['r2'])} for k, v in results.items()}
    }
    with open(outdir / "report.json", 'w') as f:
        json.dump(report, f, indent=2)
    print("Report saved to", outdir)

# ---------- CLI ----------
def parse_args():
    parser = argparse.ArgumentParser(description="Sales ML pipeline (Windows-ready, no employee fields)")
    parser.add_argument('--input', required=False, help='Path to Excel (.xlsx/.xls) or CSV file')
    parser.add_argument('--output_dir', default='./output', help='Directory to save outputs')
    parser.add_argument('--date_col', default='Date', help='Name of the date column')
    parser.add_argument('--forecast_periods', type=int, default=12, help='Months to forecast')
    parser.add_argument('--n_clusters', type=int, default=4, help='Number of customer clusters')
    parser.add_argument('--churn_days_window', type=int, default=90, help='Days window for churn labeling')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # If no input provided, show file dialog on Windows
    if not args.input:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            print("No --input provided. Choose your invoice file (xlsx/csv)...")
            file_path = filedialog.askopenfilename(title="Select invoice file",
                                                   filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")])
            if not file_path:
                print("No file selected. Exiting.")
                sys.exit(0)
            args.input = file_path
        except Exception:
            print("tkinter file dialog not available. Please re-run with --input <path_to_file>")
            sys.exit(1)

    main(args)
