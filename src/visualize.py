import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import MODELS_ROI_DIR, MODELS_SQUARE_DIR, REPORT_DIR

# Colors per phone for plotting
PHONE_COLORS = {
    'nokia':    '#1f77b4',
    'samsung':  '#ff7f0e',
    'mi8 lite': '#2ca02c',
    'poco f3':  '#d62728',
    'redmia1':  '#9467bd',
}

def load_summary(path: str) -> pd.DataFrame:
    """
    Read a classification_summary.csv and return a DataFrame with raw metrics.
    Expects columns: ['phone','model','accuracy','std','f1_macro']
    """
    return pd.read_csv(path)

def plot_metric(df: pd.DataFrame, metric: str, title: str, out_path: str):
    """
    Plot bar chart of metric percentages for each model and phone.
    metric: one of 'accuracy' or 'f1_macro'.
    """
    df_pct = df.copy()
    df_pct[f'{metric}_pct'] = df_pct[metric] * 100
    pivot = df_pct.pivot(index='model', columns='phone', values=f'{metric}_pct')
    colors = [PHONE_COLORS.get(phone, '#555555') for phone in pivot.columns]
    ax = pivot.plot(kind='bar', figsize=(10, 6), color=colors)
    ax.set_ylabel(f"{metric.replace('_',' ').title()} (%)")
    ax.set_title(title)
    ax.legend(title='Phone', loc='best')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.clf()

def generate_report():
    """
    1) Read ROI and Square classification_summary.csv
    2) Plot accuracy and F1-macro for both pipelines
    3) Build a wide Excel report with columns:
       model, phone,
       ROI_accuracy, Square_accuracy,
       ROI_f1_macro, Square_f1_macro,
       diff_accuracy, diff_f1_macro,
       Roi_std, Square_std
       all formatted as percentages with one decimal
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    # load raw data
    roi_df = load_summary(os.path.join(MODELS_ROI_DIR, 'classification_summary.csv'))
    sq_df  = load_summary(os.path.join(MODELS_SQUARE_DIR, 'classification_summary.csv'))

    # plot metrics
    plot_metric(roi_df, 'accuracy', 'Accuracy per Model per Phone (ROI)',
                os.path.join(REPORT_DIR, 'accuracy_roi.png'))
    plot_metric(sq_df, 'accuracy', 'Accuracy per Model per Phone (Square)',
                os.path.join(REPORT_DIR, 'accuracy_square.png'))
    plot_metric(roi_df, 'f1_macro', 'F1-macro per Model per Phone (ROI)',
                os.path.join(REPORT_DIR, 'f1_roi.png'))
    plot_metric(sq_df, 'f1_macro', 'F1-macro per Model per Phone (Square)',
                os.path.join(REPORT_DIR, 'f1_square.png'))

    # prepare wide report using pivot_table
    acc_wide = (
        roi_df
        .pivot_table(index=['model', 'phone'], values='accuracy', aggfunc='first')
        .rename(columns={'accuracy': 'ROI_accuracy'})
        .reset_index()
    )
    sq_acc = (
        sq_df
        .pivot_table(index=['model', 'phone'], values='accuracy', aggfunc='first')
        .rename(columns={'accuracy': 'Square_accuracy'})
        .reset_index()
    )

    f1_wide = (
        roi_df
        .pivot_table(index=['model', 'phone'], values='f1_macro', aggfunc='first')
        .rename(columns={'f1_macro': 'ROI_f1_macro'})
        .reset_index()
    )
    sq_f1 = (
        sq_df
        .pivot_table(index=['model', 'phone'], values='f1_macro', aggfunc='first')
        .rename(columns={'f1_macro': 'Square_f1_macro'})
        .reset_index()
    )

    std_roi = (
        roi_df
        .pivot_table(index=['model', 'phone'], values='std', aggfunc='first')
        .rename(columns={'std': 'Roi_std'})
        .reset_index()
    )
    std_sq = (
        sq_df
        .pivot_table(index=['model', 'phone'], values='std', aggfunc='first')
        .rename(columns={'std': 'Square_std'})
        .reset_index()
    )

    # merge all
    report_df = (acc_wide
        .merge(sq_acc,  on=['model', 'phone'])
        .merge(f1_wide, on=['model', 'phone'])
        .merge(sq_f1,   on=['model', 'phone'])
        .merge(std_roi, on=['model', 'phone'])
        .merge(std_sq,  on=['model', 'phone'])
    )

    # compute differences
    report_df['diff_accuracy']  = report_df['ROI_accuracy']   - report_df['Square_accuracy']
    report_df['diff_f1_macro']  = report_df['ROI_f1_macro']   - report_df['Square_f1_macro']

    # format as percent with one decimal
    for col in [
        'ROI_accuracy', 'Square_accuracy',
        'ROI_f1_macro', 'Square_f1_macro',
        'diff_accuracy', 'diff_f1_macro',
        'Roi_std', 'Square_std'
    ]:
        report_df[col] = (report_df[col] * 100).round(1).astype(str) + '%'

    # write Excel
    excel_path = os.path.join(REPORT_DIR, 'metrics_comparison.xlsx')
    report_df.to_excel(excel_path, index=False)
    print(f"→ Saved Excel report: {excel_path}")

if __name__ == '__main__':
    generate_report()
