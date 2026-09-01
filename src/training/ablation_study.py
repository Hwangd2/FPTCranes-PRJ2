import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def load_data(filepath, target_column="annual_salary_usd"):
    df = pd.read_csv(filepath)
    return df

def preprocess_features(df, use_cols=None, drop_cols=None):
    df = df.copy()
    # Step 1: Drop columns that are not for training regardless of scenario
    trash_cols = [
        'job_id', 'salary_min_usd', 'salary_max_usd',
        'salary_tier', 'experience_level', 'required_skills'
    ]
    cols_to_drop = [col for col in trash_cols if col in df.columns]
    
    if use_cols is not None:
        # Chỉ giữ lại các cột được chỉ định (nếu có) và loại bỏ cả cột rác nếu lỡ nằm trong use_cols
        use_cols_clean = [col for col in use_cols if col not in cols_to_drop]
        X = df[use_cols_clean]
    else:
        # Loại bỏ annual_salary_usd và trash cols (cộng thêm drop_cols tùy kịch bản)
        X = df.drop(columns=['annual_salary_usd'] + cols_to_drop, errors="ignore")
        if drop_cols is not None:
            # drop_cols có thể chứa cột bất kỳ cần loại (ví dụ ablation C/D)
            X = X.drop(columns=drop_cols, errors="ignore")
    # Step 2: One-hot encode all remaining categorical variables (object dtypes)
    X = pd.get_dummies(X, drop_first=False)
    return X

def train_and_eval(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    model = GradientBoostingRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    residuals = y_test - y_pred
    return score, y_test, y_pred, residuals, model

def ablation_study():
    file_path = "data/raw/ai_jobs_market_2025_2026.csv"
    df = load_data(file_path)
    df = df.rename(columns={'AI Engineering': 'job_category'})
    target = df["annual_salary_usd"]
    feature_cols = [col for col in df.columns if col != "annual_salary_usd"]

    # Model A: All features except trash columns (see preprocess_features)
    X_A = preprocess_features(df)
    score_A, y_test_A, y_pred_A, res_A, model_A = train_and_eval(X_A, target)
    
    # Model B: Only job_category and years_of_experience (after removing any trash cols)
    keep_cols = []
    for col in ['job_category', 'years_of_experience']:
        if col in df.columns:
            keep_cols.append(col)
    X_B = preprocess_features(df, use_cols=keep_cols)
    score_B, y_test_B, y_pred_B, res_B, model_B = train_and_eval(X_B, target)

    # Model C: All features except job_category (and always except trash columns)
    drop_cols_C = ['job_category'] if 'job_category' in df.columns else []
    X_C = preprocess_features(df, drop_cols=drop_cols_C)
    score_C, y_test_C, y_pred_C, res_C, model_C = train_and_eval(X_C, target)

    # Model D: All features except years_of_experience (and always except trash columns)
    drop_cols_D = ['years_of_experience'] if 'years_of_experience' in df.columns else []
    X_D = preprocess_features(df, drop_cols=drop_cols_D)
    score_D, y_test_D, y_pred_D, res_D, model_D = train_and_eval(X_D, target)

    print("Ablation Study Results (GradientBoostingRegressor, R2 score):")
    print(f"Model A (full features):              R2 = {score_A:.4f}")
    print(f"Model B (job_category + YoE only):    R2 = {score_B:.4f}")
    print(f"Model C (drop job_category):          R2 = {score_C:.4f}")
    print(f"Model D (drop years_of_experience):   R2 = {score_D:.4f}")

    # Plot residuals for Model A
    plot_residuals(y_test_A, y_pred_A, "residuals_model_A.png")

def plot_residuals(y_true, y_pred, filename):
    residuals = y_true - y_pred
    plt.figure(figsize=(8,6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted Salary (USD)")
    plt.ylabel("Residuals (True - Predicted)")
    plt.title("Residual Plot for Model A (Full Features)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    ablation_study()