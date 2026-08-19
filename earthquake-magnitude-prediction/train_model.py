# -*- coding: utf-8 -*-
"""
train_model.py
================
โมเดลทำนายระดับความรุนแรงของแผ่นดินไหว (Magnitude Level Classification)
จากตำแหน่งพื้นที่ (ละติจูด/ลองจิจูด) และความลึก (depth)

ขั้นตอน:
1. Import ข้อมูล
2. Exploratory Data Analysis (EDA)
3. Preprocessing (จัดการค่าว่าง/ค่าผิดปกติ, สร้าง label ระดับความรุนแรง)
4. Transform data (Feature scaling, Train/Test split)
5. สร้างโมเดล (RandomForestClassifier) และเปรียบเทียบกับโมเดลอื่น
6. ประเมินประสิทธิภาพของโมเดล (Accuracy, Classification report, Confusion matrix)
7. บันทึกโมเดล (model.pkl, scaler.pkl, label_encoder.pkl) สำหรับนำไปใช้ใน Streamlit
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ไม่ต้องเปิดหน้าต่างกราฟ (รันบน server ได้)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

RANDOM_STATE = 42
DATA_PATH = "data/thailand_earthquakes.csv"
MODEL_DIR = "model"


# ---------------------------------------------------------------------------
# 1. Import ข้อมูล (Load Data)
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    print("=" * 60)
    print("STEP 1: Import ข้อมูล")
    print("=" * 60)
    df = pd.read_csv(path)
    print(f"จำนวนแถว/คอลัมน์: {df.shape}")
    print(df.head())
    return df


# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# ---------------------------------------------------------------------------
def explore_data(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("STEP 2: Exploratory Data Analysis (EDA)")
    print("=" * 60)
    print("\nข้อมูลทั่วไป:")
    print(df.info())

    print("\nค่าว่าง (Missing values) เฉพาะคอลัมน์ที่ใช้งาน:")
    cols_used = ["latitude", "longitude", "depth", "mag"]
    print(df[cols_used].isnull().sum())

    print("\nสถิติเชิงพรรณนา (latitude, longitude, depth, mag):")
    print(df[cols_used].describe())

    # กราฟการกระจายตัวของ magnitude
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["mag"], bins=30, kde=True, ax=axes[0])
    axes[0].set_title("Distribution of Magnitude")

    sns.scatterplot(
        data=df, x="longitude", y="latitude", hue="mag", size="depth",
        palette="viridis", ax=axes[1], legend=False
    )
    axes[1].set_title("Earthquake Location (color = magnitude)")
    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/eda_overview.png", dpi=120)
    plt.close()
    print(f"\nบันทึกกราฟ EDA ไว้ที่ {MODEL_DIR}/eda_overview.png")


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
def classify_magnitude(mag: float) -> str:
    """แบ่งระดับความรุนแรงของแผ่นดินไหวตามเกณฑ์ USGS (Richter-based)"""
    if mag < 4.0:
        return "Minor"       # เบา
    elif mag < 5.0:
        return "Light"       # ค่อนข้างเบา
    elif mag < 6.0:
        return "Moderate"    # ปานกลาง
    elif mag < 7.0:
        return "Strong"      # รุนแรง
    else:
        return "Major"       # รุนแรงมาก


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 3: Preprocessing")
    print("=" * 60)

    cols_needed = ["latitude", "longitude", "depth", "mag"]
    df = df[cols_needed].copy()

    before = len(df)
    df = df.dropna(subset=cols_needed)
    print(f"ลบแถวที่มีค่าว่างในคอลัมน์ที่ใช้: {before - len(df)} แถว")

    # ค่า depth ที่ผิดปกติ (ติดลบ) ถือว่าไม่สมเหตุสมผล -> ตัดทิ้ง
    before = len(df)
    df = df[df["depth"] >= 0]
    print(f"ลบแถวที่ depth ติดลบ: {before - len(df)} แถว")

    # สร้าง target label: ระดับความรุนแรง
    df["mag_class"] = df["mag"].apply(classify_magnitude)

    print("\nการกระจายตัวของ class (mag_class):")
    print(df["mag_class"].value_counts())

    return df


# ---------------------------------------------------------------------------
# 4. Transform data (scaling + encoding + split)
# ---------------------------------------------------------------------------
def transform_data(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("STEP 4: Transform Data")
    print("=" * 60)

    feature_cols = ["latitude", "longitude", "depth"]
    X = df[feature_cols].values
    y_raw = df["mag_class"].values

    # Label Encoding สำหรับ target (string -> number)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    print("Class mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    # Train / Test split (stratify เพื่อรักษาสัดส่วน class เดิม)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train size: {X_train_scaled.shape}, Test size: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder, feature_cols


# ---------------------------------------------------------------------------
# 5. สร้างโมเดลและประเมินผล
# ---------------------------------------------------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder):
    print("\n" + "=" * 60)
    print("STEP 5: Train & Evaluate Models")
    print("=" * 60)

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",   # ข้อมูลไม่สมดุลระหว่าง class
            random_state=RANDOM_STATE,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "SVM": SVC(
            kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE
        ),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        results[name] = {"model": model, "accuracy": acc, "f1": f1}
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f} | Weighted F1: {f1:.4f}")

    # เลือกโมเดลที่ดีที่สุดจาก weighted F1 (เหมาะกับข้อมูลไม่สมดุล)
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = results[best_name]["model"]
    print(f"\n>>> โมเดลที่ดีที่สุด: {best_name} (F1={results[best_name]['f1']:.4f}) <<<")

    # รายงานผลอย่างละเอียดของโมเดลที่ดีที่สุด
    y_pred_best = best_model.predict(X_test)
    print("\nClassification Report:")
    target_names = label_encoder.classes_
    all_labels = list(range(len(target_names)))
    print(classification_report(
        y_test, y_pred_best, labels=all_labels,
        target_names=target_names, zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred_best, labels=all_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/confusion_matrix.png", dpi=120)
    plt.close()
    print(f"\nบันทึก confusion matrix ไว้ที่ {MODEL_DIR}/confusion_matrix.png")

    # Feature importance (เฉพาะกรณี RandomForest)
    if best_name == "RandomForest":
        importances = best_model.feature_importances_
        print("\nFeature importance (latitude, longitude, depth):")
        for f, imp in zip(["latitude", "longitude", "depth"], importances):
            print(f"  {f}: {imp:.4f}")

    return best_model, best_name, results


# ---------------------------------------------------------------------------
# 6. บันทึกโมเดล
# ---------------------------------------------------------------------------
def save_artifacts(model, scaler, label_encoder, feature_cols, model_name):
    print("\n" + "=" * 60)
    print("STEP 6: บันทึกโมเดล (Save Artifacts)")
    print("=" * 60)

    joblib.dump(model, f"{MODEL_DIR}/model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(label_encoder, f"{MODEL_DIR}/label_encoder.pkl")
    joblib.dump(feature_cols, f"{MODEL_DIR}/feature_cols.pkl")

    with open(f"{MODEL_DIR}/model_info.txt", "w", encoding="utf-8") as f:
        f.write(f"Best model: {model_name}\n")
        f.write(f"Features: {feature_cols}\n")
        f.write(f"Classes: {list(label_encoder.classes_)}\n")

    print(f"บันทึกไฟล์ลงในโฟลเดอร์ '{MODEL_DIR}/' เรียบร้อยแล้ว:")
    print("  - model.pkl           (โมเดลที่ผ่านการเทรน)")
    print("  - scaler.pkl          (StandardScaler สำหรับปรับสเกล feature)")
    print("  - label_encoder.pkl   (แปลง class number -> ชื่อระดับความรุนแรง)")
    print("  - feature_cols.pkl    (รายชื่อ feature ที่โมเดลใช้)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_data(DATA_PATH)
    explore_data(df)
    df_clean = preprocess_data(df)
    X_train, X_test, y_train, y_test, scaler, label_encoder, feature_cols = transform_data(df_clean)
    best_model, best_name, results = train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder)
    save_artifacts(best_model, scaler, label_encoder, feature_cols, best_name)

    print("\n" + "=" * 60)
    print("เสร็จสมบูรณ์! พร้อมนำโมเดลไปใช้กับเว็บแอป Streamlit (app.py)")
    print("=" * 60)


if __name__ == "__main__":
    main()
