"""
Script untuk retrain model KNN dan scaler menggunakan scikit-learn 1.8.0+
Gunakan Python 3.13 (atau lingkungan yang sama dengan aplikasi Flask).
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["CustomerID"])
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
    df["CustomerType"] = np.where(df["Spending Score (1-100)"] >= 60, 1, 0)
    X = df[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    y = df["CustomerType"]
    return X, y


def find_optimal_k(X_train_scaled, y_train, X_val_scaled, y_val, k_max: int = 40):
    error_rates = []
    k_range = range(1, k_max + 1)
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred_k = knn.predict(X_val_scaled)
        error_rates.append(np.mean(y_pred_k != y_val))
    optimal_k = int(np.argmin(error_rates) + 1)

    # Simpan grafik elbow ke static agar dipakai di halaman Flask
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, error_rates, color="blue", linestyle="--", marker="o", markerfacecolor="red")
    plt.title("Elbow Method for Optimal K (Error Rate)")
    plt.xlabel("Nilai K")
    plt.ylabel("Error Rate")
    plt.xticks(range(1, k_max + 1, 2))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("static/elbow_plot.png")
    plt.close()

    return optimal_k


def main():
    X, y = load_and_prepare("Mall_Customers.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    optimal_k = find_optimal_k(X_train_scaled, y_train, X_test_scaled, y_test, k_max=40)
    print(f"Optimal K: {optimal_k}")

    model = KNeighborsClassifier(n_neighbors=optimal_k)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc * 100:.2f}%")

    joblib.dump(model, "knn_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Model dan scaler tersimpan: knn_model.pkl, scaler.pkl")


if __name__ == "__main__":
    main()

