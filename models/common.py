import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled_df, scaler


def visualize_regression_performance(y_true, y_pred, algorithm_name="", save_path="regression_performance.png"):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, label="Tahminler")
    min_val = np.min([np.min(y_true), np.min(y_pred)])
    max_val = np.max([np.max(y_true), np.max(y_pred)])
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="İdeal y=x")
    plt.xlabel("Gerçek Değerler")
    plt.ylabel("Tahmin Edilen Değerler")
    r2 = r2_score(y_true, y_pred)
    plt.title(algorithm_name + f"Regression Performance (R² Score = {r2:.2f})")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print("Regression performance plot saved as:", save_path)


def visualize_confusion_matrix(y_true, y_pred, class_names, algorithm_name="", save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    title = f"{algorithm_name}: Confusion Matrix" if algorithm_name else "Confusion Matrix"
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    print("Confusion matrix saved as:", save_path)

def visualize_class_distribution(y_true, y_pred, class_names, algorithm_name="", save_path="class_distribution.png"):
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    distribution_true = {i: 0 for i in range(len(class_names))}
    distribution_pred = {i: 0 for i in range(len(class_names))}
    for val, cnt in zip(unique_true, counts_true):
        distribution_true[val] = cnt
    for val, cnt in zip(unique_pred, counts_pred):
        distribution_pred[val] = cnt
    values_true = [distribution_true[i] for i in range(len(class_names))]
    values_pred = [distribution_pred[i] for i in range(len(class_names))]
    index = np.arange(len(class_names))
    plt.figure(figsize=(8, 6))
    plt.bar(index - 0.2, values_true, width=0.4, label="Gerçek")
    plt.bar(index + 0.2, values_pred, width=0.4, label="Tahmin")
    plt.xticks(index, class_names)
    plt.xlabel("Sınıflar")
    plt.ylabel("Örnek Sayısı")
    title = f"{algorithm_name}: Class Distribution" if algorithm_name else "Class Distribution"
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print("Class distribution plot saved as:", save_path)

def print_classification_report_custom(y_true, y_pred, target_names, algorithm_name=""):
    report = classification_report(y_true, y_pred, target_names=target_names)
    if algorithm_name:
        print(f"{algorithm_name} Classification Report:\n{report}")
    else:
        print("Classification Report:\n", report)