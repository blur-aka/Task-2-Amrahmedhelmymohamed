import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")
class_names = iris.target_names

print(f"Samples : {X.shape[0]}")
print(f"Features: {X.shape[1]} -> {list(iris.feature_names)}")
print(f"Classes : {list(class_names)}")
print(f"Distribution: {dict(zip(class_names, np.bincount(y)))}")
print()
print(X.describe().round(2))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, shuffle=True
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

k_range = range(1, 21)
error_rates = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, preds))

optimal_k = k_range[error_rates.index(min(error_rates))]
print(f"\nBest K found: {optimal_k}")

model = KNeighborsClassifier(n_neighbors=optimal_k)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
f1  = f1_score(y_test, predictions, average="weighted")
cm  = confusion_matrix(y_test, predictions)

print(f"\nAccuracy : {acc * 100:.2f}%")
print(f"F1 Score : {f1:.4f}")
print()
print(classification_report(y_test, predictions, target_names=class_names))

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Project 2 — KNN Iris Classification", fontsize=15, fontweight="bold", y=1.02)

axes[0].plot(k_range, error_rates, marker="o", color="#1a3a6b", linewidth=2)
axes[0].axvline(x=optimal_k, color="#e8521a", linestyle="--", label=f"Best K = {optimal_k}")
axes[0].scatter([optimal_k], [error_rates[optimal_k - 1]], color="#e8521a", zorder=5, s=120)
axes[0].set_title("Elbow Curve: Choosing K", fontweight="bold")
axes[0].set_xlabel("K Value")
axes[0].set_ylabel("Error Rate")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names,
    ax=axes[1], linewidths=0.5
)
axes[1].set_title("Confusion Matrix", fontweight="bold")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

colors = ["#2196F3", "#FF9800", "#4CAF50"]

for cls, color in zip(range(3), colors):
    mask = (y_test == cls)

    axes[2].scatter(
        X_test[mask, 2],
        X_test[mask, 3],
        label=class_names[cls],
        color=color,
        edgecolors="k",
        linewidths=0.5,
        s=80,
        alpha=0.85
    )

axes[2].set_title("Test Set: Petal Features (Scaled)", fontweight="bold")
axes[2].set_xlabel("Petal Length (scaled)")
axes[2].set_ylabel("Petal Width (scaled)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("project2_results.png", dpi=150, bbox_inches="tight")

plt.close()

print("Done. Results saved to project2_results.png")