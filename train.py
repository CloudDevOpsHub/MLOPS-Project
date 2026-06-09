# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# -----------------------------
# 1. Create IT Career Dataset
# -----------------------------
data = {
    "Experience":        [1, 2, 3, 4, 5, 6, 2, 3, 7, 8, 10, 1, 4, 6],
    "CurrentPackage":    [4, 6, 8, 10, 12, 15, 5, 9, 18, 22, 30, 3, 11, 14],
    "SkillsCount":       [2, 3, 4, 5, 6, 7, 2, 4, 8, 9, 10, 1, 6, 7],
    "Certifications":    [0, 1, 1, 2, 3, 4, 0, 1, 5, 6, 8, 0, 2, 3],
    "CodingLevel":       [3, 4, 5, 6, 7, 8, 4, 6, 9, 9, 10, 2, 7, 8],
    # 1 → Upskilling Recommended | 0 → Strong Package Level
    "Outcome":           [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0]
}

df = pd.DataFrame(data)

print("✅ Dataset Loaded")
print("✅ Columns:", df.columns.tolist())
print(df.head())

# -----------------------------
# 2. Prepare Features & Target
# -----------------------------
X = df[[
    "Experience",
    "CurrentPackage",
    "SkillsCount",
    "Certifications",
    "CodingLevel"
]]
y = df["Outcome"]

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 4. Train ML Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy:.2f}")

# -----------------------------
# 6. Save Model
# -----------------------------
joblib.dump(model, "it_package_model.pkl")
print("✅ Model saved as it_package_model.pkl")
