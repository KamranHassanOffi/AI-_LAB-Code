# AI-_LAB-Code

# ================================
# Fraud Detection Agent Project (LightGBM Version, Clean Output)
# ================================

# 1. Dataset Acquisition
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier

# Load dataset
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
print("✅ Dataset Shape:", df.shape)
print(df.head())

# ================================
# 2. Data Preparation
# ================================
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 3. Agent Behavior Design (LightGBM)
# ================================
agent = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    is_unbalance=True,
    random_state=42,
    verbose=-1  # disables LightGBM info/warning messages
)
agent.fit(X_train, y_train)

# ================================
# 4. Implementation: Testing
# ================================
y_pred = agent.predict(X_test)
y_proba = agent.predict_proba(X_test)[:,1]

# ================================
# 5. Evaluation
# ================================
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("🎯 ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Fraud vs Non-Fraud distribution
fraud_count = df["Class"].value_counts()
print("\n🔎 Transaction Distribution in Dataset:")
print(fraud_count)
print(f"Non-Fraud: {fraud_count[0]} | Fraud: {fraud_count[1]}")

# Fraud vs Non-Fraud predictions
pred_count = pd.Series(y_pred).value_counts()
print("\n🤖 Model Predictions on Test Set:")
print(f"Predicted Non-Fraud: {pred_count.get(0,0)}")
print(f"Predicted Fraud: {pred_count.get(1,0)}")

# ================================
# 6. Visualization
# ================================
plt.figure(figsize=(6,4))
sns.countplot(x="Class", data=df)
plt.title("Distribution of Fraud vs Non-Fraud")
plt.show()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Fraud","Fraud"],
            yticklabels=["Not Fraud","Fraud"])
plt.title("Confusion Matrix")
plt.show()

importances = pd.Series(agent.feature_importances_, index=X.columns)
top_features = importances.nlargest(10)
top_features.plot(kind="barh", figsize=(8,5), title="Top 10 Important Features")
plt.show()

# ================================
# 7. Final Deliverable: Auto Sample Test
# ================================
sample = X_test.iloc[0]
prediction = agent.predict([sample])[0]
print("\n🧪 Auto Sample Test Transaction:")
print("Prediction:", "🚨 Fraud" if prediction==1 else "✅ Not Fraud")

# ================================
# 8. Optional Manual Test
# ================================
choice = input("\nDo you want to run manual test? (y/n): ").strip().lower()

if choice == "y":
    print("\n🧪 Manual Test Mode Enabled")
    try:
        time_val = input("Enter value for Time in seconds (press Enter for default 50000): ")
        time_val = float(time_val) if time_val.strip() != "" else 50000

        amount_val = input("Enter value for Amount in $ (press Enter for default 200): ")
        amount_val = float(amount_val) if amount_val.strip() != "" else 200
    except ValueError:
        print("⚠️ Invalid input. Using default values: Time=50000, Amount=200")
        time_val, amount_val = 50000, 200

    # Scale manually entered values using the SAME scaler
    time_scaled = scaler.transform([[time_val]])[0][0]
    amount_scaled = scaler.transform([[amount_val]])[0][0]

    # Use median for other features to avoid "random-looking" output
    sample = X_train.median().copy()
    sample['Time'] = time_scaled
    sample['Amount'] = amount_scaled

    prediction = agent.predict([sample])[0]
    proba = agent.predict_proba([sample])[0][1]

    print("\nManual Transaction Result:")
    print("Entered Time (seconds):", time_val)
    print("Entered Amount ($):", amount_val)
    print("Prediction:", "🚨 Fraud" if prediction == 1 else "✅ Not Fraud")
    print(f"Fraud Probability: {proba:.2f}")
