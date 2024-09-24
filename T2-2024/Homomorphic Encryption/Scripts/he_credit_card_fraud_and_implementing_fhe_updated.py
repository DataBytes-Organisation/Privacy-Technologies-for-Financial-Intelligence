# -*- coding: utf-8 -*-
"""HE Credit card fraud and implementing FHE.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18x0NejxAPDjPaAByHwxT2f1l3_Q0-Fph
"""

pip install faker

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import tenseal as ts
import json
import base64
from faker import Faker

# Initialize Faker
fake = Faker()

# Set the random seed for reproducibility
np.random.seed(42)

# Create lists for the synthetic data
names = [fake.name() for _ in range(1000)]
customer_ids = np.arange(1, 1001)
ages = np.random.randint(18, 80, size=1000)
transaction_amounts = np.random.uniform(100, 10000, size=1000)  # Random amounts between 100 and 10,000 AUD
credit_scores = np.random.randint(300, 850, size=1000)  # Random credit scores between 300 and 850

# Generate IsFraud with 85% non-fraud and 15% fraud
is_fraud = np.random.choice([0, 1], size=1000, p=[0.85, 0.15])

# Create the DataFrame
data = pd.DataFrame({
    'Cardholder_Name': names,
    'Customer_ID': customer_ids,
    'Age': ages,
    'Transaction_Amount_AUD': transaction_amounts,
    'Credit_Score': credit_scores,
    'IsFraud': is_fraud
})

# Preview the first few rows
print("Generated synthetic data (first 10 rows):")
print(data.head(10))

# Select features and target
X = data[['Transaction_Amount_AUD', 'Credit_Score']]
y = data['IsFraud']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.4, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a more powerful model (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Model Metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Printing model metrics
print("Model Metrics:")
print(f"ROC AUC: {roc_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# Initialize TenSEAL context for CKKS scheme (now with vector batching)
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.global_scale = 2**40
context.generate_galois_keys()

# Encrypt the predictions in a batched vector
encrypted_predictions = ts.ckks_vector(context, y_pred_proba.tolist())

# Perform homomorphic sum operation on the encrypted batch
encrypted_sum = encrypted_predictions.sum()

# Remove product calculation because it's not supported with CKKS
# encrypted_product = None  # CKKSVector does not support element-wise product

# Serialize the encrypted results for sending
serialized_sum = base64.b64encode(encrypted_sum.serialize()).decode('utf-8')

# Prepare the JSON response
response_data = {
    "encrypted_sum": serialized_sum,
    "model_metrics": {
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": class_report
    }
}

# Convert to JSON
response_json = json.dumps(response_data)

# Print the JSON response (In a real web application, this would be sent as an HTTP response)
print("JSON response with encrypted results:")
print(response_json)

# Decrypt the results for display
decrypted_sum = ts.ckks_vector_from(context, base64.b64decode(serialized_sum)).decrypt()[0]

# Interpret the decrypted results for end users
def interpret_risk(prob):
    if prob > 0.7:
        return "High Risk: This transaction is highly likely to be fraudulent. Immediate action is required."
    elif prob > 0.3:
        return "Moderate Risk: There’s a moderate risk associated with this transaction. Please verify it."
    else:
        return "Low Risk: This transaction appears to be normal. No further action is required."

sum_risk_message = interpret_risk(decrypted_sum)

# Print the interpreted messages
print("Risk Interpretation based on Sum:", sum_risk_message)

# Final JSON with risk interpretation
response_data.update({
    "risk_interpretation": {
        "sum_risk_message": sum_risk_message
    }
})

response_json = json.dumps(response_data)
print("Final JSON response with risk interpretation:")
print(response_json)

# Visualizations

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()

# Distribution Plot of Credit Score by Fraud Status
plt.figure(figsize=(10, 6))
sns.kdeplot(data.loc[data['IsFraud'] == 0, 'Credit_Score'], fill=True, label="Non-Fraudulent", color='green')
sns.kdeplot(data.loc[data['IsFraud'] == 1, 'Credit_Score'], fill=True, label="Fraudulent", color='red')
plt.title('Credit Score Distribution by Fraud Status')
plt.xlabel('Credit Score')
plt.ylabel('Density')
plt.legend()
plt.show()

# Boxplot of Transaction Amount by Fraud Status
plt.figure(figsize=(8, 6))
sns.boxplot(x='IsFraud', y='Transaction_Amount_AUD', data=data, palette='Set3')
plt.title('Transaction Amount by Fraud Status')
plt.xticks(ticks=[0, 1], labels=['Non-Fraudulent', 'Fraudulent'])
plt.show()

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Fraudulent', 'Fraudulent'],
            yticklabels=['Non-Fraudulent', 'Fraudulent'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

"""Objective:

The primary objective of this experiment is to build and evaluate a machine learning model that can detect fraudulent transactions. The task involves identifying whether a transaction is fraudulent or non-fraudulent based on features like transaction amount, credit score, and other synthetic features. Fraud detection is a critical task in financial institutions as it helps reduce fraudulent activity, minimize financial losses, and maintain customer trust.

Methodology:

Data Generation:

A synthetic dataset was created using the Faker library, which generated 1000 records of transactions with features such as Transaction_Amount_AUD (transaction amount in AUD), Credit_Score, and a binary target variable IsFraud indicating whether the transaction was fraudulent (1) or non-fraudulent (0).
The dataset was imbalanced, with 85% non-fraudulent transactions and 15% fraudulent transactions. This imbalance reflects real-world challenges where fraudulent transactions are much rarer than legitimate ones.

Modeling:

Random Forest Classifier was chosen for this task, as it is an ensemble method capable of handling complex classification tasks and improving over the simpler Logistic Regression model used in earlier experiments.
SMOTE (Synthetic Minority Over-sampling Technique) was applied to address the class imbalance by creating synthetic samples of the minority class (fraudulent transactions).

Features such as Transaction_Amount_AUD and Credit_Score were scaled using the StandardScaler for better model performance.
The dataset was split into training and testing sets, with 60% of the data used for training and 40% used for testing.

Model Evaluation:

After training the model, key evaluation metrics were generated: the ROC Curve, Confusion Matrix, and the Classification Report.

ROC AUC Score: The AUC score provides an aggregate measure of the model’s ability to distinguish between the two classes (fraudulent and non-fraudulent transactions). In this experiment, the ROC AUC score was 0.74, which is a significant improvement over the previous experiment that used logistic regression. However, there is still room for further improvement in distinguishing between the classes.

Confusion Matrix: The confusion matrix gives a detailed breakdown of the model's predictions. The model correctly identified 216 non-fraudulent transactions and 245 fraudulent ones but also misclassified 126 non-fraudulent transactions as fraudulent and 89 fraudulent transactions as non-fraudulent.

Classification Report: The classification report summarizes precision, recall, and F1-scores for both classes. Precision and recall for fraudulent transactions (class 1) were 0.66 and 0.73, respectively, while for non-fraudulent transactions (class 0), the model achieved a precision of 0.71 and a recall of 0.63.

Experiment Results:

ROC AUC:

The ROC AUC score of 0.74 shows that the model has a moderate ability to distinguish between fraudulent and non-fraudulent transactions. While this is a considerable improvement from the initial experiment (which had an AUC of 0.44), further work can be done to enhance this score.

Confusion Matrix:

The confusion matrix reveals that the model performed reasonably well, but it still misclassified 89 fraudulent transactions as non-fraudulent and 126 non-fraudulent transactions as fraudulent. This level of misclassification indicates a trade-off between catching fraudulent transactions and avoiding false positives (legitimate transactions flagged as fraud).

Classification Report:

Precision for non-fraudulent transactions (class 0) was 0.71, meaning 71% of transactions identified as non-fraudulent were indeed legitimate. Recall for class 0 was 0.63, meaning the model identified 63% of all legitimate transactions correctly.

For fraudulent transactions (class 1), precision was 0.66, indicating that 66% of transactions predicted as fraud were indeed fraudulent. The recall for fraudulent transactions was 0.73, meaning that 73% of actual fraudulent transactions were correctly identified by the model.

Analysis of the Outcome:

The results of this experiment demonstrate that using a more powerful model (Random Forest) and addressing class imbalance (with SMOTE) improved the model’s ability to detect fraud compared to the previous logistic regression model. However, there are still challenges with misclassifications, especially false positives (flagging legitimate transactions as fraud).

Key takeaways from this experiment:

Class Imbalance Handling: SMOTE was effective in balancing the classes and helped the model better focus on fraudulent transactions, leading to an improved recall for class 1 (fraud).

Model Improvement: The Random Forest model outperformed the logistic regression model from earlier experiments, achieving a much better AUC score and better handling of class imbalance.

ROC AUC Score: A score of 0.74 shows that the model has potential but still needs further tuning to improve accuracy, especially in reducing false positives and negatives.
"""


