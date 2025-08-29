import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os


# Select the folder for which metrics should be generated
folder_path = 'model_predictions//allSources'
# folder_path = 'model_predictions//fast'
# folder_path = 'model_predictions//mediumFast'
# folder_path = 'model_predictions//slow'


EXPECTED_CLASS_SHARE = 100 / 3  # ≈ 33.33%

def generateMetrics(input_file, file_name):
    # Load classified dataset
    df = pd.read_csv(input_file)

    # Assumes columns: 'sentiment', 'predicted_class'
    y_true = df['sentiment']
    y_pred = df['predicted_class']

    # Calculate general metrics (using macro-average)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # False Positive Rate (macro average approximation)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    fpr = (FP / (FP + TN)).mean()

    # Display metrics as a table
    print(f"\n=== Metrics for: {file_name} ===")
    results = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)', 'FPR (Macro)'],
        'Score': [round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3), round(fpr, 3)]
    })
    print(results)

    # Calculate and print per-class statistics
    print("\nClass-wise recall vs. prediction distribution:")
    class_labels = ['Negative', 'Neutral', 'Positive']
    class_ids = [0, 1, 2]

    for i, label in zip(class_ids, class_labels):
        class_mask_true = (y_true == i)
        class_mask_pred = (y_pred == i)

        correct_pct = 100 * ((y_pred[class_mask_true] == i).sum() / class_mask_true.sum())
        predicted_pct = 100 * class_mask_pred.sum() / len(y_pred)

        correct_among_predictions = 100 * ((y_pred[class_mask_true] == i).sum()) / class_mask_pred.sum() if class_mask_pred.sum() > 0 else 0

        prediction_ratio = predicted_pct / EXPECTED_CLASS_SHARE
        efficiency_score = correct_pct / prediction_ratio if prediction_ratio != 0 else 0

        weighted_efficiency = efficiency_score * (predicted_pct / 100)

        # Custom F1-like score
        if correct_pct + predicted_pct > 0:
            balanced_f1 = 2 * correct_pct * predicted_pct / (correct_pct + predicted_pct)
        else:
            balanced_f1 = 0.0

        print(f"{label}: {correct_pct:.2f}% correct / {predicted_pct:.2f}% predicted | "
              f"Correct among Predictions: {correct_among_predictions:.2f}% | "
              f"Ratio: {prediction_ratio:.2f} | Efficiency Score: {efficiency_score:.2f} | "
              f"Weighted Efficiency: {weighted_efficiency:.2f} | " f"Balanced F1: {balanced_f1:.2f}")


        # print(f"{label}: {correct_pct:.2f}% correct / {predicted_pct:.2f}% predicted | Ratio: {prediction_ratio:.2f} | Efficiency Score: {efficiency_score:.2f}")

    # Confusion matrix heatmap (count and percentage)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title(f'Confusion Matrix (Count) - {file_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    conf_percent = cm / cm.sum() * 100
    sns.heatmap(conf_percent, annot=True, fmt='.2f', cmap='Oranges',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title(f'Confusion Matrix (Percentage) - {file_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Run for all CSVs in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        generateMetrics(file_path, filename)
