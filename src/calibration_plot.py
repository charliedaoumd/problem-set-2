'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 
'''
# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Load the test datasets with predictions from both models
df_arrests_test_lr = pd.read_csv('data/df_arrests_test_with_predictions.csv') 
df_arrests_test_dt = pd.read_csv('data/df_arrests_test_with_decision_tree_predictions.csv') 

# Extract true labels and predicted probabilities
y_true_lr = df_arrests_test_lr['y']
pred_lr = df_arrests_test_lr['pred_lr']

y_true_dt = df_arrests_test_dt['y']
pred_dt = df_arrests_test_dt['pred_dt']

# Ensure predictions are probabilities (between 0 and 1)
print("Logistic Regression predictions summary:")
print(pred_lr.describe())
print("Decision Tree predictions summary:")
print(pred_dt.describe())

# Check for data consistency
print(f"Unique values in Logistic Regression predictions: {pred_lr.unique()}")
print(f"Unique values in Decision Tree predictions: {pred_dt.unique()}")

# Calculate calibration curves
def plot_calibration_curve(y_true, y_prob, model_name, n_bins=5):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label=f'{model_name} (Brier score: {brier_score_loss(y_true, y_prob):.4f})')

plt.figure(figsize=(12, 8))

# Plot calibration curve for Logistic Regression
plot_calibration_curve(y_true_lr, pred_lr, 'Logistic Regression')

# Plot calibration curve for Decision Tree
plot_calibration_curve(y_true_dt, pred_dt, 'Decision Tree')

# Formatting plot
plt.xlabel('Mean predicted value')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve')
plt.legend()
plt.grid(True)
plt.show()

# Determine which model is more calibrated
# Comparing Brier Scores for calibration
brier_lr = brier_score_loss(y_true_lr, pred_lr)
brier_dt = brier_score_loss(y_true_dt, pred_dt)

print(f"Brier score for Logistic Regression: {brier_lr:.4f}")
print(f"Brier score for Decision Tree: {brier_dt:.4f}")

if brier_lr < brier_dt:
    print("Logistic Regression is more calibrated.")
elif brier_lr > brier_dt:
    print("Decision Tree is more calibrated.")
else:
    print("Both models are equally calibrated.")