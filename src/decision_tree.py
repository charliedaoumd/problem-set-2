'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

# Import necessary packages
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Step 1: Read in the dataframe(s) from PART 3
df_arrests_train = pd.read_csv('data/df_arrests_train_with_predictions.csv')
df_arrests_test = pd.read_csv('data/df_arrests_test_with_predictions.csv')

# Prepare the features and target for the Decision Tree model
features = ['age_at_arrest', 'num_fel_arrests_last_year']
X_train = df_arrests_train[features]
y_train = df_arrests_train['pred_lr']
X_test = df_arrests_test[features]
y_test = df_arrests_test['pred_lr']

# Step 2: Create a parameter grid for Decision Tree depth
param_grid_dt = {
    'max_depth': [3, 5, 7]  # Example values for maximum tree depth
}

# Step 3: Initialize the Decision Tree model
dt_model = DecisionTreeClassifier()

# Step 4: Initialize GridSearchCV with Decision Tree and parameter grid
gs_cv_dt = GridSearchCV(
    estimator=dt_model,
    param_grid=param_grid_dt,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',  # Accuracy as the scoring metric
    n_jobs=-1  # Use all available CPU cores
)

# Step 5: Fit the Decision Tree model
print("Fitting the Decision Tree model to find the optimal depth...")
gs_cv_dt.fit(X_train, y_train)

# Step 6: Determine the optimal value for max_depth
best_depth = gs_cv_dt.best_params_['max_depth']
regularization = "most regularization" if best_depth == max(param_grid_dt['max_depth']) else "least regularization" if best_depth == min(param_grid_dt['max_depth']) else "in the middle"

print(f"The optimal value for max_depth is: {best_depth}")
print(f"This value represents: {regularization}")

# Step 7: Predict outcomes on the test set
df_arrests_test['pred_dt'] = gs_cv_dt.predict(X_test)

# Step 8: Save the resulting dataframe(s) for use in PART 5
output_file_dt = 'data/df_arrests_test_with_decision_tree_predictions.csv'
df_arrests_test.to_csv(output_file_dt, index=False)

print(f"Decision Tree predictions saved to {output_file_dt}. Ready for the next phase.")