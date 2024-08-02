'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: pred_universe, num_fel_arrests_last_year
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression


# Your code here
df_arrests = pd.read_csv('data/df_arrests.csv')

features = ['age_at_arrest', 'num_fel_arrests_last_year']
X = df_arrests[features]
y = df_arrests['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, stratify=y, random_state=42
)

param_grid = {'C': [0.1, 1, 10]}  # Values for the hyperparameter C

lr_model = LogisticRegression(solver='liblinear')  # 'liblinear' is suitable for small datasets

# Step 5: Initialize GridSearchCV using the logistic regression model and parameter grid
gs_cv = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=5, scoring='accuracy')

# Step 6: Run the model
gs_cv.fit(X_train, y_train)

# Step 7: Determine the optimal value for C and evaluate its regularization effect
optimal_C = gs_cv.best_params_['C']
print(f"Optimal value for C: {optimal_C}")

# Evaluate regularization based on the value of C
if optimal_C < 1:
    regularization = "Most Regularization"
elif optimal_C == 1:
    regularization = "In the Middle"
else:
    regularization = "Least Regularization"
    
print(f"Value C: {regularization}.")

# Step 8: Predict for the test set and add predictions to the dataframe
y_pred_test = gs_cv.predict(X_test)
y_pred_train = gs_cv.predict(X_train)

X_test_with_predictions = X_test.copy()
X_test_with_predictions['y'] = y_test.values
X_test_with_predictions['pred_lr'] = y_pred_test

X_train_with_predictions = X_train.copy()
X_train_with_predictions['y'] = y_train.values
X_train_with_predictions['pred_lr'] = y_pred_train

X_test_with_predictions.to_csv('data/df_arrests_test_with_predictions.csv', index=False)
X_train_with_predictions.to_csv('data/df_arrests_train_with_predictions.csv', index=False)

print("Dataframes with predictions saved. Ready for the next phase.")


