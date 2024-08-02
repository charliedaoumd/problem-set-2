'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot


# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Run ETL process
    print("Running ETL process...")
    import etl

    # PART 2: Run preprocessing steps
    print("Running preprocessing...")
    import preprocessing

    # PART 3: Run logistic regression
    print("Running logistic regression...")
    import logistic_regression

    # PART 4: Run decision tree
    print("Running decision tree...")
    import decision_tree

    # PART 5: Run calibration plot
    print("Creating calibration plots...")
    import calibration_plot

if __name__ == "__main__":
    main()