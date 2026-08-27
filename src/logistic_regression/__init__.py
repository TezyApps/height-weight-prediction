import pandas as pd
import sklearn.linear_model as lm
from sklearn.metrics import accuracy_score
from pickle import dump

from utils import pretty_log as pl, log_title as lt

def main():
    pl("Welcome to Logistic Regression!", "Gender prediction by Height & Weight")

    # 1. Load Data
    gender_classification = pd.read_csv('resources/weight-height.csv')
    pl('📥 Loading data', gender_classification)

    # 2. Data Understanding
    shape = gender_classification.shape
    null_entries = gender_classification.isnull().sum()
    data_types = gender_classification.dtypes
    # pl('💡 Data Understanding', (shape, '\n', null_entries, '\n', data_types))
    print('💡 Data Understanding')

    pl('shape', shape)
    pl('Null Entries', null_entries)
    pl('Data Types', data_types)

    # 3. Data preparation

    # 3a. Data cleaning 
    pl('🧹 Data Cleaning', 'Nothing to clean, skipped')

    # 3b. Feature Engineering
    data_prep = gender_classification.copy()
    data_prep['gender_val'] = data_prep['Gender'].replace(to_replace=['Male', 'Female'], value=[0,1]).astype('int64')
    # Convert metrics from ft/pounds to cm/kg in height, weight data:
    data_prep['height_cm'] = data_prep['Height'] * 2.54         # converting ft ➡ cm
    data_prep['weight_kg'] = data_prep['Weight'] / 2.205        # converting pounds ➡ kg

    data_prep.drop(
        ['Height', 'Gender', 'Weight'], 
        axis = 1,
        inplace = True
        )
    pl('⚙️  Transformed Data', data_prep)

    # 4. Assumptions?
    # same as linear regression ? - verify or not required here?

    # 5. Model Building
    X = data_prep.drop('gender_val', axis=1) # if axis = 0, does it drop row?
    y = data_prep['gender_val']
    pl('🔧 Model Building : X', (X, X.shape))
    pl('🔧 Model Building : y', (y, y.shape))

    # 6. Model Training
    logistic_regression_model = lm.LogisticRegression()
    logistic_regression_model.fit(X, y)

    # 6a. Deliverables
    m = logistic_regression_model.coef_
    y_intercept = logistic_regression_model.intercept_
    pl('🧮 Model Training is complete : m', m)
    pl('🧮 Model Training is complete : y_intercept', y_intercept)

    # 7. Model Testing
    y_pred = logistic_regression_model.predict(X)
    test_model = data_prep.copy()
    test_model['y_pred'] = y_pred
    pl('🧪 Model Testing for X', test_model)

    # 8. Model Evaluation
    score = accuracy_score(y, y_pred)
    pl('✨ Model Evaluation', f'Accuracy Score ➡ {(score * 100): .2f} %')
    diff = test_model[test_model['gender_val'] != test_model['y_pred']]
    pl(f' ≏ Differences : {len(diff)} records mismatching out of {len(test_model)} records ', diff)

    # 9. Model deployment
    pkl_file = 'resources/gender_classification.pkl'
    dump(logistic_regression_model, open(pkl_file, 'wb'))
    pl('📝 Model Deployment is complete', pkl_file)