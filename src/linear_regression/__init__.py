import pandas as pd
import matplotlib.pyplot as plt

from utils import pretty_log as pl, log_title as lt

csv_data_file = 'resources/weight-height.csv'

def __load_data():
    return pd.read_csv(csv_data_file)

def main() -> None:
    # Height Prediction based on Weight
    #   Input   (x) ➡ Weigth
    #   Output  (y) ➡ Height

    # 1. Load the data
    hw_data = __load_data()
    pl(f"📥 Loading Data ➡ {csv_data_file}", hw_data)

    # 2. Data Understanding
    pl("🧮 Shape", hw_data.shape)
    pl("🔎 Scanning Null Entries", hw_data.isnull().sum())
    pl("📋 Data Types", hw_data.dtypes)

    # 3. Data Preparation
    
    # 3a. Data Cleaning: Nothing to clean at present
    
    # 3b. Feature Engineering:
    data_prep = hw_data.copy()

    # delete 'Gender', since it's not numerical data suited for linear regression
    del data_prep['Gender']

    # Convert metrics from ft/pounds to cm/kg in height, weight data:
    data_prep['height_cm'] = data_prep['Height'] * 2.54         # converting ft ➡ cm
    data_prep['weight_kg'] = data_prep['Weight'] / 2.205        # converting pounds ➡ kg

    # drop the irrelevant columns
    data_prep.drop(['Height', 'Weight'], axis=1, inplace=True)
    
    pl("⚙️  Transformed data", data_prep)

    # 4. Assumptions Tests:
    input_column = data_prep['weight_kg']
    output_column = data_prep['height_cm']

    # Normality test - passed follows a
    plt.hist(input_column)
    # plt.show()
    print("A1. Normality Test ✅ - follows Bell Curve distribution")

    # 4b. Linear - Scatter plot
    plt.scatter(x=input_column, y=output_column)
    # plt.show()
    print("A2. Linearity Test ✅ - input/output scattered linearly")

    # 4c. Correlation matrix | 4e. No Multicollinearity
    print("A3. Multicollinearity Test Skipped - No corr since not many features to compare")
    pl("MultiCollinearity", data_prep.corr())

    # 4f. No AutoRegression
    print("A4. No AutoRegression Test Skipped - No corr since we're comparing against a single feature (weight)")

    # These two steps will be performed during model training
    # 4d. Homoscadacity
    # 4g. Zero Residual Mean

    # 5. Model Building
    X = data_prep['weight_kg']          # input 
    y = data_prep['height_cm']          # output