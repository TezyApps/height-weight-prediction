
from utils import pretty_log as pl, log_title as lt
import pandas as pd

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
    # 4a. Linear - Scatter plot
    # 4b. Correlation matrix
    # 4c. Homoscadacity
    # 4d. No Multicollinearity
    # 4e. No AutoRegression
    # 4f. Zero Residual Mean
