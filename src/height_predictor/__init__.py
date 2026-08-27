from pickle import load
from utils import pretty_log as pl, log_title as lt
import pandas as pd

def main():
    # 1. Load .pkl intelligence file
    pkl_file = 'resources/height_prediction.pkl'
    predictor = load(open(pkl_file, 'rb'))      # rb ➡ Read Bytes

    lt('🖥️  Welcome to Height Predictor 🔮')
    weight_sample = pd.DataFrame([[60]], columns=['weight_kg'])
    height = predictor.predict(weight_sample)
    print(f'Weight: 60 kg ➡ {height} cm')

    # 2. Load .pkl gender classification file
    gc_pkl = 'resources/gender_classification.pkl'
    lt('🖥️  Welcome to Gender Predictor 🔮')
    classfier = load(open(gc_pkl, 'rb'))
    hw_sample = pd.DataFrame([[170, 60]], columns=['height_cm', 'weight_kg'])
    gender_pred = classfier.predict(hw_sample)
    gender_results = ['Male' if g == 1 else 'Female' for g in gender_pred]
    print(gender_results)