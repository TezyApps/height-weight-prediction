from pickle import load
from utils import pretty_log as pl, log_title as lt

def main():
    # 1. Load .pkl intelligence file
    pkl_file = 'resources/height_prediction.pkl'
    predictor = load(open(pkl_file, 'rb'))      # rb ➡ Read Bytes

    lt('🖥️  Welcome to Height Predictor 🔮')
    height = predictor.predict([[60]])
    print(f'Weight: 60 kg ➡ {height} cm')