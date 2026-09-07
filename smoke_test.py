from pathlib import Path
import joblib, pandas as pd

ROOT = Path(__file__).resolve().parent
model_path = ROOT/'models/final_model.pkl'
data_path = ROOT/'data/WA_Fn-UseC_-HR-Employee-Attrition.csv'
assert model_path.exists() and data_path.exists()
bundle = joblib.load(model_path)
df = pd.read_csv(data_path)
from src import preprocessing, features
clean = preprocessing.clean_data(df)
fe = features.engineer_features(clean)
X = fe[bundle['feature_columns']].head(5)
proba = bundle['pipeline'].predict_proba(X)[:,1]
assert ((proba >= 0) & (proba <= 1)).all()
print('Employee attrition smoke test: PASS')
