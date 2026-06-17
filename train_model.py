from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd, joblib

# sesuaikan dengan notebook asli
df = pd.read_csv('hotel_bookings.csv')
X = df.drop(columns=['is_canceled'])
y = df['is_canceled']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# tambahkan preprocessing/encoding sesuai notebook
smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(...)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    random_state=42
)
