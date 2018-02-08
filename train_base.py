import argparse
import pickle

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

import numpy as np

import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

import xgboost


FEATURES = ['is_Jan',
            'is_Feb',
            'is_Mar',
            'is_Apr',
            'is_May',
            'is_Jun',
            'is_Jul',
            'is_Aug',
            'is_Sep',
            'is_Oct',
            'is_Nov',
            'is_Dec',
            'is_g1',
            'is_g2',
            'is_g3',
            'is_turf',
            'is_dirt',
            'is_obstacle',
            'is_right',
            'is_left',
            'is_straight',
            'distance',
            'is_sunny',
            'is_cloudy',
            'is_rainy',
            'is_turf_good',
            'is_turf_slightly_heavy',
            'is_turf_heavy',
            'is_turf_bad',
            'is_dirt_good',
            'is_dirt_slightly_heavy',
            'is_dirt_heavy',
            'is_dirt_bad',
            'is_male',
            'is_female',
            'is_castrated',
            'age',
            'weight',
            'horse_weight',
            'horse_weight_difference',
            'enter_times',
            'win_rate',
            'mean_prise',
            'jocky_enter_times',
            'jocky_win_rate',
            'jocky_mean_prise',
            'trainer_times',
            'trainer_win_rate',
            'trainer_mean_prise',
            'owner_times',
            'owner_win_rate',
            'owner_mean_prise']


parser = argparse.ArgumentParser()
parser.add_argument('-i',
                    '--infile',
                    help='入力となる CSV ファイル',
                    type=str,
                    required=True)
parser.add_argument('-o',
                    '--outfile',
                    help='出力となる CSV ファイル',
                    type=str,
                    required=True)
args = parser.parse_args()
df = pd.read_csv(args.infile)

X = df[FEATURES].values
y = (df['order'] == 1.0).values
y = np.array([1 if i else 0 for i in y], dtype=np.int32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
count = int(sum(y_train))
rus = RandomUnderSampler(ratio={0: count, 1: count}, random_state=71)
X_train, y_train = rus.fit_sample(X_train, y_train)

xgb = xgboost.XGBClassifier()

xgb = GridSearchCV(
    xgboost.XGBClassifier(),
    {'learning_rate': [0.01, 0.05, 0.1, 0.2],
     'subsample': [0.5, 0.75, 1.0],
     'max_depth': [2, 4, 6],
     'n_estimators': [25, 50, 100, 200]},
    cv=4,
    scoring='f1',
    verbose=2,
    n_jobs=-1)
xgb.fit(X_train, y_train)

print('XGB')
print("Best parameters set found on development set: %s" % xgb.best_params_)

y_true, y_pred = y_test, xgb.predict(X_test)
print(classification_report(y_true, y_pred))
print("test accuracy: {}".format(accuracy_score(y_test, y_pred)))
print("confusion_matrix:")
print(confusion_matrix(y_test, y_pred))

with open(args.outfile, "wb") as f:
    pickle.dump(xgb, f)
