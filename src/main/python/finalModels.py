import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

# data_test = pd.read_pickle('~/projects/petFinder/data/test_complete.pkl')
data_test = pd.read_pickle('~/projects/petFinder/data/v2/test_complete.pkl')
X_test = data_test.drop(['Name',
                         'PetID',
                         'AdoptionSpeed',
                         'Description'], axis=1)

# data = pd.read_pickle('~/projects/petFinder/data/train_complete.pkl')
data = pd.read_pickle('~/projects/petFinder/data/v2/train_complete.pkl')
X = data.drop(['Name',
               'PetID',
               'AdoptionSpeed',
               'Description'], axis=1)
y = data['AdoptionSpeed']

# XG Boost Classifier
xg_classifier = xgb.XGBClassifier(objective='reg:logistic', learning_rate=0.1,
                                  max_depth=10, n_estimators=200)

xg_classifier.fit(X, y)
y_pred_xg = xg_classifier.predict(X_test).round()

# Gradient Boosting Classifier
gbm = GradientBoostingClassifier(random_state=10, n_estimators=200)
gbm.fit(X, y)
y_pred_gbm = gbm.predict(X_test).round()

petid = data_test['PetID']

final_pred_xg = pd.Series(y_pred_xg, name='AdoptionSpeed')
submission_xg = pd.concat([petid, final_pred_xg], axis=1)
# submission_xg.to_csv('~/projects/petFinder/data/xg/submission.csv', index=False)
submission_xg.to_csv('~/projects/petFinder/data/v3/xgsubmission.csv', index=False)

final_pred_gbm = pd.Series(y_pred_gbm, name='AdoptionSpeed')
submission_gbm = pd.concat([petid, final_pred_gbm], axis=1)
# submission_gbm.to_csv('~/projects/petFinder/data/gbm/submission.csv', index=False)
submission_gbm.to_csv('~/projects/petFinder/data/v3/gbmsubmission.csv', index=False)
