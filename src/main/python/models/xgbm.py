import xgboost as xgb

# XG Boost
# from src.main.python.models.runModels import run_predictions
#
# xg_classifier = xgb.XGBClassifier(
#     max_depth=7, n_estimators=100)
# run_predictions(xg_classifier, 'XGBoost Classifier')


# XG Boost Regressor
# from src.main.python.models.runRegressionModels import run_regression_predictions
#
# xg_reg = xgb.XGBRegressor(
#     max_depth=7,
#     n_estimators=200)
# run_regression_predictions(xg_reg, 'XG Boost Regressor')


# stacking from sentiment analysis
from src.main.python.models.runStackingModels import run_stacking_predictions

xg_classifier = xgb.XGBClassifier(
    max_depth=7, n_estimators=200)
run_stacking_predictions(xg_classifier, 'XGBoost Classifier')


# countVectorizer -- doesnt work as xgb doesnt take vector
# from src.main.python.models.runModelWithsentiment import run_sentiment_predictions
#
# xg_classifier = xgb.XGBClassifier(
#     max_depth=7, n_estimators=100)
# run_sentiment_predictions(xg_classifier, 'XGBoost Classifier')
