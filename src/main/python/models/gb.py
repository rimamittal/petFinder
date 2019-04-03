from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from src.main.python.models.runModels import run_predictions

# Gradient Boosting Regressor
# gb = GradientBoostingRegressor(max_depth=10,
#                                n_estimators=200,
#                                random_state=2)
#
# run_predictions(gb, "Gradient Boosting Regressor")

# Gradient Boosting Classifier
gbm = GradientBoostingClassifier(random_state=15, n_estimators=250)
run_predictions(gbm, "Gradient Boosting Classifier")
