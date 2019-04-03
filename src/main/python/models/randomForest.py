from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# Random Forest
from src.main.python.models.runModels import run_predictions

randomForest = RandomForestClassifier(random_state=1, n_estimators=200, max_depth=9)
run_predictions(randomForest, "Random Forest Classifier")

# Random Forest Regressor
# rf = RandomForestRegressor(n_estimators=300,
#                            min_samples_leaf=0.12,
#                            random_state=0)
# run_predictions(rf, "Random Forest Regressor")