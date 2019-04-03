from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier

classifiers = [
    GradientBoostingClassifier(random_state=15, n_estimators=250),
    xgb.XGBClassifier(
        max_depth=7, n_estimators=200),
    RandomForestClassifier(random_state=1, n_estimators=200, max_depth=9),
    DecisionTreeClassifier(random_state=1, max_depth=10)
]
