from sklearn.svm import SVC
from src.main.python.models.runModels import run_predictions

# Linear SVC
svc = SVC(kernel='linear', C=1)
run_predictions(svc, "Linear SVC")

# RBF SVC
rbf_svc = SVC(kernel='rbf', gamma=0.7, C=1)
run_predictions(svc, "RBF SVC")

# Poly SVC
poly_svc = SVC(kernel='poly', degree=3, C=1)
run_predictions(poly_svc, 'POLY SVC')