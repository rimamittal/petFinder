import pandas as pd
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.naive_bayes import MultinomialNB
import numpy as np


def Stacking(model, train, y, test, n_fold):
    folds = StratifiedKFold(n_splits=n_fold, random_state=1)
    test_pred = pd.DataFrame()
    test_pred_proba = pd.DataFrame()
    train_pred = pd.DataFrame()
    train_pred_prob = pd.DataFrame()
    for train_indices, val_indices in folds.split(train, y.values):
        print("in for loop")
        x_train, x_val = train.iloc[train_indices], train.iloc[val_indices]
        y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

        model.fit(X=x_train, y=y_train)
        train_pred = pd.concat([train_pred, pd.Series(model.predict(x_val))], axis=0)
        train_pred_prob = pd.concat([train_pred_prob, pd.DataFrame(model.predict_proba(x_val))], axis=0)

        test_pred = pd.concat([test_pred, pd.Series(model.predict(test))], axis=1)
        test_pred_proba = pd.concat([test_pred_proba, pd.DataFrame(model.predict_proba(test))], axis=1)
    test_pred_proba['x0'] = test_pred_proba.iloc[:, 0] * test_pred_proba.iloc[:, 5]
    test_pred_proba['x1'] = test_pred_proba.iloc[:, 1] * test_pred_proba.iloc[:, 6]
    test_pred_proba['x2'] = test_pred_proba.iloc[:, 2] * test_pred_proba.iloc[:, 7]
    test_pred_proba['x3'] = test_pred_proba.iloc[:, 3] * test_pred_proba.iloc[:, 8]
    test_pred_proba['x4'] = test_pred_proba.iloc[:, 4] * test_pred_proba.iloc[:, 9]

    test_pred_proba = test_pred_proba[['x0', 'x1', 'x2', 'x3', 'x4']]

    return test_pred, train_pred, train_pred_prob, test_pred_proba


train = pd.read_csv('/Users/Rima/projects/petFinder/data/train/train.csv')
train['Description'].fillna('', inplace=True)
train['length'] = train['Description'].apply(len)
test = pd.read_csv('/Users/Rima/projects/petFinder/data/test/test.csv')
test['Description'].fillna('', inplace=True)
test['length'] = test['Description'].apply(len)

data_classes = train

x = data_classes['Description']
y = data_classes['AdoptionSpeed']
x_holdout = test['Description']


def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


vocab = CountVectorizer(analyzer=text_process).fit(x)
r0 = x[0]
vocab0 = vocab.transform([r0])

x = vocab.transform(x)
x = pd.DataFrame(x.todense(), columns=vocab.get_feature_names())

x_holdout = vocab.transform(x_holdout)
x_holdout = pd.DataFrame(x_holdout.todense(), columns=vocab.get_feature_names())

# density = (x.nnz / (x.shape[0] * x.shape[1])) * 100
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=101)

# mnb = MultinomialNB()
# mnb.fit(x_train, y_train)
# predmnb = mnb.predict(x_test)
# print("Confusion Matrix for Multinomial Naive Bayes:")
# print(confusion_matrix(y_test, predmnb))
# print("Score:", round(accuracy_score(y_test, predmnb) * 100, 2))
# print("Classification Report:", classification_report(y_test, predmnb))

from sklearn.ensemble import RandomForestClassifier

rmfr = RandomForestClassifier(n_estimators=150)
# rmfr.fit(x_train, y_train)
# predrmfr = rmfr.predict(x_test)
# print("Confusion Matrix for Random Forest Classifier:")
# print(confusion_matrix(y_test, predrmfr))
# print("Score:", round(accuracy_score(y_test, predrmfr) * 100, 2))
# print("Classification Report:", classification_report(y_test, predrmfr))

# from sklearn.tree import DecisionTreeClassifier
#
# dt = DecisionTreeClassifier()
# dt.fit(x_train, y_train)
# preddt = dt.predict(x_test)
# print("Confusion Matrix for Decision Tree:")
# print(confusion_matrix(y_test, preddt))
# print("Score:", round(accuracy_score(y_test, preddt) * 100, 2))
# print("Classification Report:", classification_report(y_test, preddt))

# from xgboost import XGBClassifier
#
# xgb = XGBClassifier()
test_predicted_sentiment_class, train_predicted_sentiment_class, train_predicted_sentiment, test_predicted_sentiment = Stacking(
    model=rmfr, n_fold=10, train=x, test=x_holdout, y=y)

# from sklearn.neural_network import MLPClassifier
#
# mlp = MLPClassifier()
# mlp.fit(x_train, y_train)
# predmlp = mlp.predict(x_test)
# print("Confusion Matrix for Multilayer Perceptron Classifier:")
# print(confusion_matrix(y_test, predmlp))
# print("Score:", round(accuracy_score(y_test, predmlp) * 100, 2))
# print("Classification Report:")
# print(classification_report(y_test, predmlp))


# from sklearn.neural_network import MLPClassifier
#
# mlp = MLPClassifier()
# mlp.fit(x_train, y_train)
# predmlp = mlp.predict(x_test)
# print("Score:", round(accuracy_score(y_test, predmlp) * 100, 2))
#
# train_predicted_sentiment = pd.DataFrame(mlp.predict_proba(x))
# train_predicted_sentiment_class = pd.DataFrame(mlp.predict(x))
# test_predicted_sentiment = pd.DataFrame(mlp.predict_proba(x_holdout))
# test_predicted_sentiment_class = pd.DataFrame(mlp.predict(x_holdout))

train_predicted_sentiment.to_csv('/Users/Rima/projects/petFinder/data/v7/train_predicted_sentiment.csv', index=False)
train_predicted_sentiment_class.to_csv('/Users/Rima/projects/petFinder/data/v7/train_predicted_sentiment_class.csv',
                                       index=False)
test_predicted_sentiment.to_csv('/Users/Rima/projects/petFinder/data/v7/test_predicted_sentiment.csv', index=False)
test_predicted_sentiment_class.to_csv('/Users/Rima/projects/petFinder/data/v7/test_predicted_sentiment_class.csv',
                                      index=False)
