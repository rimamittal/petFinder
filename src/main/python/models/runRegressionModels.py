import pandas as pd
from matplotlib import pyplot
from numpy import sort
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from xgboost import plot_importance, XGBClassifier, XGBRegressor
from sklearn.feature_selection import SelectFromModel
import numpy as np


def getcontinous_y_values(y):
    if y == 0:
        return -1
    if y == 1:
        return 1
    if y == 2:
        return 2
    if y == 3:
        return 3
    if y == 4:
        return 4


def getDiscreteValues(y):
    y = round(y)
    if y == -1:
        return 0
    return y


def run_regression_predictions(modelObject, modelName):
    # data = pd.read_pickle('~/projects/petFinder/data/train_complete.pkl')
    data = pd.read_pickle('/Users/Rima/projects/petFinder/data/v4/complete_train.pkl')

    # some last minute EDA
    categoricals = ['Dewormed', 'FurLength', 'Gender', 'Health', 'State', 'Sterilized', 'Type', 'Vaccinated',
                    'color1Name', 'color2Name', 'color3Name']
    data = pd.get_dummies(data, columns=categoricals)

    X = data.drop(['Name',
                   'PetID',
                   'AdoptionSpeed',
                   'Description',
                   'labels',
                   'color1',
                   'color2',
                   'color3',
                   'breed1Name',
                   'breed2Name',
                   'joyLikelihood',
                   'sorrowLikelihood',
                   'angerLikelihood',
                   'surpriseLikelihood',
                   'underExposedLikelihood',
                   'blurredLikelihood',
                   'headwearLikelihood',
                   'iscolor3Matching'], axis=1)
    y = data['AdoptionSpeed']

    data_test = pd.read_pickle('/Users/Rima/projects/petFinder/data/v4/complete_test.pkl')
    data_test = pd.get_dummies(data_test, columns=categoricals)

    X_holdout = data_test.drop(['Name',
                                'PetID',
                                'AdoptionSpeed',
                                'Description',
                                'labels',
                                'color1',
                                'color2',
                                'color3',
                                'breed1Name',
                                'breed2Name',
                                'joyLikelihood',
                                'sorrowLikelihood',
                                'angerLikelihood',
                                'surpriseLikelihood',
                                'underExposedLikelihood',
                                'blurredLikelihood',
                                'headwearLikelihood',
                                'iscolor3Matching'], axis=1)

    missing_cols = set(X.columns) - set(X_holdout.columns)
    for c in missing_cols:
        X_holdout[c] = 0

    X_holdout = X_holdout[X.columns]

    # y = data['AdoptionSpeed']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train = y_train.apply(lambda x: getcontinous_y_values(x))
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    #
    modelObject.fit(x_train, y_train)

    # print(modelObject.feature_importances_)

    # pyplot.bar(range(len(modelObject.feature_importances_)), modelObject.feature_importances_)
    # pyplot.show()

    # plot feature importance - for xgboost
    # plot_importance(modelObject)
    # pyplot.show()

    y_pred_test_reg = pd.Series(modelObject.predict(x_test))
    print(y_pred_test_reg.head(5))
    y_pred_test = y_pred_test_reg.apply(lambda x: getDiscreteValues(x))
    print(y_test.head(5))

    y_pred_train_reg = pd.Series(modelObject.predict(x_train))

    y_pred_train = y_pred_train_reg.apply(lambda x: getDiscreteValues(x))

    y_pred_holdout = pd.Series(modelObject.predict(X_holdout))

    # train_results = pd.DataFrame(
    #     {'y_pred_train': y_pred_train,
    #      'y_train': y_train
    #      })
    #
    # test_results = pd.DataFrame(
    #     {'y_pred_test': y_pred_test,
    #      'y_test': y_test
    #      })

    # print(train_results.head(20))
    # print(test_results.head(20))

    test_acc = accuracy_score(y_test, y_pred_test)
    train_acc = accuracy_score(y_train, y_pred_train)

    kappa = cohen_kappa_score(y_test, y_pred_test, weights='quadratic')

    print("Test set accuracy by " + modelName + " : {:.2f}".format(test_acc))
    print("unique adoption speeds in y_test = ", y_test.unique())
    print("unique adoption speeds in y_pred = ", np.unique(y_pred_test))

    print("Train set accuracy by " + modelName + " : {:.2f}".format(train_acc))
    print("unique adoption speeds in y_train = ", y_train.unique())
    print("unique adoption speeds in y_pred_train = ", np.unique(y_pred_train))

    print("Kappa by " + modelName + " : {:.2f}".format(kappa))

    print(confusion_matrix(y_test, y_pred_test))

    ## Feature Selection
    # print("Trying feature selection using thresholds")
    # thresholds = sort(modelObject.feature_importances_)
    # print(thresholds)

    # for thresh in thresholds:
    #     # select features using threshold
    #     selection = SelectFromModel(modelObject, threshold=thresh, prefit=True)
    #     select_X_train = selection.transform(x_train)
    #     # train model
    #     selection_model = XGBRegressor()
    #     selection_model.fit(select_X_train, y_train)
    #     # eval model
    #     select_X_test = selection.transform(x_test)
    #     y_pred = selection_model.predict(select_X_test)
    #     predictions = [round(value) for value in y_pred]
    #     accuracy = accuracy_score(y_test, predictions)
    #     print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))
    ## Feature Selection

    # fitting this on test set
    print('Fitting the Model on Test Set')

    # data_test = pd.read_pickle('/Users/Rima/projects/petFinder/data/v4/complete_test.pkl')
    # data_test = pd.get_dummies(data_test, columns=categoricals)
    #
    # X_holdout = data_test.drop(['Name',
    #                             'PetID',
    #                             'AdoptionSpeed',
    #                             'Description',
    #                             'labels',
    #                             'color1',
    #                             'color2',
    #                             'color3',
    #                             'breed1Name',
    #                             'breed2Name'], axis=1)
    #
    # missing_cols = set(X.columns) - set(X_holdout.columns)
    # for c in missing_cols:
    #     X_holdout[c] = 0
    #
    # X_holdout = X_holdout[X.columns]

    # predictions = modelObject.predict(X_holdout).round()
    predictions = y_pred_holdout.apply(lambda x: getDiscreteValues(x))
    petid = data_test['PetID']
    final_pred = pd.Series(predictions, name='AdoptionSpeed')
    submission = pd.concat([petid, final_pred], axis=1)
    submission.to_csv('/Users/Rima/projects/petFinder/data/v4/submission7.csv', index=False)
