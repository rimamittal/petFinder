import pandas as pd
from matplotlib import pyplot
from numpy import sort
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from xgboost import plot_importance, XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import numpy as np


def run_stacking_predictions(modelObject, modelName):
    # data = pd.read_pickle('~/projects/petFinder/data/train_complete.pkl')
    pd.set_option('display.max_columns', 100)

    test_predictions_prob = pd.read_csv('/Users/Rima/projects/petFinder/data/v7/test_predicted_sentiment.csv')
    # test_predictions_prob = test_predictions_prob.iloc[:, 1:6]
    test_predictions_prob.columns = ['x0', 'x1', 'x2', 'x3', 'x4']

    # test_predictions = pd.read_csv('/Users/Rima/projects/petFinder/data/v5/test_predicted_sentiment_class.csv')
    # test_predictions = test_predictions['0']

    train_predictions_prob = pd.read_csv('/Users/Rima/projects/petFinder/data/v7/train_predicted_sentiment.csv')
    # train_predictions_prob = train_predictions_prob.iloc[:, 1:6]
    train_predictions_prob.columns = ['x0', 'x1', 'x2', 'x3', 'x4']

    # train_predictions = pd.read_csv('/Users/Rima/projects/petFinder/data/v5/train_predicted_sentiment_class.csv')
    # train_predictions = train_predictions['0']

    data = pd.read_pickle('/Users/Rima/projects/petFinder/data/v4/complete_train.pkl')
    print(data.shape)
    data = pd.concat([train_predictions_prob , data], axis=1)
    print(data.shape)


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
                   'iscolor3Matching',
                   'RescuerID'], axis=1)
    y = data['AdoptionSpeed']

    data_test = pd.read_pickle('/Users/Rima/projects/petFinder/data/v4/complete_test.pkl')
    print(data_test.shape)
    data_test = pd.concat([test_predictions_prob, data_test], axis=1)
    print(data_test.shape)

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
                                'iscolor3Matching',
                                'RescuerID'], axis=1)

    missing_cols = set(X.columns) - set(X_holdout.columns)
    for c in missing_cols:
        X_holdout[c] = 0

    X_holdout = X_holdout[X.columns]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



    # class 0 vs rest
    print("fitting 1st")
    y0 = (y_train == 0).astype('int')
    modelObject.fit(x_train, y0)
    y_pred0_test = pd.DataFrame(modelObject.predict_proba(x_test)[:, 0] * 0.8)
    y_pred0_train = pd.DataFrame(modelObject.predict_proba(x_train)[:, 0] * 0.8)
    y_pred0_holdout = pd.DataFrame(modelObject.predict_proba(X_holdout)[:, 0] * 0.8)
    y_pred_test = y_pred0_test
    y_pred_train = y_pred0_train
    y_pred_holdout = y_pred0_holdout

    # class 1 vs rest
    print("fitting 2nd")
    y1 = (y_train == 1).astype('int')
    modelObject.fit(x_train, y1)
    y_pred1_test = pd.DataFrame(modelObject.predict_proba(x_test)[:, 0])
    y_pred1_train = pd.DataFrame(modelObject.predict_proba(x_train)[:, 0])
    y_pred1_holdout = pd.DataFrame(modelObject.predict_proba(X_holdout)[:, 0])
    y_pred_test = pd.concat([y_pred_test, y_pred1_test], axis=1)
    y_pred_train = pd.concat([y_pred_train, y_pred1_train], axis=1)
    y_pred_holdout = pd.concat([y_pred_holdout, y_pred1_holdout], axis=1)

    # class 2 vs rest
    print("fitting 3rd")
    y2 = (y_train == 2).astype('int')
    modelObject.fit(x_train, y2)
    y_pred2_test = pd.DataFrame(modelObject.predict_proba(x_test)[:, 0])
    y_pred2_train = pd.DataFrame(modelObject.predict_proba(x_train)[:, 0])
    y_pred2_holdout = pd.DataFrame(modelObject.predict_proba(X_holdout)[:, 0])
    y_pred_test = pd.concat([y_pred_test, y_pred2_test], axis=1)
    y_pred_train = pd.concat([y_pred_train, y_pred2_train], axis=1)
    y_pred_holdout = pd.concat([y_pred_holdout, y_pred2_holdout], axis=1)

    # class 3 vs rest
    print("fitting 4th")
    y3 = (y_train == 3).astype('int')
    modelObject.fit(x_train, y3)
    y_pred3_test = pd.DataFrame(modelObject.predict_proba(x_test)[:, 0])
    y_pred3_train = pd.DataFrame(modelObject.predict_proba(x_train)[:, 0])
    y_pred3_holdout = pd.DataFrame(modelObject.predict_proba(X_holdout)[:, 0])
    y_pred_test = pd.concat([y_pred_test, y_pred3_test], axis=1)
    y_pred_train = pd.concat([y_pred_train, y_pred3_train], axis=1)
    y_pred_holdout = pd.concat([y_pred_holdout, y_pred3_holdout], axis=1)

    # class 4 vs rest
    print("fitting 5th")
    y4 = (y_train == 4).astype('int')
    modelObject.fit(x_train, y4)
    y_pred4_test = pd.DataFrame(modelObject.predict_proba(x_test)[:, 0])
    y_pred4_train = pd.DataFrame(modelObject.predict_proba(x_train)[:, 0])
    y_pred4_holdout = pd.DataFrame(modelObject.predict_proba(X_holdout)[:, 0])
    y_pred_test = pd.concat([y_pred_test, y_pred4_test], axis=1)
    y_pred_train = pd.concat([y_pred_train, y_pred4_train], axis=1)
    y_pred_holdout = pd.concat([y_pred_holdout, y_pred4_holdout], axis=1)

    y_pred_test.columns = ['not0', 'not1', 'not2', 'not3', 'not4']
    y_pred_train.columns = ['not0', 'not1', 'not2', 'not3', 'not4']
    y_pred_holdout.columns = ['not0', 'not1', 'not2', 'not3', 'not4']

    y_pred_test['0'] = y_pred_test['not1'] + y_pred_test['not2'] + y_pred_test['not3'] + y_pred_test['not4']
    y_pred_test['1'] = y_pred_test['not0'] + y_pred_test['not2'] + y_pred_test['not3'] + y_pred_test['not4']
    y_pred_test['2'] = y_pred_test['not1'] + y_pred_test['not0'] + y_pred_test['not3'] + y_pred_test['not4']
    y_pred_test['3'] = y_pred_test['not1'] + y_pred_test['not2'] + y_pred_test['not0'] + y_pred_test['not4']
    y_pred_test['4'] = y_pred_test['not1'] + y_pred_test['not2'] + y_pred_test['not3'] + y_pred_test['not0']

    y_pred_train['0'] = y_pred_train['not1'] + y_pred_train['not2'] + y_pred_train['not3'] + y_pred_train['not4']
    y_pred_train['1'] = y_pred_train['not0'] + y_pred_train['not2'] + y_pred_train['not3'] + y_pred_train['not4']
    y_pred_train['2'] = y_pred_train['not1'] + y_pred_train['not0'] + y_pred_train['not3'] + y_pred_train['not4']
    y_pred_train['3'] = y_pred_train['not1'] + y_pred_train['not2'] + y_pred_train['not0'] + y_pred_train['not4']
    y_pred_train['4'] = y_pred_train['not1'] + y_pred_train['not2'] + y_pred_train['not3'] + y_pred_train['not0']

    y_pred_holdout['0'] = y_pred_holdout['not1'] + y_pred_holdout['not2'] + y_pred_holdout['not3'] + y_pred_holdout[
        'not4']
    y_pred_holdout['1'] = y_pred_holdout['not0'] + y_pred_holdout['not2'] + y_pred_holdout['not3'] + y_pred_holdout[
        'not4']
    y_pred_holdout['2'] = y_pred_holdout['not1'] + y_pred_holdout['not0'] + y_pred_holdout['not3'] + y_pred_holdout[
        'not4']
    y_pred_holdout['3'] = y_pred_holdout['not1'] + y_pred_holdout['not2'] + y_pred_holdout['not0'] + y_pred_holdout[
        'not4']
    y_pred_holdout['4'] = y_pred_holdout['not1'] + y_pred_holdout['not2'] + y_pred_holdout['not3'] + y_pred_holdout[
        'not0']

    y_pred_test.drop(columns=['not0', 'not1', 'not2', 'not3', 'not4'], inplace=True)
    y_pred_train.drop(columns=['not0', 'not1', 'not2', 'not3', 'not4'], inplace=True)
    y_pred_holdout.drop(columns=['not0', 'not1', 'not2', 'not3', 'not4'], inplace=True)

    y_pred_test.columns = [0, 1, 2, 3, 4]
    y_pred_train.columns = [0, 1, 2, 3, 4]
    y_pred_holdout.columns = [0, 1, 2, 3, 4]

    y_pred_test = y_pred_test.idxmax(axis=1)
    y_pred_train = y_pred_train.idxmax(axis=1)
    y_pred_holdout = y_pred_holdout.idxmax(axis=1)
    predictions = y_pred_holdout

    # y = data['AdoptionSpeed']
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    #
    # modelObject.fit(x_train, y_train)

    # print(modelObject.feature_importances_)

    # pyplot.bar(range(len(modelObject.feature_importances_)), modelObject.feature_importances_)
    # pyplot.show()

    # plot feature importance - for xgboost
    # plot_importance(modelObject)
    # pyplot.show()

    # y_pred_test = modelObject.predict(x_test)
    # y_pred_train = modelObject.predict(x_train)

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
    print("unique adoption speeds in y_test = ", y_test.unique())
    print("unique adoption speeds in y_pred = ", np.unique(y_pred_test))

    train_acc = accuracy_score(y_train, y_pred_train)
    print("unique adoption speeds in y_train = ", y_train.unique())
    print("unique adoption speeds in y_pred_train = ", np.unique(y_pred_train))

    kappa = cohen_kappa_score(y_test, y_pred_test, weights='quadratic')

    print("Test set accuracy by " + modelName + " : {:.2f}".format(test_acc))
    print("Train set accuracy by " + modelName + " : {:.2f}".format(train_acc))
    print("Kappa by " + modelName + " : {:.2f}".format(kappa))

    print(confusion_matrix(y_test, y_pred_test))

    # ## Feature Selection
    # print("Trying feature selection using thresholds")
    # thresholds = sort(modelObject.feature_importances_)
    #
    # for thresh in thresholds:
    #     # select features using threshold
    #     selection = SelectFromModel(modelObject, threshold=thresh, prefit=True)
    #     select_X_train = selection.transform(x_train)
    #     # train model
    #     selection_model = XGBClassifier()
    #     selection_model.fit(select_X_train, y_train)
    #     # eval model
    #     select_X_test = selection.transform(x_test)
    #     y_pred = selection_model.predict(select_X_test)
    #     predictions = [round(value) for value in y_pred]
    #     accuracy = accuracy_score(y_test, predictions)
    #     print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))
    # ## Feature Selection

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
    petid = data_test['PetID']
    final_pred = pd.Series(predictions, name='AdoptionSpeed')
    submission = pd.concat([petid, final_pred], axis=1)
    submission.to_csv('/Users/Rima/projects/petFinder/data/v8/submission-stacking.csv', index=False)
