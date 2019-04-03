# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import GradientBoostingClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))


def merge(df, text_df, image_df):
    completeData = pd.merge(df, text_df, left_on='PetID', right_on='petid', how='left')
    completeData = pd.merge(completeData, image_df, left_on='PetID', right_on='petid', how='left')
    completeData.drop(columns=['petid_x', 'petid_y'], inplace=True)
    completeData.fillna(0, inplace=True)
    return completeData


def cleanup(df):
    df['Name'].fillna('No Name', inplace=True)
    df['NumColors'] = df.loc[:, 'Color1':'Color3'].apply(
        lambda row: bool(row.Color1) + bool(row.Color2) + bool(row.Color3), axis=1)
    df['IsMixedBreed'] = df.loc[:, 'Breed1':'Breed2'].apply(
        lambda row: bool(row.Breed1) * bool(row.Breed2), axis=1)
    name_frequency_df = pd.DataFrame(df['Name'].value_counts())
    df['NameFrequency'] = name_frequency_df.loc[df['Name'], 'Name'].values
    df['Description'].fillna('', inplace=True)
    df['WordCount'] = df['Description'].str.split().apply(lambda x: len(x))
    df = df.drop('Description', axis=1)
    return df


train = pd.read_csv("~/projects/petFinder/data/train/train.csv")
test = pd.read_csv("~/projects/petFinder/data/test/test.csv")

train_df = cleanup(train)
test_df = cleanup(test)

train_text_df = pd.read_csv("~/projects/petFinder/data/csv/pet_text_data.csv")
test_text_df = pd.read_csv("~/projects/petFinder/data/csv/pet_text_data_test.csv")

train_image_df = pd.read_csv("~/projects/petFinder/data/csv/pet_image_data_final.csv")
test_image_df = pd.read_csv("~/projects/petFinder/data/csv/pet_image_data_test_final.csv")

data = merge(train_df, train_text_df, train_image_df)
data_test = merge(test_df, test_text_df, test_image_df)

X_test = data_test.drop(['Name',
                         'PetID',
                         'RescuerID'], axis=1)

X = data.drop(['Name',
               'PetID',
               'AdoptionSpeed',
               'RescuerID'], axis=1)
y = data['AdoptionSpeed']
# Any results you write to the current directory are saved as output.
gbm = GradientBoostingClassifier(random_state=10, n_estimators=200)
gbm.fit(X, y)

y_pred_gbm = gbm.predict(X_test).round()

petid = data_test['PetID']

final_pred_xg = pd.Series(y_pred_gbm, name='AdoptionSpeed')
submission_xg = pd.concat([petid, final_pred_xg], axis=1)

submission_xg.to_csv('submission.csv', index=False)
