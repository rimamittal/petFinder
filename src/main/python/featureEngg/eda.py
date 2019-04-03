import pandas as pd

test = pd.read_pickle('/Users/Rima/projects/petFinder/data/v2/test_cleaned_Features_v2.pk')
train = pd.read_pickle('/Users/Rima/projects/petFinder/data/v2/train_cleaned_Features_v2.pk')

popularBreeds = train.loc[train['AdoptionSpeed'] == 0, ['Breed1', 'Breed2']]
popularBreed1 = popularBreeds['Breed1'].unique()
popularBreed2 = popularBreeds['Breed2'].unique()


def eda(data):
    # Name
    data['hasName'] = data['Name'] != 'No Name'
    data['hasName'] = data['hasName'].astype('int')

    # Age
    data['isYoung'] = (data['Age'] < 2).astype('int')
    data['isOld'] = (data['Age'] > 12).astype('int')

    # Breed

    data['ispopularBreed1'] = data.apply(lambda x: x['Breed1'] in popularBreed1, axis=1).astype('int')
    data['ispopularBreed2'] = data.apply(lambda x: x['Breed2'] in popularBreed2, axis=1).astype('int')

    # mapping color with colorlabels
    colorlabels = pd.read_csv('/Users/Rima/projects/petFinder/data/v4/color_labels.csv')
    data['color1Name'] = data['Color1'].map(colorlabels.set_index('ColorID')['ColorName'])
    data['color2Name'] = data['Color2'].map(colorlabels.set_index('ColorID')['ColorName'])
    data['color3Name'] = data['Color3'].map(colorlabels.set_index('ColorID')['ColorName'])
    data[['color1Name', 'color2Name', 'color3Name']] = data[['color1Name', 'color2Name', 'color3Name']].fillna(
        'No Color')

    data.drop(columns=['Color1', 'Color2', 'Color3'], inplace=True)

    # Merge Image color data and see if color in image matches color in csv
    # TO DO

    # De-wormed
    data['Dewormed'] = data['Dewormed'].astype('category')

    # Fee
    data['isNotFree'] = (data['Fee'] != 0).astype('int')

    # FurLength
    data['FurLength'] = data['FurLength'].astype('category')

    # Gender
    data['Gender'] = data['Gender'].astype('category')

    # Health
    data['Health'] = data['Health'].astype('category')

    # Quantity
    data['isSingle'] = (data['Quantity'] == 1).astype('int')

    # State
    data['State'] = data['State'].astype('category')

    # Sterilized
    data['Sterilized'] = data['Sterilized'].astype('category')

    # Type
    data['Type'] = data['Type'].astype('category')

    # Vaccinated
    data['Vaccinated'] = data['Vaccinated'].astype('category')

    # Videos
    data['hasVideo'] = (data['VideoAmt'] > 1).astype('int')

    data['hasLotOfVideos'] = (data['VideoAmt'] > 5).astype('int')

    return data


train = eda(train)
test = eda(test)

train.to_pickle('/Users/Rima/projects/petFinder/data/v4/csv_features_train.pkl')
test.to_pickle('/Users/Rima/projects/petFinder/data/v4/csv_features_test.pkl')