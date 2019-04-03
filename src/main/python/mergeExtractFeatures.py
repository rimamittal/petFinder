import pandas as pd

train_features = pd.read_pickle('/Users/Rima/projects/petFinder/data/v4/csv_features_train.pkl')
test_features = pd.read_pickle('/Users/Rima/projects/petFinder/data/v4/csv_features_test.pkl')

train_images = pd.read_pickle('/Users/Rima/projects/petFinder/data/v4/image_train.pkl')
test_images = pd.read_pickle('/Users/Rima/projects/petFinder/data/v4/image_test.pkl')


def mergeandCreateFeatures(features, images):
    completeData = pd.merge(features, images, left_on='PetID', right_on='petid', how='left')
    completeData.drop(columns=['petid'], inplace=True)

    breed_labels = pd.read_csv('/Users/Rima/projects/petFinder/data/v4/breed_labels.csv')

    completeData['breed1Name'] = completeData['Breed1'].map(breed_labels.set_index('BreedID')['BreedName'])
    completeData['breed2Name'] = completeData['Breed2'].map(breed_labels.set_index('BreedID')['BreedName'])
    completeData['breed2Name'].fillna("not_recognized", inplace=True)

    completeData['labels'].fillna(" ", inplace=True)
    completeData['breed1Name'].fillna("not_recognized", inplace=True)

    completeData['breed1Name'] = completeData['breed1Name'].str.lower()
    completeData['breed2Name'] = completeData['breed2Name'].str.lower()
    completeData['labels'] = completeData['labels'].str.lower()

    completeData['isBreed1Matching'] = completeData[['breed1Name', 'labels']].apply(
        lambda x: x['breed1Name'] in x['labels'], axis=1).astype('int')
    completeData['isBreed2Matching'] = completeData[['breed2Name', 'labels']].apply(
        lambda x: x['breed2Name'] in x['labels'], axis=1).astype('int')

    # colors from csv features
    completeData['color1Name'] = completeData['color1Name'].str.lower()
    completeData['color2Name'] = completeData['color2Name'].str.lower()
    completeData['color3Name'] = completeData['color3Name'].str.lower()

    # colors from images
    completeData['color1'] = completeData['color1'].str.lower()
    completeData['color2'] = completeData['color2'].str.lower()
    completeData['color3'] = completeData['color3'].str.lower()

    completeData['iscolor1Matching'] = (completeData['color1'] == completeData['color1Name']).astype('int')
    completeData['iscolor2Matching'] = (completeData['color2'] == completeData['color2Name']).astype('int')
    completeData['iscolor3Matching'] = (completeData['color3'] == completeData['color3Name']).astype('int')

    # Null Values
    completeData.loc[:, 'detectionConfidence': 'cat'] = completeData.loc[:, 'detectionConfidence': 'cat'].fillna(0)
    completeData.loc[:, 'color1': 'color3'] = completeData.loc[:, 'color1': 'color3'].fillna("")

    return completeData


complete_train = mergeandCreateFeatures(train_features, train_images)
complete_test = mergeandCreateFeatures(test_features, test_images)

complete_train.to_pickle('/Users/Rima/projects/petFinder/data/v4/complete_train.pkl')
complete_test.to_pickle('/Users/Rima/projects/petFinder/data/v4/complete_test.pkl')
