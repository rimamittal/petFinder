import pandas as pd

# a = pd.read_pickle('/Users/Rima/projects/petFinder/data/v2/test_complete.pkl')
# b = pd.read_pickle('/Users/Rima/projects/petFinder/data/v2/train_complete.pkl')
#
# a.to_csv('/Users/Rima/projects/petFinder/data/v2/test_complete.csv', index=False)
# b.to_csv('/Users/Rima/projects/petFinder/data/v2/train_complete.csv', index=False)
#
#
# c = pd.read_pickle('/Users/Rima/projects/petFinder/data/pet_text_data.pkl')
# d = pd.read_pickle('/Users/Rima/projects/petFinder/data/pet_text_data_test.pkl')
#
# c.to_csv('/Users/Rima/projects/petFinder/data/csv/pet_text_data.csv')
# d.to_csv('/Users/Rima/projects/petFinder/data/csv/pet_text_data_test.csv')

# train_text_df = pd.read_csv("~/projects/petFinder/data/csv/pet_text_data.csv")
# plt.plot(train_text_df['magnitude'], train_text_df['score'])
# plt.show()


# test = pd.read_pickle('/Users/Rima/projects/petFinder/data/v2/test_cleaned_Features_v2.pk')
# train = pd.read_pickle('/Users/Rima/projects/petFinder/data/v2/train_cleaned_Features_v2.pk')
#
# train_image_df = pd.read_pickle("~/projects/petFinder/data/firstPhoto/pet_first_image_data_final.pkl")
# test_image_df = pd.read_pickle("~/projects/petFinder/data/firstPhoto/pet_first_image_data_test_final.pkl")
#
# completeDataTrain = pd.merge(train, train_image_df, left_on='PetID', right_on='petid', how='left')
# completeDataTrain.drop(columns=['petid'], inplace=True)
# completeDataTrain.fillna(0, inplace=True)
#
#
# completeDataTest = pd.merge(test, test_image_df, left_on='PetID', right_on='petid', how='left')
# completeDataTest.drop(columns=['petid'], inplace=True)
# completeDataTest.fillna(0, inplace=True)
#
# completeDataTrain.to_pickle("../../../data/v2/train_complete.pkl")
# completeDataTest.to_pickle("../../../data/v2/test_complete.pkl")

breed_labels = pd.read_csv('/Users/Rima/projects/petFinder/data/image/breed_labels.csv')
image_labels = pd.read_pickle('/Users/Rima/projects/petFinder/data/image/pet_image_data.pkl')
train = pd.read_csv('/Users/Rima/projects/petFinder/data/v2/train_complete.csv')

train['breed1Name'] = train['Breed1'].map(breed_labels.set_index('BreedID')['BreedName'])
train['breed2Name'] = train['Breed2'].map(breed_labels.set_index('BreedID')['BreedName'])
train['breed2Name'].fillna("not_recognized", inplace=True)

first_image = image_labels[image_labels['photo_num'] == '1']
train['labelsfromImage'] = train['PetID'].map(first_image.set_index('petid')['labels'])
train['labelsfromImage'].fillna(" ", inplace=True)
train['breed1Name'].fillna("not_recognized", inplace=True)

train['breed1Name'] = train['breed1Name'].str.lower()
train['breed2Name'] = train['breed2Name'].str.lower()
train['labelsfromImage'] = train['labelsfromImage'].str.lower()

train['isBreed1Matching'] = train[['breed1Name', 'labelsfromImage']].apply(lambda x: x['breed1Name'] in x['labelsfromImage'], axis=1).astype('int')
train['isBreed2Matching'] = train[['breed1Name', 'labelsfromImage']].apply(lambda x: x['labelsfromImage'].find(x['breed1Name']), axis=1).astype('int')




