import pandas as pd


def merge(df, text_df, image_df):
    completeData = pd.merge(df, text_df, left_on='PetID', right_on='petid', how='left')
    completeData = pd.merge(completeData, image_df, left_on='PetID', right_on='petid', how='left')
    completeData.drop(columns=['petid_x', 'petid_y'], inplace=True)
    completeData.fillna(0, inplace=True)
    return completeData


train_df = pd.read_pickle("~/projects/petFinder/data/train.pkl")
test_df = pd.read_pickle("~/projects/petFinder/data/test.pkl")

train_text_df = pd.read_pickle("~/projects/petFinder/data/pet_text_data.pkl")
test_text_df = pd.read_pickle("~/projects/petFinder/data/pet_text_data_test.pkl")

train_image_df = pd.read_pickle("~/projects/petFinder/data/firstPhoto/pet_first_image_data_final.pkl")
test_image_df = pd.read_pickle("~/projects/petFinder/data/firstPhoto/pet_first_image_data_test_final.pkl")

# train_image_df = pd.read_pickle("~/projects/petFinder/data/pet_image_data_final.pkl")
# test_image_df = pd.read_pickle("~/projects/petFinder/data/pet_image_data_test_final.pkl")

complete_train_df = merge(train_df, train_text_df, train_image_df)
complete_test_df = merge(test_df, test_text_df, test_image_df)

# complete_train_df.to_pickle("../../../data/train_complete.pkl")
# complete_test_df.to_pickle("../../../data/test_complete.pkl")

complete_train_df.to_pickle("../../../data/firstPhoto/train_complete.pkl")
complete_test_df.to_pickle("../../../data/firstPhoto/test_complete.pkl")
