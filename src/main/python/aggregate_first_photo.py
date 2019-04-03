import pandas as pd

from src.main.python.aggregate import clean_image

image_train_df = pd.read_pickle("~/projects/petFinder/data/pet_image_data.pkl")
image_test_df = pd.read_pickle("~/projects/petFinder/data/pet_image_data_test.pkl")

image_train_df = clean_image(image_train_df)
image_test_df = clean_image(image_test_df)

image_train_df = image_train_df[image_train_df['photo_num'] == 1]
image_train_df.drop(columns=['photo_num'], inplace=True)

image_test_df = image_test_df[image_test_df['photo_num'] == 1]
image_test_df.drop(columns=['photo_num'], inplace=True)

image_train_df.to_pickle("../../../data/firstPhoto/pet_first_image_data_final.pkl")
image_test_df.to_pickle("../../../data/firstPhoto/pet_first_image_data_test_final.pkl")
