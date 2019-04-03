import pandas as pd
from src.main.python.imageData import *
from src.main.python.text.textData import *

path_to_json = '/Users/Rima/Documents/Q2/petfinderkaggle/petfinder-adoption-prediction/test/test_sentiment/'
files = read_json(path_to_json)

df_columns = ['petid', 'magnitude', 'score']
pet_text_data_test = pd.DataFrame(columns=df_columns)
pet_text_data_test = construct_df_from_text_json(files, pet_text_data_test, path_to_json)
pet_text_data_test.to_pickle("../../../../data/pet_text_data_test.pkl")



