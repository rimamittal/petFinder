import pandas as pd
from src.main.python.imageData import *
from src.main.python.text.textData import *

path_to_json = '/Users/Rima/Documents/Q2/petfinderkaggle/petfinder-adoption-prediction/train/train_sentiment/'
files = read_json(path_to_json)

df_columns = ['petid', 'magnitude', 'score']
pet_text_data = pd.DataFrame(columns=df_columns)
pet_text_data = construct_df_from_text_json(files, pet_text_data, path_to_json)
pet_text_data.to_pickle("../../../../data/pet_text_data.pkl")



