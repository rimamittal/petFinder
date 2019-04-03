import pandas as pd

train_df = pd.read_csv("~/projects/petFinder/data/train/train.csv")
test_df = pd.read_csv("~/projects/petFinder/data/test/test.csv")


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


train_copy_df = cleanup(train_df)
test_copy_df = cleanup(test_df)

train_copy_df.to_pickle("../../../data/train.pkl")
test_copy_df.to_pickle("../../../data/test.pkl")
