import pandas as pd


def clean_image(df):
    df['photo_num'] = df['photo_num'].astype('int')
    df['detectionConfidence'] = df['detectionConfidence'].astype('float')
    df['cropHintsConfidence'] = df['cropHintsConfidence'].astype('float')
    df['joyLikelihood'] = df['joyLikelihood'].astype('int')
    df['colors_in_image'] = df['colors_in_image'].astype('int')
    df['sorrowLikelihood'] = df['sorrowLikelihood'].astype('int')
    df['angerLikelihood'] = df['angerLikelihood'].astype('int')
    df['surpriseLikelihood'] = df['surpriseLikelihood'].astype('int')
    df['underExposedLikelihood'] = df['underExposedLikelihood'].astype('int')
    df['blurredLikelihood'] = df['blurredLikelihood'].astype('int')
    df['headwearLikelihood'] = df['headwearLikelihood'].astype('int')
    df['labelcount'] = df['labelcount'].astype('int')
    df['street dog'] = df['street dog'].astype('float')
    df['dog like mammal'] = df['dog like mammal'].astype('float')
    df['aegean cat'] = df['aegean cat'].astype('float')
    df['carnivoran'] = df['carnivoran'].astype('float')
    df['small to medium sized cats'] = df['small to medium sized cats'].astype('float')
    df['cat like mammal'] = df['cat like mammal'].astype('float')
    df['dog breed'] = df['dog breed'].astype('float')
    df['snout'] = df['snout'].astype('float')
    df['whiskers'] = df['whiskers'].astype('float')
    df['domestic short haired cat'] = df['domestic short haired cat'].astype('float')
    df['puppy'] = df['puppy'].astype('float')
    df['dog breed group'] = df['dog breed group'].astype('float')
    df['fauna'] = df['fauna'].astype('float')
    df['kitten'] = df['kitten'].astype('float')
    df['european shorthair'] = df['european shorthair'].astype('float')
    df['dog'] = df['dog'].astype('float')
    df['sporting group'] = df['sporting group'].astype('float')
    df['cat'] = df['cat'].astype('float')
    return df


train_images = pd.read_pickle("/Users/Rima/projects/petFinder/data/v4/pet_image_data_train.pkl")
test_images = pd.read_pickle("/Users/Rima/projects/petFinder/data/v4/pet_image_data_test.pkl")

train_images = clean_image(train_images)
test_images = clean_image(test_images)


def aggregate_photos(df):
    columns = ['petid', 'photo_num', 'labels', 'color1', 'color2', 'color3']
    categoricals = df[columns]
    categoricals = categoricals[categoricals['photo_num'] == 1]
    categoricals.drop(columns=['photo_num'], inplace=True)
    df = df.drop(columns=['labels', 'color1', 'color2', 'color3'])
    first_photo = df[df['photo_num'] == 1]
    rest_photo = df[df['photo_num'] != 1]
    first_photo = first_photo.groupby('petid').sum() * 0.8
    rest_photo = rest_photo.groupby('petid').sum() * 0.2
    all_photos = pd.concat([first_photo, rest_photo])
    all_photos = all_photos.groupby('petid').sum()
    all_photos.reset_index(inplace=True)
    all_photos = pd.merge(all_photos, categoricals, on='petid')

    all_photos.drop(columns=['photo_num'], inplace=True)

    return all_photos


train_images = aggregate_photos(train_images)
test_images = aggregate_photos(test_images)

train_images.to_pickle("/Users/Rima/projects/petFinder/data/v4/image_train.pkl")
test_images.to_pickle("/Users/Rima/projects/petFinder/data/v4/image_test.pkl")

