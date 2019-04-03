import pandas as pd

xg = pd.read_csv('~/projects/petFinder/data/xg/submission.csv')
gbm = pd.read_csv('~/projects/petFinder/data/gbm/submission.csv')

combined = pd.merge(xg,gbm, on='PetID')
combined['AdoptionSpeed'] = ((combined['AdoptionSpeed_x'] + combined['AdoptionSpeed_y'])/2).round()

combined.drop(columns = ['AdoptionSpeed_y', 'AdoptionSpeed_x'], inplace= True)

combined['AdoptionSpeed'] = combined['AdoptionSpeed'].astype('int')
combined.to_csv('~/projects/petFinder/data/submission/submission.csv', index=False)