# PetFinder
This is a project to predict the adoption speed of pets with a specific online profile.
Kaggle Competition: https://www.kaggle.com/c/petfinder-adoption-prediction
Steps to run :

#Clone the project from github
1. Clone the project

#Create a virtual env
2. virtualenv env

#Enter the virtual Environment
3. source env/bin/activate

#Run a clean build - this will remove any existing target folders.
4. pip install pybuilder
5. pyb clean

##Run a build
6. pyb

##Install the dependencies
7. pyb install

## Steps:
1. Read Images - readImageData.py and readImageDataTest.py
2. Aggregate and clean image data - aggregate.py
3. clean test and train - cleanup.py
4. New Features - eda.py
5. Merge text, images and data - merge.py
6. Run Models - all py files in src/main/python/models
