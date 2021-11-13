import pandas as pd
from preprocess import preprocessing

questions = pd.read_csv('boot_camp_train.csv').drop(columns='Unnamed: 0').dropna()
que = preprocessing(questions.head(10))
print(que)