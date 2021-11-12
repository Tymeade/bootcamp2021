import pandas as pd

from models import model_embed, model_tfidf
from search import ElasticModel

data_file = 'boot_camp_train.csv'


def get_score(model):
    questions = pd.read_csv(data_file).dropna()
    total = 0
    correct = 0

    questions['Вопрос'] = questions['Вопрос'].str.lower()
    questions['1'] = questions['1'].str.lower()
    questions['2'] = questions['2'].str.lower()
    questions['3'] = questions['3'].str.lower()
    questions['4'] = questions['4'].str.lower()

    for _, row in questions.iterrows():
        q = row['Вопрос']
        print('---------------------------------')
        print(q)
        answers = list(row[['1', '2', '3', '4']])
        print(answers)

        our_answer = model.answer(q, answers)

        print('our_answer', our_answer,
              'correct answer', row['Правильный ответ'])

        total += 1
        if our_answer == row['Правильный ответ']:
            correct += 1

    print('Score', correct / total * 100, '%')


if __name__ == '__main__':
    get_score(ElasticModel())