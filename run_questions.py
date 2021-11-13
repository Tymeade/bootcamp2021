import pandas as pd

from preprocess import find_negative, normalize_pymorphy, preprocessing
from search import ElasticModelQuotes, ElasticModelWiki, ElasticModelSource

data_file = 'boot_camp_train.csv'


def get_score(model):
    questions = preprocessing(pd.read_csv(data_file).dropna())
    total = 0
    correct = 0

    questions['Вопрос'] = questions['Вопрос'].str.lower()
    questions['1'] = questions['1'].str.lower()
    questions['2'] = questions['2'].str.lower()
    questions['3'] = questions['3'].str.lower()
    questions['4'] = questions['4'].str.lower()

    for _, row in questions.iterrows():
        q = row['Вопрос']
        # print('---------------------------------')
        # print(q)
        answers = list(row[['1', '2', '3', '4']])
        # print(answers)

        our_answer = model.answer(q, answers, False)

        print('our_answer', our_answer,
              'correct answer', row['Правильный ответ'])

        total += 1
        if our_answer == row['Правильный ответ']:
            correct += 1
        print('Score', correct / total * 100, '%, out of ', total)

    print('Score', correct / total * 100, '%')


def get_one_question():
    questions = pd.read_csv(data_file).dropna().sample(frac=1)

    for _, row in questions.iterrows():
        q = row['Вопрос']
        answers = list(row[['1', '2', '3', '4']])
        correct = row['Правильный ответ']
        yield q, answers, correct


def get_one_question_neg():
    questions = pd.read_csv(data_file).dropna().sample(frac=1)

    for _, row in questions.iterrows():
        q = row['Вопрос']
        q_words = normalize_pymorphy(q)
        neg = find_negative(q_words)
        answers = list(row[['1', '2', '3', '4']])
        correct = row['Правильный ответ']
        yield q, answers, correct, neg


def compare(model_wiki, model_quotes):
    questions = preprocessing(pd.read_csv(data_file).dropna())
    total = 0
    correct_wiki = 0
    correct_quotes = 0
    correct_wiki_not_quotes = 0
    correct_quotes_not_wiki = 0
    correct_both = 0
    all_incorrect = 0
    one_of = 0

    questions['Вопрос'] = questions['Вопрос'].str.lower()
    questions['1'] = questions['1'].str.lower()
    questions['2'] = questions['2'].str.lower()
    questions['3'] = questions['3'].str.lower()
    questions['4'] = questions['4'].str.lower()

    for _, row in questions.iterrows():
        q = row['norm']
        # print('---------------------------------')
        # print(q)
        answers = list(row[['1', '2', '3', '4']])
        # print(answers)

        answer_wiki = model_wiki.answer(q, answers, False)
        answer_quotes = model_quotes.answer(q, answers, False)
        correct_answer = row['Правильный ответ']
        wiki_correct = answer_wiki == correct_answer
        quotes_correct = answer_quotes == correct_answer

        # if row['pogovorki']:
        #     wiki_correct = quotes_correct

        total += 1

        if wiki_correct:
            correct_wiki += 1
        if quotes_correct:
            correct_quotes += 1
        if wiki_correct and quotes_correct:
            correct_both += 1
        if wiki_correct and not quotes_correct:
            correct_wiki_not_quotes += 1
        if not wiki_correct and quotes_correct:
            correct_quotes_not_wiki += 1
        if not wiki_correct and not quotes_correct:
            all_incorrect += 1
        if wiki_correct or quotes_correct:
            one_of += 1

        if total % 10 == 0:
            print('----------')
            print(total)
            print('Score wiki', correct_wiki / total * 100, '%')
            print('correct_quotes', correct_quotes / total * 100, '%')
            print('Score correct_both', correct_both / total * 100, '%')
            print('Score correct_wiki_not_quotes',
                  correct_wiki_not_quotes / total * 100, '%')
            print('Score correct_quotes_not_wiki',
                  correct_quotes_not_wiki / total * 100, '%')
            print('Score all_incorrect', all_incorrect / total * 100, '%')
            print('Score one_of', one_of / total * 100, '%')

    print('----------')
    print(total)
    print('Score wiki', correct_wiki / total * 100, '%')
    print('correct_quotes', correct_quotes / total * 100, '%')
    print('Score correct_both', correct_both / total * 100, '%')
    print('Score correct_wiki_not_quotes',
          correct_wiki_not_quotes / total * 100, '%')
    print('Score correct_quotes_not_wiki',
          correct_quotes_not_wiki / total * 100, '%')
    print('Score all_incorrect', all_incorrect / total * 100, '%')
    print('Score one_of', one_of / total * 100, '%')


if __name__ == '__main__':
    # compare(ElasticModelWiki(), ElasticModelQuotes())
    # q = get_one_question()
    get_score(ElasticModelSource())
    pass
