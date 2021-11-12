from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import json
import pymystem3
import pymorphy2
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    PER,
    NamesExtractor,
    Doc
)


nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

morph = pymorphy2.MorphAnalyzer()
stem = pymystem3.Mystem()
morph_vocab = MorphVocab()
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
names_extractor = NamesExtractor(morph_vocab)

def normalize_pymorphy(text):
    tokens = re.findall('[A-Za-zА-Яа-яЁё]+\-[A-Za-zА-Яа-яЁё]+|[A-Za-zА-Яа-яЁё]+', text)
    words = []
    for t in tokens:
        pv = morph.parse(t)
        words.append(pv[0].normal_form)# + '_' + str(pv[0].tag.POS)
#                     )
    return words

negative_words = ['нет', 'не', 'без']

def find_negative(text):
    filtered_words = [word for word in text if word in negative_words]
    return int(bool(filtered_words))

stop_words = nltk.corpus.stopwords.words('russian')

def remove_stop_words(text):
    filtered_words = [word for word in text if word not in stop_words]
    return filtered_words

question_words = ['как', 'какой', 'где', 'что', 'кто', 'куда', 'откуда', 'когда', 'чем', 'сколько', 'чей']
def find_question(text):
    filtered_words = [word for word in text if word in question_words]
    return filtered_words


def normalize_pymystem_nouns(text):
    tokens = stem.analyze(text)
    words = []
    tags = []
    for t in tokens:
        if 'analysis' in t.keys():
            if t['analysis'] != []:
                if t['analysis'][0]['gr'][0] == 'S':
                    words.append(t['analysis'][0]['lex'])
    if words != []:
        return list(set(words))
    else:
        return np.nan

def normalize_pymystem_verbs(text):
    tokens = stem.analyze(text)
    words = []
    tags = []
    for t in tokens:
        if 'analysis' in t.keys():
            if t['analysis'] != []:
                if t['analysis'][0]['gr'][0] == 'V':
                    words.append(t['analysis'][0]['lex'])
    if words != []:
        return list(set(words))
    else:
        return np.nan

def normalize_pymystem_adjs(text):
    tokens = stem.analyze(text)
    words = []
    tags = []
    for t in tokens:
        if 'analysis' in t.keys():
            if t['analysis'] != []:
                if t['analysis'][0]['gr'][0] == 'A':
                    words.append(t['analysis'][0]['lex'])
    if words != []:
        return list(set(words))
    else:
        return np.nan

posl_words = ['пословица', 'поговорка']
def find_posl(text):
    filtered_words = [word for word in text if word in posl_words]
    return int(bool(filtered_words))

def find_citation(s):
    first, last = "«", "»"
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return np.nan

stop_words_modified = ['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так',
                       'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было',
                       'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг',
                       'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж',
                       'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть',
                       'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего',
                       'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого',
                       'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас',
                       'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть',
                       'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много',
                       'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше',
                       'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между', 'который',
                       'свой',  'согласно', 'обычно', 'сколько', 'весь', 'часто', 'чей', 'всё', 'это', 'её', 'его',
                       'как', 'какой', 'где', 'что', 'кто', 'куда', 'откуда', 'когда', 'чем', 'сколько', 'чей', 'всё-таки', 'все-таки']

def remove_stop_words_mod(text):
    filtered_words = [word for word in text if word not in stop_words_modified]
    return filtered_words

def find_pog_text(text):
    try:
        first_1 = text[0].index("поговорка")
        res_1 = text[1].split(' ')[first_1+1:]
        return ' '.join(res_1)
    except ValueError:
        return np.nan
def find_posl_text(text):
    try:
        first_1 = text[0].index("пословица")
        res_1 = text[1].split(' ')[first_1+1:]
        return ' '.join(res_1)
    except ValueError:
        return np.nan

def find_names(text):
    types = []
    names = []
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    for span in doc.spans:
        span.normalize(morph_vocab)
        types.append(span.type)
        names.append(span.normal)
        if types != [] or names != []:
            return types, names
        else:
            return np.nan



def preprocessing(questions):
    questions['norm'] = questions['Вопрос'].apply(normalize_pymorphy)
    questions['negative'] = questions['norm'].apply(lambda x: find_negative(x))
    questions['norm_wo_stop_words'] = questions['norm'].apply(lambda x: remove_stop_words(x))
    questions['question'] = questions['norm'].apply(lambda x: find_question(x))
    questions['noun'] = questions['Вопрос'].apply(normalize_pymystem_nouns)
    questions['verb'] = questions['Вопрос'].apply(normalize_pymystem_verbs)
    questions['adj'] = questions['norm_wo_stop_words'].apply(lambda x: normalize_pymystem_adjs(' '.join(x)))
    questions['posl'] = questions['norm'].apply(lambda x: find_posl(x))
    questions['citation'] = questions['Вопрос'].apply(find_citation)
    questions['norm_wo_stop_words_mod'] = questions['norm'].apply(lambda x: remove_stop_words_mod(x))
    questions['pogovorka'] = questions[['norm', 'Вопрос']].apply(find_pog_text, axis=1)
    questions['poslovica'] = questions[['norm', 'Вопрос']].apply(find_posl_text, axis=1)
    questions['pogovorki'] = (questions['pogovorka'].fillna('') + ' ' + questions['poslovica'].fillna('')).str.strip(' ')
    questions = questions.drop(columns=['pogovorka', 'poslovica'])
    questions['names_types'] = questions['Вопрос'].apply(find_names)
    questions['name_type'], questions['name'] = questions['names_types'].str[0], questions['names_types'].str[1]

    return questions

