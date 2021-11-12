import os

import scipy.spatial.distance

os.environ['DP_ROOT_PATH'] = '/mnt/d/deeppavlov'

from deeppavlov import build_model, configs
from deeppavlov.core.common.file import read_json


class ODQA:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            config = read_json(configs.odqa.ru_odqa_infer_wiki)
            self._model = build_model(config, download=True)
        return self._model

    def predict(self, question):
        return self.model([question])[0]

    def answer(self, question, answers):
        odqa_answer = self.predict(question)
        print('odqa answer', odqa_answer)

        for n, answer in enumerate(answers, 1):
            if answer in odqa_answer:
                return n + 1

        return 1


class Embeddings:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            bert_config = read_json(configs.embedder.bert_embedder)
            self._model = build_model(bert_config)

        return self._model

    def predict(self, texts):
        _, _, _, _, embeds, _, _ = self.model(texts)
        return embeds

    def answer(self, question, answers):
        q_embed, *ans_embed = self.predict([question] + answers)

        best_answer = min([(n, scipy.spatial.distance.cosine(q_embed, ans))
                           for n, ans in enumerate(ans_embed, 1)
                           ], key=lambda x: x[1])

        print('embeds answer', best_answer[0])

        return best_answer[0]


class TfIdf:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            bert_config = read_json(configs.doc_retrieval.ru_ranker_tfidf_wiki)
            self._model = build_model(bert_config, download=True)

        return self._model

    def answer(self, question, answers):
        q, *ans = self.model([question] + answers)

        for n, a in enumerate(ans, 1):
            if q == a:
                print('Guessed!')
                return n

        return 1


model_odqa = ODQA()
model_embed = Embeddings()
model_tfidf = TfIdf()
