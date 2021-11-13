import elasticsearch
import numpy as np
from elasticsearch import Elasticsearch

max_s = 300


def normalize(data):
    return (data) / max_s


class ElasticModelWiki():
    es = Elasticsearch(timeout=60)
    index = 'ruwiki'

    def _get_score(self, text):
        res = self.es.search(index=self.index,
                             body={"query": {"match":
                                 {
                                     "text": text}}})

        return sum(x['_score'] for x in res['hits']['hits'])

    def answer(self, question, answers, negation):
        scores = [(n, self._get_score(question + ' ' + ans))
                  for n, ans in enumerate(answers, 1)]

        if negation:
            best = min(scores, key=lambda x: x[1])
        else:
            best = max(scores, key=lambda x: x[1])
        # print('best', best)

        return best[0]

    def get_scores(self, question, answers):
        try:
            scores = np.array([self._get_score(question + ' ' + ans)
                               if ans
                               else 0
                               for n, ans in enumerate(answers, 0)])

        except elasticsearch.exceptions.ConnectionError:
            return np.array([100, 100, 100, 100])

        scores = list(normalize(scores))

        return scores


class ElasticModelQuotes(ElasticModelWiki):
    index = 'ruwikiquote'


class ElasticModelSource(ElasticModelWiki):
    index = 'wikisource'

