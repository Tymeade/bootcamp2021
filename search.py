from elasticsearch import Elasticsearch


class ElasticModel():
    es = Elasticsearch()

    def _get_score(self, text):
        res = self.es.search(index="ruwikiquote",
                             body={"query": {"match":
                                 {
                                     "text": text}}})

        return sum(x['_score'] for x in res['hits']['hits'])

    def answer(self, question, answers):
        scores = [(n, self._get_score(question + ' ' + ans))
                  for n, ans in enumerate(answers, 1)]

        best = max(scores, key=lambda x: x[1])
        print('best', best)

        return best[0]
