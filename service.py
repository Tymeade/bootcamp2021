from flask import Flask
from flask import request

from models import model_odqa

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    data = request.form.get('data')

    #  data:
    #  {
    #
    #    'number of game': 5,
    #
    #    'question': "Что есть у Пескова?",
    #    'answer_1': "Усы",
    #    'answer_2': "Борода",
    #    'answer_3': "Лысина",
    #    'answer_4': "Третья нога",
    #
    #    'question money': 4000,
    #    'saved money': 1000,
    #    'available help': ["fifty fifty", "can mistake", "take money"]
    #
    #  }
    response = model_odqa.answer(data['question'],
                                 [
                                     data['answer_1'],
                                     data['answer_2'],
                                     data['answer_3'],
                                     data['answer_4'],
                                 ])
    resp = {
        'answer': response,
    }

    #  resp:
    #  {
    #
    #    'help': "fifty fifty",
    #
    #  }

    #  resp:
    #  {
    #
    #    'help': "can mistake",
    #    'answer': 1,
    #
    #  }

    #  resp:
    #  {
    #
    #    'end game': "take money",
    #
    #  }

    return resp


@app.route("/result_question", methods=['POST'])
def result_question():
    data = request.form.get('data')

    #  data:
    #  {
    #
    #    'number of game': 5,
    #
    #    'question': "Что есть у Пескова?",
    #    'answer': 1,
    #
    #    'bank': 4000,
    #    'saved money': 1000,
    #    'response type': "good"
    #
    #  }

    #  data:
    #  {
    #
    #    'number of game': 5,
    #
    #    'question': "Что есть у Пескова?",
    #    'answer': 4,
    #
    #    'bank': 1000,
    #    'saved money': 1000,
    #    'response type': "bad"
    #
    #  }

    return {'data': 'ok'}


app.run(host='127.0.0.1', port=12302)

# команда-1: port=12301
# команда-2: port=12302
# команда-3: port=12303
# команда-4: port=12304
# команда-5: port=12305
# команда-6: port=12306
# команда-7: port=12307
