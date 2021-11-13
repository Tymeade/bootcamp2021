from flask import Flask, request

from game2 import GameEnv2

app = Flask(__name__)
model = GameEnv2({})


@app.route("/predict", methods=['POST'])
def predict():
    data: dict = request.form.get('data')

    action = model.get_action(data['question'],
                              [
                                  data['answer_1'],
                                  data['answer_2'],
                                  data['answer_3'],
                                  data['answer_4'],
                              ],
                              data['question money'],
                              )

    if action == 'take money':
        resp = {
            'end game': "take money",
        }
    elif action in [0, 1, 2, 3]:
        resp = {
            'answer': action + 1,
        }
    else:
        resp = {
            'help': action,
        }

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
