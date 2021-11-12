FROM python:3.6-buster

RUN pip install tensorflow-gpu==1.14.0
RUN pip install deeppavlov
RUN pip install pandas

RUN mkdir /mnt/d/deeppavlov && chmod +rwx /mnt/d/deeppavlov

COPY . /app

RUN pip3 install -r /app/requirements.txt

ENV PYTHONPATH='/app'

CMD ['python', '/app/service.py']