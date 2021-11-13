FROM python:3.6-buster

RUN pip install pandas
RUN pip install 'ray[rllib]'

COPY . /app

RUN pip3 install -r /app/requirements.txt

ENV PYTHONPATH='/app'

ENTRYPOINT ["python"]
CMD ["/app/service.py"]