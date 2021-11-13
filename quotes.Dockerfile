FROM eprosvirina/data

RUN pip install pandas
RUN pip install 'ray[rllib]'

COPY . /app

RUN pip3 install -r /app/requirements.txt

ENV PYTHONPATH='/app'
RUN chmod 777 /app/start_elastic.sh

CMD ["bash", '-c', '/app/start_elastic.sh']