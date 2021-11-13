FROM python:3.6-buster

COPY ruwikiquote-20211108-cirrussearch-content.json.gz ruwikiquote

RUN wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.15.2-linux-x86_64.tar.gz
RUN wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.15.2-linux-x86_64.tar.gz.sha512
RUN shasum -a 512 -c elasticsearch-7.15.2-linux-x86_64.tar.gz.sha512
RUN tar -xzf elasticsearch-7.15.2-linux-x86_64.tar.gz


ENV index wikiquote
ENV dump ruwikiquote
RUN mkdir chunks
RUN cd chunks && zcat ../$dump | split -a 10 -l 500 - $index

