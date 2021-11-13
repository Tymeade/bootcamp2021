FROM python:3.6-buster

# Install Elasticsearch.
RUN wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.15.2-linux-x86_64.tar.gz
RUN wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.15.2-linux-x86_64.tar.gz.sha512
RUN shasum -a 512 -c elasticsearch-7.15.2-linux-x86_64.tar.gz.sha512
RUN tar -xzf elasticsearch-7.15.2-linux-x86_64.tar.gz


