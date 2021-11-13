wget https://www.transfer.sh/VFqcVw/elastic.gzip --no-check-certificate
tar -zcvfx elastic.gzip
cd /elastic && ./bin/elasticsearch -d -p 100000

cd / && python /app/service.py