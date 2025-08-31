FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN sed -i 's/\r$//' run.sh
RUN chmod +x run.sh

CMD ["./run.sh"]
