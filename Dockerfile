# syntax=docker/dockerfile:1.4
FROM python:3.11 AS builder

WORKDIR /app

COPY requirements.txt /app
RUN pip3 install -r requirements.txt --verbose

COPY . /app

ENTRYPOINT ["python3"]
CMD ["app.py"]


