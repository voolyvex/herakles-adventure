# Dockerfile, Image, Container
FROM python:3.11

ADD main.py .

RUN pip install

CMD ["python", "./main.py"]
