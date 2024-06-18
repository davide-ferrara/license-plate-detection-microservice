FROM python:3.8-slim

LABEL maintainer="Davide Ferrara <frrdvd98m07h224l@studenti.umime.it>, Lorenzo Carnevale <lcarnevale@unime.it>" \
      name="Plate Recognition Microservice" \
      description="Docker container for running a plate detection microservice" \
    #   url="https://hub.docker.com/r/petronetto/pytorch-alpine" \
    #   vcs-url="https://github.com/petronetto/pytorch-alpine" \
      vendor="University of Messina" \
      version="1.0"

RUN echo "|--> Updating"; \
    apt update -y; \
    apt upgrade -y; \
    pip3 install --upgrade pip;

COPY requirements.txt /opt/app/requirements.txt

WORKDIR /opt/app

RUN echo "|--> Install dependencies"; \
    apt install -y libgl1 libglib2.0-0; \
    pip install -r requirements.txt;

COPY app /opt/app

CMD ["python3", "main.py", "-c", "config.yaml", "-v"]