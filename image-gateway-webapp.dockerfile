FROM python:3.8.12-slim


WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install --upgrade pip  \
    && pip install -r /tmp/requirements.txt

COPY ["gateway_webapp.py", "proto.py", "./"]

EXPOSE 6969

ENTRYPOINT ["gunicorn", "--bind=localhost:6969", "gateway_webapp:app"]