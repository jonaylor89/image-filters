FROM kennethreitz/pipenv

WORKDIR /app

COPY . /app/

RUN pipenv install --deploy --system

RUN mkdir -p /app/datasets/Cancerous_cell_smears

CMD ["python3", "/app/main.py"]
