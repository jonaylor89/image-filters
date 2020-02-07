FROM kennethreitz/pipenv

WORKDIR /app

COPY . /app/

RUN pipenv install --deploy --system

CMD ["python3", "/app/main.py", "Cancerous_cell_smears"]
