FROM kennethreitz/pipenv

WORKDIR /app

COPY . /app/

RUN pipenv install --deploy --system

RUN mkdir /app/Cancerous_cell_smears

CMD ["python3", "/app/main.py", "/app/Cancerous_cell_smears"]
