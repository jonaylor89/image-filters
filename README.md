
# CMSC 630 Image Analysis Project

## Execution

### With docker

```sh
docker run -it -v $HOME/Repos/CMSC630_Project_1/datasets:/app/datasets jonaylor/cmsc_project_1
```

### Without docker

```sh
pip3 install --user pipenv
pipenv install
pipenv run python main.py
```
