FROM python:3.9-latest as app

ENV python_user=python_user
ENV PYTHONUNBUFFERED=1

RUN useradd -m ${python_user}
RUN useradd -aG ${python_user} sudo 

WORKDIR /project/dir/${python_user}

COPY ./models ./models
COPY ./rest ./rest
COPY ./deployment/entrypoint.sh ./entrypoint.sh
COPY ./pyproject.toml ./pyproject.toml
COPY ./requirements ./requirements
COPY ./poetry.lock ./poetry.lock

RUN pip install --upgrade pip && pip install poetry

RUN poetry export --format=requirements.txt \
--output=requirements/prod_requirements.txt --without-hashes

RUN pip install -r requirements/prod_requirements.txt 
RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["sh", "entrypoint.sh"]
