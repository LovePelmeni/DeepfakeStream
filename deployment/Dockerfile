FROM python:3.9-latest as app
LABEL maintainer="kirklimushin@gmail.com"

# project http environment variables

ENV APPLICATION_HOST=localhost;
ENV APPLICATION_PORT=8000;
ENV VERSION=1.0.0;

# cors settings
ENV ALLOWED_METHODS=*;
ENV ALLOWED_HEADERS=*;
ENV ALLOWED_ORIGINS=*;

# superuser environment variables 

ENV python_user=python_user
ENV PYTHONUNBUFFERED=1

RUN useradd -m ${python_user}
RUN useradd -aG ${python_user} sudo 

WORKDIR /project/dir/${python_user}

# copying main functionality inside the container

COPY ./models ./models
COPY ./rest ./rest
COPY ./deployment/entrypoint.sh ./entrypoint.sh


# copying additional configurations for linters, formatters and dependency managers
COPY ./pyproject.toml ./pyproject.toml
COPY ./poetry.lock ./poetry.lock
COPY ./flake8 ./flake8
COPY ./requirements ./requirements

RUN mkdir logs

# installing and upgrading 
RUN pip install --upgrade pip && pip install poetry

# exporting dependency list
RUN poetry export --format=requirements.txt \
--output=requirements/prod_requirements.txt --without-hashes

# installing dependencies
RUN pip install -r requirements/prod_requirements.txt 

# running deployment pipeline
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["sh", "entrypoint.sh"]
