FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04 as nvcc_build
LABEL author=kirklimushin@gmail.com 

# setting up NVCC, CUDA environment variables 

FROM python:3.9-slim-bullseye 
WORKDIR /workspace 

# copying nvidia nvcc build files
COPY --chown=nvcc_build ./ ./workspace

# project http environment variables

ENV APPLICATION_HOST=localhost;
ENV APPLICATION_PORT=8000;
ENV VERSION=1.0.0;

# cors settings
ENV ALLOWED_METHODS=*;
ENV ALLOWED_HEADERS=*;
ENV ALLOWED_ORIGINS=*;

# copying additional configurations for linters, formatters and dependency managers

COPY ./src ./src
COPY ./tests ./tests
COPY ./experiments ./experiments
COPY ./monitoring ./monitoring
COPY ./weights ./weights
COPY ./pyproject.toml ./pyproject.toml
COPY ./poetry.lock ./poetry.lock
COPY ./flake8 ./flake8
COPY ./requirements ./requirements

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