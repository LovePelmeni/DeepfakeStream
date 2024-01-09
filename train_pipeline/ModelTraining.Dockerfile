FROM --platform=arm64 python:3.9-latest 
LABEL maintainer=kirklimushin@gmail.com 

# copying training data and pipeline source code

COPY ./pipeline ./pipeline 
COPY ../experiments/current_experiment/train_data ./experiments/train_data
COPY ../requirements/prod_requirements.txt ./prod_requirements.txt

# installing dependencies and upgrading interpreter

RUN pip install --upgrade pip
RUN pip install -r prod_requirements.txt 

# running training pipeline
ENTRYPOINT ["python3", "pipeline.py"]
