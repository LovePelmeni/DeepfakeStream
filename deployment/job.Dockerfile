FROM --platform=arm64 python:3.9-bullseye 
LABEL author=kirklimushin@gmail.com
WORKDIR /project/dir/

# copying up source code for running analytics job

COPY ./weights ./weights
COPY ./src/preprocessing ./src/preprocesing
COPY ./src/batch_processing ./src/batch_processing

# installing package and libraries
RUN pip install --upgrade pip
RUN pip install -r requirements/job_requirements.txt

# running prediction job 
ENTRYPOINT ["python", "-m", "./src/batch_processing/job.py"]