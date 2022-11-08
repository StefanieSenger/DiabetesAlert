FROM python:3.9.13-slim

#RUN pip install pyenv # doesn't work, not sure if I need to make it run, because for installing the requirements.txt I think it's not necessary (?)

WORKDIR /app
COPY ["requirements.txt", "./"]

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ["scripts/predict.py", "scripts/svm_model.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
