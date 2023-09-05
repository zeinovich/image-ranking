FROM python:3.9
WORKDIR /usr/src/backend

COPY ./requirements.txt ./requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip && pip install -r ./requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8888

ENTRYPOINT [ "gunicorn", "--bind", "0.0.0.0:8888", "backend:app" ]
