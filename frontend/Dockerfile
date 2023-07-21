FROM python:3.9
WORKDIR /usr/src/frontend

COPY ./requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r ./requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run", "app.py" ]