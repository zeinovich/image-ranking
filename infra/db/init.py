from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
from os import getenv

load_dotenv('../db.env')

USER = getenv('POSTGRES_USER')
PASSWORD = getenv('POSTGRES_PASSWORD')
DB = getenv('POSTGRES_DB')
DATA_PATH = getenv('DATA_PATH')


def read_data(path: str) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df.columns = df.columns.str.lower()

    return df


engine = create_engine(f'postgresql://{USER}:{PASSWORD}@localhost:5432/{DB}')

df = read_data(DATA_PATH)
df.to_sql('styles_v1', engine)
