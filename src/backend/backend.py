from flask import Flask, request, jsonify
from PIL import Image
from feature_extractor.extractor import FeatureExtractor
from ranker.ranker import Ranker
from io import BytesIO
from sqlalchemy import create_engine
import pandas as pd
import pickle
import logging
import base64
from dotenv import load_dotenv
from os import getenv

load_dotenv('../db.env')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.FileHandler('../logs/backend.log'),
                              logging.StreamHandler()])

logger = logging.getLogger('backend')

FEATURE_EXTRACTOR_PATH = './backend/ML-models/feature-extractor.pth'
RANKER_PATH = './backend/ML-models/ranker.pkl'
SCALER_PATH = './backend/ML-models/scaler.pkl'

USER = getenv('POSTGRES_USER')
PASSWORD = getenv('POSTGRES_PASSWORD')
DB = getenv('POSTGRES_DB')

logger.info(f'{FEATURE_EXTRACTOR_PATH=}')
logger.info(f'{RANKER_PATH=}')
logger.info(f'{SCALER_PATH=}')

assert USER is not None, 'Got None db username'
assert PASSWORD is not None, 'Got None db password'
assert DB is not None, 'Got None db name'

logger.info('Successfully loaded environment')
try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

except Exception as e:
    logger.error(f'Error loading scaler: {e}')
    logger.info('Setting scaler to None')
    scaler = None

try:
    feature_extractor = FeatureExtractor(FEATURE_EXTRACTOR_PATH,
                                         scaler=scaler)
    ranker = Ranker(RANKER_PATH)

except Exception as e:
    logger.error(f'Error loading model: {e}')
    raise e

try:
    connection = f'postgresql://{USER}:{PASSWORD}@localhost:5432/{DB}'
    engine = create_engine(connection)
    logger.info('Connected to DB')

except Exception as e:
    logger.error(f'Error connecting to DB: {e}')
    raise e

app = Flask(__name__)


def get_predictions(data: dict) -> list:
    image = data['image']
    logger.info(f'Got image: {type(image)}')

    image = Image.open(BytesIO(base64.b64decode(image)))
    logger.info(f'Opened image: {type(image)} of size: {image.size}')

    features = feature_extractor.extract(image)
    logger.info(f'Extracted features: {features.shape}')

    predictions = ranker.rank(features).tolist()
    logger.info(f'Predictions: {predictions}')

    return predictions


def get_info_from_db(ids: list):
    ids = tuple(ids)
    logger.info(f'Got ids: {ids}')

    query = f"""
    SELECT t.default, t.search, t.back, t.front, t.left, t.right, t.top
    FROM styles_v1 as t
    WHERE index IN {ids}
    """

    logger.info(f'Query: {query}')

    try:
        df = pd.read_sql(query, engine)
        logger.info(f'Got df: {df.shape}')

        predictions = df.to_dict(orient='records')
        logger.info(f'Got predictions: {len(predictions)}')

        return predictions

    except Exception as e:
        logger.error(f'Error getting info from DB: {e}')


@app.route('/api/v1.0/predict', methods=['POST'])
def predict():

    logger.info('Got request')

    try:
        data = request.get_json()
        logger.info(f'Got JSON: {type(data)}')

        prediction_ids = get_predictions(data)
        predictions_urls = get_info_from_db(prediction_ids)

        return jsonify({'predictions': predictions_urls,
                        'status_code': 200})

    except Exception as e:
        logger.error(f'Error in predict: {e}')
        return jsonify({'error': f'{e}',
                        'status_code': 400})


@app.route('/api/v1.0/health', methods=['GET'])
def health():
    return jsonify({'status': 0})


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8888)
