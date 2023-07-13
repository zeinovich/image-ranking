from flask import Flask, request, jsonify
from PIL import Image
from feature_extractor.extractor import FeatureExtractor
from ranker.ranker import Ranker
from io import BytesIO, StringIO
import pickle
import logging
import base64

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.FileHandler('../logs/backend.log'),
                              logging.StreamHandler()])

logger = logging.getLogger(__name__)

FEATURE_EXTRACTOR_PATH = './backend/ML-models/feature-extractor.pth'
RANKER_PATH = './backend/ML-models/ranker.pkl'
SCALER_PATH = './backend/ML-models/scaler.pkl'

logger.info(f'{FEATURE_EXTRACTOR_PATH=}')
logger.info(f'{RANKER_PATH=}')
logger.info(f'{SCALER_PATH=}')

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

app = Flask(__name__)


@app.route('/api/v1.0/predict', methods=['POST'])
def predict():

    logger.info('Got request')
    
    try:
        data = request.get_json()
        logger.info(f'Got JSON: {type(data)}')

        image = data['image']
        logger.info(f'Got image: {type(image)}')

        image = Image.open(BytesIO(base64.b64decode(image)))
        logger.info(f'Opened image: {type(image)} of size: {image.size}')

        features = feature_extractor.extract(image)
        logger.info(f'Extracted features: {features.shape}')

        predictions = ranker.rank(features).tolist()
        logger.info(f'Predictions: {predictions}')

        return jsonify({'predictions': predictions,
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
