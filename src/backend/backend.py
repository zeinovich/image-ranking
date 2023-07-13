from flask import Flask, request, jsonify
from PIL import Image
from feature_extractor.extractor import FeatureExtractor
from ranker.ranker import Ranker
import pickle
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='../logs/backend.log',
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
    try:
        data = request.get_json(force=True)
        logger.info(f'Got request: {data}')
        image = data['files']['image']
        logger.info(f'Got image: {type(image)}')

        image = Image.open(image)
        logger.info(f'Opened image: {type(image)} of size: {image.size}')

        features = feature_extractor.extract(image)
        logger.info(f'Extracted features: {features.shape}')

        predictions = ranker.predict(features)
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
