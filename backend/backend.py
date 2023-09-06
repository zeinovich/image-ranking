"""
Flask app for serving predictions

Usage:
    python backend.py
    gunicorn backend:app
"""

from flask import Flask, request, jsonify
from sqlalchemy import create_engine
from PIL import Image

from io import BytesIO
import pandas as pd
import pickle
import logging
import base64
import traceback

# from dotenv import load_dotenv
import os

from feature_extractor.extractor import FeatureExtractor
from ranker.ranker import Ranker
from segmentation import Segmentor

# load_dotenv("./backend.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        # logging.FileHandler("./logs/backend.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("backend")

FEATURE_EXTRACTOR_PATH = os.getenv("FEATURE_EXTRACTOR_PATH")
SEGMENTATION_MODEL = os.getenv("SEGMENTATION_MODEL")
RANKER_PATH = os.getenv("RANKER_PATH")
SCALER_PATH = os.getenv("SCALER_PATH")

USER = os.getenv("POSTGRES_USER")
PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB = os.getenv("POSTGRES_DB")

logger.info(os.getcwd())
logger.info(f"{FEATURE_EXTRACTOR_PATH=}")
logger.info(f"{SEGMENTATION_MODEL=}")
logger.info(f"{RANKER_PATH=}")
logger.info(f"{SCALER_PATH=}")

logger.info(f"{USER=}")
logger.info(f"{PASSWORD=}")
logger.info(f"{DB=}")

logger.info("Successfully loaded environment")

try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

except Exception as e:
    logger.error(f"Error loading scaler: {e}")
    logger.info("Setting scaler to None")
    scaler = None

try:
    feature_extractor = FeatureExtractor(
        model_path=FEATURE_EXTRACTOR_PATH, scaler=scaler
    )
    logger.info(f"{feature_extractor}")

    ranker = Ranker(model_path=RANKER_PATH)
    logger.info(f"{ranker}")

    segmentor = Segmentor(model_path=SEGMENTATION_MODEL)
    logger.info(f"{segmentor}")
    logger.info("Succesfully loaded models")

except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise e


try:
    connection = f"postgresql://{USER}:{PASSWORD}@postgres:5432/{DB}"
    engine = create_engine(connection)
    logger.info("Connected to DB")

except Exception as e:
    logger.error(f"Error connecting to DB: {e}")
    raise e

app = Flask(__name__)
logger.info("[READY] Flask app ready")


def load_image_from_json(data: dict) -> Image:
    """
    Load image from JSON

    Args:
        data (dict): JSON data

    Returns:
        Image: Image"""

    image = data["image"]
    logger.info(f"Got image: {type(image)}")

    image = Image.open(BytesIO(base64.b64decode(image)))
    return image


def get_predictions(
    image: Image, feature_extractor: FeatureExtractor, ranker: Ranker
) -> list:
    """
    Get predictions from image

    Args:
        image (Image): Image
        feature_extractor (FeatureExtractor): Feature extractor
        ranker (Ranker): Ranker

    Returns:
        list: Predictions"""

    logger.info(f"Opened image: {type(image)} of size: {image.size}")
    features = feature_extractor.extract(image)
    logger.info(f"Extracted features: {features.shape}")

    distances, ids = ranker.rank(features)
    logger.info(f"Predictions: {distances.shape}, {ids.shape}")

    return (distances.tolist(), ids.tolist())


def get_info_from_db(distances: list, ids: list) -> dict:
    """
    Get info from DB

    Args:
        distances (list): Distances
        ids (list): Ids

    Returns:
        dict: Predictions"""

    # sort to query in order
    sorted_ids = pd.DataFrame(data=[ids, distances]).T
    sorted_ids.columns = ["id", "distance"]

    # turn into tuple for SQL query
    ids = tuple(ids)
    logger.info(f"Got ids: {ids}")
    logger.info(f"Got ids_dist: {sorted_ids}")

    query = f"""
    SELECT *
    FROM styles_v1 as t
    WHERE index IN {ids}
    """

    logger.info(f"Query: {query}")

    try:
        df = pd.read_sql(query, engine)
        logger.info(f"Got df: {df.shape}")

        # merge df and sorted_ids
        df = (
            pd.merge(
                left=df,
                right=sorted_ids,
                how="left",
                left_on="index",
                right_on="id",
            )
            .sort_values("distance", ascending=True)
            .reset_index(drop=True)
        )

        predictions = df.to_dict(orient="records")
        logger.info(
            f'Got predictions: {[pred["index"] for pred in predictions]}'
        )

        return predictions

    except Exception as e:
        logger.error(f"Error getting info from DB: {e}")


def dump_image(image: Image):
    """
    Dump image to base64 format to make it JSON serializable

    Args:
        image (Image): Image

    Returns:
        str: Image in base64 format"""

    with open("./image.png", "wb") as img:
        image.save(img)

    with open("./image.png", "rb") as img:
        mystr = img.read()

    im_bytes = base64.b64encode(mystr).decode()
    return im_bytes


@app.route("/api/v1.0/predict", methods=["POST"])
def predict():

    logger.info("Got request")

    try:
        data = request.get_json()
        logger.info(f"Got JSON: {type(data)}")

        image = load_image_from_json(data)
        segmented_image, _ = segmentor.segment(image)

        # convert from RGBA to RGB and dump to base64
        segmented_image = Image.fromarray(segmented_image).convert("RGB")
        segmented_bytes = dump_image(segmented_image)

        distances, ids = get_predictions(
            segmented_image, feature_extractor, ranker
        )
        predictions_urls = get_info_from_db(distances, ids)

        if predictions_urls is None:
            return jsonify(
                {"error": "No predictions found", "status_code": 404}
            )

        return jsonify(
            {
                "predictions": predictions_urls,
                "segmented_image": segmented_bytes,
                "status_code": 200,
            }
        )

    except Exception as e:
        logger.error(f"Error in predict: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"{e}", "status_code": 400})


@app.route("/api/v1.0/health", methods=["GET"])
def health():
    return jsonify({"status": 200})


if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=8888)
