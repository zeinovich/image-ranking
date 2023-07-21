import streamlit as st
import requests
import base64
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("./logs/frontend.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

WIDTH = 360


def predict(img_file):
    placeholder = st.empty()
    placeholder.write("Classifying...")

    im_bytes = img_file.getvalue()
    im_b64 = base64.b64encode(im_bytes).decode("utf8")

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    payload = json.dumps({"image": im_b64})
    response = requests.post(
        "http://backend:8888/api/v1.0/predict", data=payload, headers=headers
    )

    placeholder.empty()
    if response.status_code == 200:
        predictions = response.json()["predictions"]
        return predictions

    else:
        st.write("Error in classification")


st.write("Test")

img_file = st.file_uploader("Upload file", type=["png", "jpg", "jpeg"])

if img_file is not None:

    predictions = None
    logger.info(f"Got image: {type(img_file)}")
    st.image(img_file, width=WIDTH)
    st.write("")

    if st.button("***Find similar***"):
        predictions = predict(img_file)

    if predictions is not None:
        for pred in predictions:
            st.subheader(f"({pred['index']}) {pred['productdisplayname']}")
            col1, col2 = st.columns(2)
            col1.image(pred["default"])
            col2._html(pred["productdescriptors"], scrolling=True, height=450)
            st.write("")
