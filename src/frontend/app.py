import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64
import json
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.FileHandler('../logs/frontend.log'),
                              logging.StreamHandler()])

logger = logging.getLogger(__name__)

def predict(img_file):
    st.write('Classifying...')
    
    try:
        img = Image.open(img_file)
        logger.info(f'Opened image: {type(img)} of size: {img.size}')

        # save image to local file
        with open('tmp/tmp.jpg', 'w', encoding='utf-8') as f:
            img.save(f, format='JPEG')
        
        logger.info('Saved image to local file')

    except Exception as e:
        logger.error(f'Error in image: {e}')
        st.write('Error in image')
        return

    with open('tmp/tmp.jpg', 'rb') as f:
        im_bytes = f.read()

    im_b64 = base64.b64encode(im_bytes).decode("utf8")
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    
    payload = json.dumps({"image": im_b64})
    response = requests.post('http://localhost:8888/api/v1.0/predict',
                             data=payload, headers=headers)
        

    if response.status_code == 200:
        predictions = response.json()['predictions']
        st.write(f'Predictions: {predictions}')

    else:
        st.write('Error in classification')

st.write('Test')

img_file = st.file_uploader('Upload file', type=['png', 'jpg', 'jpeg'])

if img_file is not None:
    
    logger.info(f'Got image: {type(img_file)}')
    st.image(img_file, caption='Uploaded Image', use_column_width=True)
    st.write('')

    if st.button('Classify'):
        predict(img_file)
        