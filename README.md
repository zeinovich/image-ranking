# Image-Based Recommendation System
## Motivation
Recommendation systems are key for every retail to success. They greatly improve not only user experience, but also revenue as accurate recommendations make users buy more. State of the art recommendation systems developed by Amazon or Google use a great variety of features and user history, however I will build a baseline solution using only image similarity.
## System

![image](https://github.com/zeinovich/image-recsys/assets/114425094/f812e71f-6980-4318-8c8f-4da609e0c5a3)


##  Data
For now, I used only [Fashion Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) from kaggle. It contains ~44K images of clothes, shoes and accessories from retail stores. Also it provides a lot of meta about every images, like price, discount, display categories, etc. Data was divided 80%:20% train and test set.
## Models
As a baseline, I used *EfficientNetV2-S* for image feature extraction and *Nearest Neighbors* to query similar images. After feature extraction, StandardScaler is applied on features.

    base_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    model = nn.Sequential(base_model.features,
                           base_model.avgpool)

    knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
As all images in dataset are on white background, I decided to implement segmnetation as preprocessing. For that, [pretrained DeepLabV3 with ResNet-50 backbone](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/) is used. [Original segmentation pipeline](https://colab.research.google.com/drive/1P9Pyq92ywLa6SRPmfnctgzScsXvJrrOM?usp=sharing#scrollTo=w_7SNhWQIZn7)
### Training Pipeline
Dataset weighs too much for my laptop (25GB), so I decided to use custom ImageFolder. To get item, it sends *HTTP request* to the link provided in dataset. Then it processes it using BytesIO to read bytes from response and PIL Image class. To keep track of image IDs, ***y*** variable acts as target, but never gets used. See [ImageURLFolder implementation](https://github.com/zeinovich/image-recsys/blob/72ac903f9e81643c1781842d6a40d5bcbd6bcab2/ml/training/make_features.py#L20).

To train model, run command:

    python make_features.py --input_file PATH --output_file PATH --view STR --model_path PATH --img_size INT

- it accepts only csv, pkl and parquet files
- view argument must be one of "default", "right", "left"
- model must be of pth format
- to choose img_size see your model's docs

For validation StratifiedKFold was used. As classification metric *precision, recall and f1-score* were used.

Models achieved  **93%** precision, recall on subCategory and **99,8%** precision, recall on masterCategory.
## Data Storage
For data storage, I used plain [PostgreSQL](https://hub.docker.com/_/postgres). See [docker-compose file](https://github.com/zeinovich/image-recsys/blob/main/docker-compose.yaml). 

## Backend
[Backend](https://github.com/zeinovich/image-recsys/tree/main/backend) is written in Flask framework. It accepts image' bytes from frontend as *HTTP request*, [segments foreground](https://github.com/zeinovich/image-recsys/tree/main/backend/segmentation/segmentor.py) and [extracts features](https://github.com/zeinovich/image-recsys/tree/main/backend/feature_extractor/extractor.py) from it, then [KNN](https://github.com/zeinovich/image-recsys/tree/main/backend/ranker/ranker.py) queries *n* similar images. After that, it makes call to database to get additional info of images and sends *HTTP request* back to frontend.

To run backend on gunicorn (from /backend):

    gunicorn --bind 0.0.0.0:8888 --timeout 120 backend:app

On plain Flask (from /backend):

    python backend.py
    
## Frontend
[Frontend](https://github.com/zeinovich/image-recsys/tree/main/frontend) is written using [streamlit](https://streamlit.io/). It accepts image file and sends it to backend. After processing, it gets back queried images with additional info. For now, it displays only product name and description.

To run frontend (from /frontend):

    python app.py

## Docker
Dockerfiles:
- [Backend Dockerfile](https://github.com/zeinovich/image-recsys/blob/main/backend/Dockerfile)
- [Frontend Dockerfile](https://github.com/zeinovich/image-recsys/tree/blob/frontend/Dockerfile)
- [Postgres Dockerfile](https://hub.docker.com/_/postgres)
- [PgAdmin Dockerfile](https://hub.docker.com/r/dpage/pgadmin4/)

As orchestration, Docker Compose is used. See [docker-compose.yaml](https://github.com/zeinovich/image-recsys/blob/main/docker-compose.yaml)

Run (from /):

    docker compose up -d

Also, kubernetes (minikube) configuration is available. However, it doesn't work properly yet.
