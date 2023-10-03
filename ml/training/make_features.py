import numpy as np
import pandas as pd

from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s

from utils import make_url_df, ImageURLWithoutClasses


@torch.no_grad()
def get_embeddings(
    model: nn.Module, dataloader: DataLoader, device: str, batch_size: int
) -> np.ndarray:

    model.eval()
    model.to(device)
    write = 0

    with torch.no_grad():
        for x, idx in tqdm(dataloader, total=len(dataloader), miniters=1):

            x = x.to(device)
            emb = model(x)

            emb = emb.detach()
            emb = emb.reshape(batch_size, -1)

            idx = idx.detach()
            idx = idx.reshape(batch_size, -1)

            # if is written first time
            if not write:
                write += 1
                embeddings = emb
                idxs = idx

            else:
                embeddings = torch.cat((embeddings, emb), 0)
                idxs = torch.cat((idxs, idx), 0)

        embeddings = torch.cat((idxs, embeddings), 1)
        embeddings = embeddings.detach().cpu().numpy()

    return embeddings


def read_input_file(input_file: str) -> pd.DataFrame:
    if input_file.endswith(".csv"):
        df = pd.read_csv(input_file)

    elif input_file.endswith(".pkl"):
        df = pd.read_pickle(input_file)

    elif input_file.endswith(".parquet"):
        df = pd.read_parquet(input_file)

    else:
        raise ValueError("The input file must be a CSV, parquet or a pickle file.")

    return df


def save_output_file(emb: np.ndarray, output_file: str) -> None:
    emb_df = pd.DataFrame(emb)

    if output_file.endswith(".csv"):
        emb_df.to_csv(output_file, index=False)

    elif output_file.endswith(".pkl"):
        emb_df.to_pickle(output_file)

    elif output_file.endswith(".parquet"):
        emb_df.to_parquet(output_file)

    else:
        raise ValueError("The output file must be a CSV, parquet or a pickle file.")


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make embeddings for the images in the dataset."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data",
        help="Path to the input file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data",
        help="Path to the output file.",
    )
    parser.add_argument(
        "--view",
        type=str,
        default="default",
        help="View of the image to use. One of [default, left, right]. \
Uses 'default' by default ",
    )
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    parser.add_argument(
        "--img_size", type=int, default=224, help="Image size. Defaults to 224"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size. Defaults to 1"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers. Defaults to 1"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use. Defaults to 'cpu'"
    )
    args = parser.parse_args()

    return args


def main():
    args = cli()

    print("ARGS:")
    for k, arg in args._get_kwargs():
        print(f"{k}={arg}")
    print("---------------")

    # Set the constants.
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    INPUT_FILE = args.input_file
    OUTPUT_FILE = args.output_file
    IMG_SIZE = args.img_size
    MODEL_PATH = args.model_path
    DEVICE = args.device

    # Load the model.
    model = efficientnet_v2_s()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))

    model = nn.Sequential(model.features, model.avgpool)
    print("Model is loaded")

    # Make the URL DataFrame.
    df = read_input_file(INPUT_FILE)
    url_df = make_url_df(df, view=args.view)
    print(f"URL Dataframe is made (shape={url_df.shape})")

    # Make the Dataset and DataLoader.
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            T.CenterCrop((IMG_SIZE, IMG_SIZE)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageURLWithoutClasses(url_df, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
    )
    print("DataLoader is made")

    # Get the embeddings.

    print("Extracting features...")
    embeddings = get_embeddings(
        model, dataloader, device=args.device, batch_size=BATCH_SIZE
    )
    print("Embeddings are made")
    print(f"{type(embeddings)=}     {embeddings.shape}")

    # Save the embeddings.
    save_output_file(embeddings, OUTPUT_FILE)
    print("Embeddings are saved")
    print("Exiting")


# To test run (from project's root):
# python ml/training/make_features.py \
#        --input_file ml/data/styles-sample.pkl \
#        --output_file ml/data/test_output.csv \
#        --model_path backend/ml-models/feature_extractor.pth \
#        --img_size 300 \
#        --batch_size 1 \

# Takes about a 2 minutes on a CPU(4 cores) to test on 100 images

if __name__ == "__main__":
    main()
