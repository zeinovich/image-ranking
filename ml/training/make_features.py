import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from io import BytesIO
import requests
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from utils import make_url_df


class ImageURLWithoutClasses(Dataset):
    def __init__(self, url_df: pd.DataFrame, transform=None):
        self.url_df = url_df
        self.transform = transform

    def __getitem__(self, index):
        idx, url = self.url_df.iloc[index]
        response = requests.get(url)
        x = Image.open(BytesIO(response.content))
        y = torch.Tensor([int(idx)])

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.url_df)


@torch.no_grad()
def get_embeddings(model, dataloader, device, batch) -> np.ndarray:

    model.eval()
    model.to(device)
    embeddings = {}

    for x, idx in tqdm(dataloader, total=len(dataloader), miniters=1):

        x = x.to(device)
        emb = model(x)

        emb = emb.detach().cpu().numpy()
        emb = emb.reshape(batch, -1)

        idx = idx.detach().cpu().numpy()
        idx = int(idx.reshape(1, -1)[0])

        embeddings[idx] = emb

    return embeddings


def process_embeddings(embeddings: dict) -> pd.DataFrame:
    """
    Process the embeddings dictionary into a numpy array.

    Args:
        embeddings (dict): Dictionary with the embeddings.

    Returns:
        np.ndarray: Numpy array with the embeddings.
    """

    output_dim = list(embeddings.values())[0].shape[1]
    embeddings = {k: v.reshape(output_dim) for k, v in embeddings.items()}
    emb_df = pd.DataFrame.from_dict(embeddings, orient="index")
    emb_df = emb_df.reset_index()
    emb_df = emb_df.rename(columns={"index": "id"})
    emb_df = emb_df.sort_values(by="id")
    return emb_df


def read_input_file(input_file: str) -> pd.DataFrame:
    if input_file.endswith(".csv"):
        df = pd.read_csv(input_file)

    elif input_file.endswith(".pkl"):
        df = pd.read_pickle(input_file)

    elif input_file.endswith(".parquet"):
        df = pd.read_parquet(input_file)

    else:
        raise ValueError("The input file must be a CSV or a pickle file.")

    return df


def save_output_file(emb_df: pd.DataFrame, output_file: str):
    if output_file.endswith(".csv"):
        emb_df.to_csv(output_file, index=False)

    elif output_file.endswith(".pkl"):
        emb_df.to_pickle(output_file)

    elif output_file.endswith(".parquet"):
        emb_df.to_parquet(output_file)

    else:
        raise ValueError("The output file must be a CSV or a pickle file.")


def cli():
    parser = argparse.ArgumentParser(
        description="Make embeddings for the images in the dataset."
    )
    parser.add_argument(
        "--input_file", type=str, default="data", help="Path to the input file."
    )
    parser.add_argument(
        "--output_file", type=str, default="data", help="Path to the output file."
    )
    parser.add_argument(
        "--view",
        type=str,
        default="default",
        help="View of the image to use. One of [default, left, right]",
    )
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    parser.add_argument("--img_size", type=int, default=224, help="Image size.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use.")
    args = parser.parse_args()
    return args


def main():
    args = cli()
    print(args)

    # Set the constants.
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    INPUT_FILE = args.input_file
    OUTPUT_FILE = args.output_file
    IMG_SIZE = args.img_size

    # Load the model.
    model = torch.load(args.model_path, map_location=args.device)
    print("model loaded")

    # Make the URL DataFrame.
    df = read_input_file(INPUT_FILE)
    url_df = make_url_df(df, view=args.view)
    print(f"url_df made {url_df.shape}")

    # Make the Dataset and DataLoader.
    transform = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ImageURLWithoutClasses(url_df, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
    )
    print("dataset and dataloader made")

    # Get the embeddings.
    embeddings = get_embeddings(model, dataloader, device=args.device, batch=BATCH_SIZE)
    print("embeddings made")
    emb_df = process_embeddings(embeddings)
    print("embeddings processed")

    # Save the embeddings.

    save_output_file(emb_df, OUTPUT_FILE)
    print("embeddings saved")


if __name__ == "__main__":
    main()
