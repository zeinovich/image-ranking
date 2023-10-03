import pandas as pd
from PIL import Image
from io import BytesIO
import requests

import torch
from torch.utils.data import Dataset


def make_url_df(df: pd.DataFrame, view="default") -> pd.DataFrame:
    """
    Make a DataFrame with the image URLs for each ID.

    Args:
        df (pd.DataFrame): DataFrame with the IDs and images.
        view (str): View of the image to use.

    Returns:
        pd.DataFrame: DataFrame with the IDs and URLs.
    """
    df = df.reset_index()
    df = df.rename(columns={"index": "id"})
    df = df[["id", "images"]]

    assert all(df["images"].apply(lambda x: isinstance(x, dict)))
    assert all(df["images"].apply(lambda x: view in x))

    df["images"] = df["images"].apply(lambda x: x[view])
    return df


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
