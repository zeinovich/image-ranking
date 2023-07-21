import numpy as np
import pandas as pd

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