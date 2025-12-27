import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
import os

MODEL_DIR = "/app/trained_model"
os.makedirs(MODEL_DIR, exist_ok=True)

ENCODER_PATH = f"{MODEL_DIR}/encoder.pkl"
FEATURE_COLS_PATH = f"{MODEL_DIR}/feature_cols.pkl"


def save_encoder(encoder):
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)


def load_encoder():
    with open(ENCODER_PATH, "rb") as f:
        return pickle.load(f)


def save_feature_columns(columns):
    with open(FEATURE_COLS_PATH, "wb") as f:
        pickle.dump(columns, f)


def load_feature_columns():
    with open(FEATURE_COLS_PATH, "rb") as f:
        return pickle.load(f)


def drop_unused_columns(df: pd.DataFrame):
    DROP_COLS = ["Diameter"]
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def transform_data(df: pd.DataFrame, encoder=None):
    df = drop_unused_columns(df)

    if encoder is None:
        encoder = load_encoder()

    encoded = encoder.transform(df[["Sex"]])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(["Sex"]),
        index=df.index
    )

    df = pd.concat([df.drop(columns=["Sex"]), encoded_df], axis=1)

    feature_cols = load_feature_columns()
    df = df.reindex(columns=feature_cols, fill_value=0)

    return df


def prepare_training_data(df: pd.DataFrame):
    df = drop_unused_columns(df)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(df[["Sex"]])
    save_encoder(encoder)

    encoded = encoder.transform(df[["Sex"]])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(["Sex"]),
        index=df.index
    )
    df = pd.concat([df.drop(columns=["Sex"]), encoded_df], axis=1)

    feature_cols = list(df.drop(columns=["Rings"]).columns)
    save_feature_columns(feature_cols)

    return df