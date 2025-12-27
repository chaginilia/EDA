import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from eda import prepare_training_data


if __name__ == '__main__':
    abalone = fetch_ucirepo(id=1)
    X = abalone.data.features
    y = abalone.data.targets

    df = X.copy()
    df["Rings"] = y

    df = prepare_training_data(df)

    x = df.drop(["Rings"], axis=1)
    y = df["Rings"]

    model = CatBoostRegressor(
        iterations=500,
        loss_function='RMSE',
        learning_rate=None,
        random_seed=42,
        verbose=100
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size=.85,
        random_state=42
    )

    cat_features = []

    model.fit(
        x_train,
        y_train,
        eval_set=(x_test, y_test),
        cat_features=cat_features
    )

    cv(
        Pool(x, y),
        model.get_params(),
        fold_count=5
    )

    model.save_model('/app/trained_model/trained_model.cbm')
    print("Модель сохранена")
