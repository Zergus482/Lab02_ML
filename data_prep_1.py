
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import SEED, TRAIN_URL, TEST_URL, POPULARITY_M, HOUSE_TOP_K
from utils import save_artifact


#1. Загрузка 

def load_raw():
    train = pd.read_csv(TRAIN_URL)
    test = pd.read_csv(TEST_URL)
    return train, test


# 2. Препроцессинг

DROP_COLS = ["S.No", "mother", "father", "heir", "spouse"]
ALIVE_FLAGS = ["isAliveMother", "isAliveFather", "isAliveHeir", "isAliveSpouse"]


def build_features(train, test):
    y = train["isAlive"].astype(int)
    names_train = train["name"].copy()
    names_test = test["name"].copy()

    train = train.drop(columns=["isAlive", "name"])
    test = test.copy()

    train = train.drop(columns=DROP_COLS, errors="ignore")
    test = test.drop(columns=DROP_COLS, errors="ignore")

    for df in (train, test):
        df["popularity_log"] = np.log10(df["popularity"].fillna(0) * POPULARITY_M + 1)
    train = train.drop(columns=["popularity"])
    test = test.drop(columns=["popularity"])

    # --- age / dateOfBirth: заполняем медианой + флаг ---
    for df in (train, test):
        df["is_age_known"] = df["age"].notna().astype(int)
        df["age"] = df["age"].fillna(df["age"].median())
        df["dateOfBirth"] = df["dateOfBirth"].fillna(df["dateOfBirth"].median())

    # culture: LabelEncoder 
    # пропуски -> "unknown", чтобы не терять строки
    train["culture"] = train["culture"].fillna("unknown")
    test["culture"] = test["culture"].fillna("unknown")

    le_culture = LabelEncoder()
    le_culture.fit(train["culture"])

    train["culture_le"] = le_culture.transform(train["culture"])

    #  если категория не встречалась в train ставим -1
    known = set(le_culture.classes_)
    test["culture_le"] = test["culture"].apply(
        lambda v: le_culture.transform([v])[0] if v in known else -1
    )

    train = train.drop(columns=["culture"])
    test = test.drop(columns=["culture"])

    # house: top-10 + other + unknown -> one-hot
    train["house"] = train["house"].fillna("unknown")
    test["house"] = test["house"].fillna("unknown")

    top_houses = train["house"].value_counts().head(HOUSE_TOP_K).index.tolist()

    def house_onehot(df):
        s = df["house"].where(df["house"].isin(top_houses), "other")
        dummies = pd.get_dummies(s, prefix="house").astype(int)
        df = pd.concat([df.drop(columns=["house"]), dummies], axis=1)
        return df

    train = house_onehot(train)
    test = house_onehot(test)

   # выравниваем тест и треин
    test = test.reindex(columns=train.columns, fill_value=0)

    # title -> просто флаг 
    for df in (train, test):
        df["has_title"] = df["title"].notna().astype(int)
    train = train.drop(columns=["title"])
    test = test.drop(columns=["title"])

    # isAliveX флаги -> 0/1/-1 
    for df in (train, test):
        for c in ALIVE_FLAGS:
            df[c] = df[c].map({1.0: 1, 0.0: 0}).fillna(-1).astype(int)

    # порядок колонок одинаковый
    test = test[train.columns]
    feature_names = train.columns.tolist()

    # масштабирование
    num_cols = ["age", "dateOfBirth", "popularity_log", "numDeadRelations"]
    scaler = StandardScaler()
    train[num_cols] = scaler.fit_transform(train[num_cols])
    test[num_cols] = scaler.transform(test[num_cols])

    # перестраховка и перевод всего в числа
    train = train.astype(float)
    test = test.astype(float)



    return (
        train.reset_index(drop=True),
        y.reset_index(drop=True),
        test.reset_index(drop=True),
        feature_names,
        names_train.reset_index(drop=True),
        names_test.reset_index(drop=True),
    )


#  Сплит 70/15/15

def split_70_15_15(X, y, names, seed=SEED):
    X_tr, X_tmp, y_tr, y_tmp, n_tr, n_tmp = train_test_split(
        X, y, names, test_size=0.3, random_state=seed, stratify=y
    )
    X_val, X_te, y_val, y_te, n_val, n_te = train_test_split(
        X_tmp, y_tmp, n_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te, n_tr, n_val, n_te


# --- 4. main ---

def main():
    print("Загружаю данные")
    train, test = load_raw()
    print("  train:", train.shape, " test:", test.shape)

    print("Обработка даннфх")
    X, y, X_kaggle, features, names_train, names_kaggle = build_features(train, test)

    print("Сплит 70/15/15")
    X_tr, X_val, X_te, y_tr, y_val, y_te, n_tr, n_val, n_te = \
        split_70_15_15(X, y, names_train)

    # переводим в numpy 
    data = {
        "X_train": X_tr.to_numpy(),
        "X_val":   X_val.to_numpy(),
        "X_test":  X_te.to_numpy(),
        "y_train": y_tr.to_numpy(),
        "y_val":   y_val.to_numpy(),
        "y_test":  y_te.to_numpy(),
        "X_full":  X.to_numpy(),   
        "y_full":  y.to_numpy(),
        "X_kaggle_test": X_kaggle.to_numpy(),
        "feature_names": features,
        "names_train": n_tr.to_numpy(),
        "names_val":   n_val.to_numpy(),
        "names_test":  n_te.to_numpy(),
        "names_kaggle_test": names_kaggle.to_numpy(),
    }
    save_artifact("data", data)

    print("\nГотово.")
    print("X_train:", X_tr.shape, " X_val:", X_val.shape, " X_test:", X_te.shape)
    print("доля alive: %.3f / %.3f / %.3f" % (y_tr.mean(), y_val.mean(), y_te.mean()))
    print("признаков:", len(features))
    print("сохранено в artifacts/data.pkl")


if __name__ == "__main__":
    main()