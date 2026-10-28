# Вспомогательные функции: сохранить/загрузить артефакты

import pickle
import os

ARTIFACTS = "artifacts"


def save_artifact(name, obj):
    """Сохраняет объект в artifacts/<name>.pkl"""
    os.makedirs(ARTIFACTS, exist_ok=True)
    with open(f"{ARTIFACTS}/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def load_artifact(name):
    """Загружает artifacts/<name>.pkl"""
    with open(f"{ARTIFACTS}/{name}.pkl", "rb") as f:
        return pickle.load(f)


def load_data():
    """Загружает подготовленные данные (см. data_prep.py)"""
    return load_artifact("data")