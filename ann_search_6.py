
import time
import numpy as np
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score

from config import SEED, N_TREES, N_NEIGHBORS_ANN, QUERY_CHARACTER
from utils import load_data, save_artifact

np.random.seed(SEED)

# 1. Поиск индекса персонажа по имени
def find_character_index(names, query_name):
    """Ищет индекс персонажа в массиве имён."""
    names = np.array(names, dtype=str)
    # точное совпадение
    matches = np.where(names == query_name)[0]
    if len(matches) > 0:
        return int(matches[0])

    # если точного нет - берём ближайший по алфавиту
    print(f"  [warn] '{query_name}' не найден. Ищу ближайший по алфавиту...")
    sorted_names = sorted(names)
    # находим первый, который >= query_name
    candidates = [n for n in sorted_names if n >= query_name]
    if candidates:
        fallback = candidates[0]
    else:
        fallback = sorted_names[-1]
    print(f"  [warn] Использую '{fallback}' вместо '{query_name}'")
    return int(np.where(names == fallback)[0][0])

# 2. Точный поиск (brute-force через sklearn)
def exact_search(X, query_vector, k):
    """Точный поиск k ближайших соседей (евклидова метрика)."""
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X)
    dists, indices = nn.kneighbors(query_vector.reshape(1, -1))
    return indices[0], dists[0]


# 3. ANN-поиск через Annoy
def annoy_search(X, query_vector, k, n_trees):
    """Строит AnnoyIndex и ищет k соседей."""
    dim = X.shape[1]
    index = AnnoyIndex(dim, metric="euclidean")

    # добавляем все векторы
    for i, vec in enumerate(X):
        index.add_item(i, vec)

    # строим индекс
    t0 = time.time()
    index.build(n_trees)
    build_time = time.time() - t0

    # ищем соседей
    t0 = time.time()
    indices = index.get_nns_by_vector(query_vector, k, include_distances=False)
    search_time = time.time() - t0

    return indices, build_time, search_time

# 4. Влияние n_trees на recall и время
def study_n_trees(X, query_vector, k, exact_indices, tree_grid):
    """
    Для каждого n_trees из tree_grid строит Annoy, ищет соседей
    и считает recall@k и время поиска.
    """
    recalls = []
    times = []

    for n_trees in tree_grid:
        indices, build_t, search_t = annoy_search(X, query_vector, k, n_trees)

        # recall@k = |ANN_top_k ∩ exact_top_k| / k
        overlap = len(set(indices) & set(exact_indices))
        recall = overlap / k
        recalls.append(recall)
        times.append(search_t)

        print(f"  n_trees={n_trees:3d}: recall={recall:.3f}, "
              f"search_time={search_t*1000:.2f} ms, build_time={build_t:.3f} s")

    return np.array(recalls), np.array(times)

# 5. main
def main():
    data = load_data()
    X_train = data["X_train"].astype(float)
    y_train = data["y_train"].astype(int)
    names_train = data["names_train"]

    print("Часть 6. ANN (Annoy)")
    print("=" * 60)
    print(f"n_trees = {N_TREES}, k = {N_NEIGHBORS_ANN}, "
          f"query = '{QUERY_CHARACTER}'")

    #  Находим персонажа 
    query_idx = find_character_index(names_train, QUERY_CHARACTER)
    query_name = names_train[query_idx]
    query_vector = X_train[query_idx]

    print(f"\n[1] Персонаж найден: '{query_name}' (индекс {query_idx})")

    # Точный поиск (исключаем самого себя) 
    print("\n[2] Точный поиск (brute-force, sklearn)...")
    # Исключаем query из обучающей выборки, чтобы не найти самого себя
    mask = np.ones(len(X_train), dtype=bool)
    mask[query_idx] = False
    X_search = X_train[mask]
    names_search = names_train[mask]

    # ищем k+1, чтобы потом убрать самого себя, если попадётся
    exact_idx_local, exact_dists = exact_search(X_search, query_vector, N_NEIGHBORS_ANN)
    exact_names = names_search[exact_idx_local]
    exact_global_idx = np.where(mask)[0][exact_idx_local]

    print(f"  Точные соседи: {list(exact_names)}")

    # ANN-поиск (с фиксированным n_trees)
    print(f"\n[3] ANN-поиск (n_trees={N_TREES})...")
    ann_idx, build_t, search_t = annoy_search(
        X_search, query_vector, N_NEIGHBORS_ANN, N_TREES
    )
    ann_names = names_search[ann_idx]

    print(f"  ANN-соседи: {list(ann_names)}")
    print(f"  build_time = {build_t:.4f} s, search_time = {search_t*1000:.2f} ms")

    # recall@k
    overlap = len(set(ann_idx) & set(exact_idx_local))
    recall = overlap / N_NEIGHBORS_ANN
    print(f"  recall@{N_NEIGHBORS_ANN} = {recall:.3f} ({overlap} из {N_NEIGHBORS_ANN})")

    # Исследование n_trees 
    print("\n[4] Исследование влияния n_trees...")
    tree_grid = [1, 2, 3, 5, 7, 9, 12, 15, 20, 30, 50]
    recalls, times = study_n_trees(
        X_search, query_vector, N_NEIGHBORS_ANN, exact_idx_local, tree_grid
    )

    # график recall vs n_trees
    plt.figure(figsize=(8, 4))
    plt.plot(tree_grid, recalls, marker="o")
    plt.xlabel("n_trees")
    plt.ylabel(f"recall@{N_NEIGHBORS_ANN}")
    plt.title(f"Влияние n_trees на recall@{N_NEIGHBORS_ANN}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("artifacts/ann_recall_vs_trees.png", dpi=120)
    plt.close()
    print("  сохранено в  artifacts/ann_recall_vs_trees.png")

    # график времени поиска vs n_trees
    plt.figure(figsize=(8, 4))
    plt.plot(tree_grid, times, marker="s", color="orange")
    plt.xlabel("n_trees")
    plt.ylabel("время поиска, сек")
    plt.title("Время поиска vs n_trees")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("artifacts/ann_time_vs_trees.png", dpi=120)
    plt.close()
    print("  сохранено в artifacts/ann_time_vs_trees.png")

    #  f1_score на валидации
    # Оценим качество классификации 1NN на основе ANN-поиска.
    print("\n[5] Оценка f1_score на валидации (1NN через Annoy)...")
    X_val = data["X_val"].astype(float)
    y_val = data["y_val"].astype(int)

    # строим один AnnoyIndex на всём train для скорости
    dim = X_train.shape[1]
    index_full = AnnoyIndex(dim, metric="euclidean")
    for i, vec in enumerate(X_train):
        index_full.add_item(i, vec)
    index_full.build(N_TREES)

    # предсказания 1NN для валидации
    y_pred = []
    for x in X_val:
        nn_idx = index_full.get_nns_by_vector(x, 1, include_distances=False)[0]
        y_pred.append(y_train[nn_idx])

    f1 = f1_score(y_val, y_pred, average="binary")
    print(f"  f1_score (1NN через Annoy, n_trees={N_TREES}) = {f1:.4f}")

    # Сохраняем результаты
    results = {
        "query_name": query_name,
        "exact_names": list(exact_names),
        "ann_names": list(ann_names),
        "recall": recall,
        "build_time": build_t,
        "search_time": search_t,
        "tree_grid": tree_grid,
        "recalls": recalls.tolist(),
        "times": times.tolist(),
        "f1_score": f1,
    }
    save_artifact("ann_results", results)
    print("\nСохранено в artifacts/ann_results.pkl")


if __name__ == "__main__":
    main()