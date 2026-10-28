
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

from config import SEED, K_MAX, KNN_P
from utils import load_data, save_artifact

np.random.seed(SEED)

# Метрика Минковского
def minkowski_distance(X, x, p):
    """
    Считает расстояния от точки x до всех точек матрицы X.
    X: (n, d), x: (d,), возвращает вектор (n,).
    """
    diff = np.abs(X - x)             
    return (diff ** p).sum(axis=1) ** (1.0 / p)


# 2. Веса соседей
def compute_weights(weights_type, dists):
    """
    dists - отсортированные по возрастанию расстояния до k ближайших соседей.
    Возвращает массив весов той же длины.
    """
    k = len(dists)

    if weights_type == "uniform":
        return np.ones(k)

    if weights_type == "distance":
        # 1/d, чтобы не делить на 0 - добавляем эпсилон
        return 1.0 / (dists + 1e-9)

    if weights_type == "linear":
        # i-й сосед (i=0 самый близкий) получает вес (k - i) / k
        return np.array([(k - i) / k for i in range(k)])

    if weights_type == "exponential":
        # q^(i-1), q = 0.5
        q = 0.5
        return np.array([q ** i for i in range(k)])

    raise ValueError(f"Неизвестный тип весов: {weights_type}")


# 3. Классификатор kNN
def knn_classify(X_train, y_train, X_query, k, p=KNN_P, weights_type="linear"):
    """
    Классифицирует все объекты X_query.
    Возвращает массив предсказаний (0/1).
    """
    preds = np.zeros(len(X_query), dtype=int)

    for i, x in enumerate(X_query):
        # расстояния до всех обучающих объектов
        dists = minkowski_distance(X_train, x, p)

        # индексы k ближайших
        idx = np.argsort(dists)[:k]

        # веса соседей
        w = compute_weights(weights_type, dists[idx])
        labels = y_train[idx]

        # взвешенное голосование
        # суммируем веса по классам, берём класс с большей суммой
        score_0 = w[labels == 0].sum()
        score_1 = w[labels == 1].sum()

        preds[i] = 1 if score_1 >= score_0 else 0

    return preds



# LOO-подбор k

def loo_knn(X, y, k_max, p=KNN_P, weights_type="linear"):
    """
    Leave-one-out для kNN на обучающей выборке.
    Для каждого объекта классифицируем его по всем остальным.
    Возвращает массив accuracy по каждому k = 1..k_max.
    """
    n = len(X)
    # накапливаем число правильных ответов для каждого k
    correct = np.zeros(k_max + 1)

    for i in range(n):
        # обучающая выборка без i-го объекта
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr, y_tr = X[mask], y[mask]

        # расстояния от x_i до всех остальных
        dists = minkowski_distance(X_tr, X[i], p)
        order = np.argsort(dists)

        # для каждого k берём первых k соседей из этого порядка
        for k in range(1, k_max + 1):
            idx = order[:k]
            w = compute_weights(weights_type, dists[idx])
            labels = y_tr[idx]

            score_0 = w[labels == 0].sum()
            score_1 = w[labels == 1].sum()
            pred = 1 if score_1 >= score_0 else 0

            if pred == y[i]:
                correct[k] += 1

    return correct / n


def main():
    data = load_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]


    print("Часть 2. kNN (вариант 15: minkowski p=3, linear)")
    print("=" * 60)
    print("train:", X_train.shape, " val:", X_val.shape)

    #  LOO для поиска лучшего k 
    print("\n[1] LOO-подбор k = 1..%d ..." % K_MAX)
    t0 = time.time()
    loo_acc = loo_knn(X_train, y_train, K_MAX, p=KNN_P, weights_type="linear")
    print("    время: %.1f сек" % (time.time() - t0))


    accs = loo_acc[1:]
    best_k = int(np.argmax(accs)) + 1
    print("    лучший k = %d (LOO acc = %.4f)" % (best_k, accs[best_k - 1]))

    # График LOO 
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, K_MAX + 1), accs, marker="o")
    plt.axvline(best_k, color="red", linestyle="--", label=f"best k = {best_k}")
    plt.xlabel("k")
    plt.ylabel("LOO accuracy")
    plt.title("Зависимость LOO accuracy от k (linear weights, minkowski p=3)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/knn_loo.png", dpi=120)
    plt.close()
    print("    сохранено в  artifacts/knn_loo.png")

    # 5.3. Сравнение 4 типов весов на валидации 
    print("\n[2] Сравнение весов на валидации при k = %d" % best_k)
    weight_results = {}
    for w_type in ["uniform", "distance", "linear", "exponential"]:
        preds = knn_classify(X_train, y_train, X_val, k=best_k,
                             p=KNN_P, weights_type=w_type)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="binary")
        weight_results[w_type] = {"acc": acc, "f1": f1}
        print("    %-12s acc=%.4f  f1=%.4f" % (w_type, acc, f1))

    #  Сравнение со sklearn 
    print("\n[3] Сравнение со sklearn KNeighborsClassifier")
    sk_model = KNeighborsClassifier(
        n_neighbors=best_k, metric="minkowski", p=KNN_P, weights="distance"
    )
    sk_model.fit(X_train, y_train)
    sk_preds = sk_model.predict(X_val)
    sk_acc = accuracy_score(y_val, sk_preds)
    sk_f1 = f1_score(y_val, sk_preds, average="binary")
    print("    sklearn (distance) acc=%.4f  f1=%.4f" % (sk_acc, sk_f1))

    # для нашего лучшего типа весов
    my_preds = knn_classify(X_train, y_train, X_val, k=best_k,
                            p=KNN_P, weights_type="linear")
    my_acc = accuracy_score(y_val, my_preds)
    my_f1 = f1_score(y_val, my_preds, average="binary")
    print("    наш    (linear)    acc=%.4f  f1=%.4f" % (my_acc, my_f1))

    # Сохраняем результаты 
    results = {
        "best_k": best_k,
        "loo_acc": loo_acc,
        "weight_results": weight_results,
        "sklearn_acc": sk_acc,
        "sklearn_f1": sk_f1,
    }
    save_artifact("knn_results", results)
    print("\nСохранил artifacts/knn_results.pkl")


if __name__ == "__main__":
    main()