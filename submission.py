
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from config import SEED, KNN_P, H_MAX, N_TREES
from utils import load_data, load_artifact

np.random.seed(SEED)


# Реализации всех моделей (для переобучения)
def kernel_quartic(r):
    r = np.asarray(r, dtype=float)
    mask = np.abs(r) <= 1
    out = np.zeros_like(r)
    out[mask] = (1 - r[mask] ** 2) ** 2
    return out


def minkowski_distance(X, x, p=KNN_P):
    diff = np.abs(X - x)
    return (diff ** p).sum(axis=1) ** (1.0 / p)


def compute_weights(weights_type, dists):
    k = len(dists)
    if weights_type == "uniform":
        return np.ones(k)
    if weights_type == "distance":
        return 1.0 / (dists + 1e-9)
    if weights_type == "linear":
        return np.array([(k - i) / k for i in range(k)])
    if weights_type == "exponential":
        q = 0.5
        return np.array([q ** i for i in range(k)])
    raise ValueError(weights_type)


def knn_predict(X_train, y_train, X_query, k, weights_type):
    preds = np.zeros(len(X_query), dtype=int)
    for i, x in enumerate(X_query):
        dists = minkowski_distance(X_train, x)
        idx = np.argsort(dists)[:k]
        w = compute_weights(weights_type, dists[idx])
        labels = y_train[idx]
        s0 = w[labels == 0].sum()
        s1 = w[labels == 1].sum()
        preds[i] = 1 if s1 >= s0 else 0
    return preds


def parzen_predict(X_train, y_train, X_query, h=None, mode="fixed", k=None):
    preds = np.zeros(len(X_query), dtype=int)
    for i, x in enumerate(X_query):
        dists = minkowski_distance(X_train, x)
        if mode == "fixed":
            h_i = h
        else:
            sorted_d = np.sort(dists)
            h_i = sorted_d[k - 1] if k - 1 < len(sorted_d) else sorted_d[-1]
            if h_i <= 0:
                h_i = 1e-6
        w = kernel_quartic(dists / h_i)
        s0 = w[y_train == 0].sum()
        s1 = w[y_train == 1].sum()
        preds[i] = 1 if s1 >= s0 else 0
    return preds


def potential_predict(X_train, y_train, X_query, h):
    preds = np.zeros(len(X_query), dtype=int)
    for i, x in enumerate(X_query):
        dists = minkowski_distance(X_train, x)
        potentials = kernel_quartic(dists / h)
        U0 = potentials[y_train == 0].sum()
        U1 = potentials[y_train == 1].sum()
        preds[i] = 1 if U1 >= U0 else 0
    return preds


# 2. main
def main():
    data = load_data()
    X_full = data["X_full"].astype(float)
    y_full = data["y_full"].astype(int)
    X_kaggle = data["X_kaggle_test"].astype(float)
    names_kaggle = data["names_kaggle_test"]

    print("Часть 8. Формирование submission.csv")
    print("=" * 60)
    print("X_full:", X_full.shape, " X_kaggle_test:", X_kaggle.shape)

    # --- 8.1. Собираем все результаты ---
    candidates = {}  # name -> {acc, f1, predict_fn, note}

    # kNN (все 4 веса - перебираем при выборе лучшего)
    try:
        knn_res = load_artifact("knn_results")
        best_k = knn_res["best_k"]
        for w in ["uniform", "distance", "linear", "exponential"]:
            r = knn_res["weight_results"][w]
            candidates[f"kNN ({w})"] = {
                "acc": r["acc"], "f1": r["f1"],
                "note": f"k={best_k}, {w}",
                "predict": (lambda X, k=best_k, w=w:
                            knn_predict(X_full, y_full, X, k, w)),
            }
    except FileNotFoundError:
        print("[warn] knn_results.pkl не найден")

    # aNN fixed / variable
    try:
        parz_res = load_artifact("parzen_results")
        candidates["aNN (fixed h)"] = {
            "acc": parz_res["acc_fixed"], "f1": parz_res["f1_fixed"],
            "note": f"h={parz_res['best_h']:.3f}",
            "predict": (lambda X, h=parz_res["best_h"]:
                        parzen_predict(X_full, y_full, X, h=h, mode="fixed")),
        }
        candidates["aNN (variable k)"] = {
            "acc": parz_res["acc_var"], "f1": parz_res["f1_var"],
            "note": f"k={parz_res['best_k']}",
            "predict": (lambda X, k=parz_res["best_k"]:
                        parzen_predict(X_full, y_full, X, mode="variable", k=k)),
        }
    except FileNotFoundError:
        print("[warn] parzen_results.pkl не найден")

    # potential
    try:
        pot_res = load_artifact("potential_results")
        candidates["Potential"] = {
            "acc": pot_res["acc_best"], "f1": pot_res["f1_best"],
            "note": f"h={pot_res['best_h']:.3f}",
            "predict": (lambda X, h=pot_res["best_h"]:
                        potential_predict(X_full, y_full, X, h=h)),
        }
    except FileNotFoundError:
        print("[warn] potential_results.pkl не найден")

    # logistic regression
    try:
        lin_res = load_artifact("linear_results")
        best_C = lin_res["best_C"]

        def _logreg_predict(X, C=best_C):
            model = LogisticRegression(
                C=C, penalty="l2", solver="liblinear",
                max_iter=1000, random_state=SEED
            )
            model.fit(X_full, y_full)
            return model.predict(X)

        candidates["LogisticRegression"] = {
            "acc": lin_res["acc_logreg"], "f1": lin_res["f1_logreg"],
            "note": f"L2, C={best_C:.3f}",
            "predict": _logreg_predict,
        }
    except FileNotFoundError:
        print("[warn] linear_results.pkl не найден")

    if not candidates:
        print("\nНет ни одного результата. Сначала запусти knn.py, parzen.py и т.д.")
        return

    #  Печатаем сводную таблицу 
    print("\n[1] Все модели (метрики на валидации):\n")
    print("    %-24s %-8s %-8s %s" % ("метод", "acc", "f1", "note"))
    print("    " + "-" * 65)
    for name, c in candidates.items():
        print("    %-24s %.4f   %.4f   %s"
              % (name, c["acc"], c["f1"], c["note"]))

    #  Выбираем лучшую по accuracy (при равенстве - по f1) 
    best_name = max(candidates, key=lambda n: (candidates[n]["acc"],
                                                candidates[n]["f1"]))
    best = candidates[best_name]
    print("\n[2] Лучшая модель: %s  (acc=%.4f, f1=%.4f)"
          % (best_name, best["acc"], best["f1"]))

    # Предсказание на kaggle test
    print("\n[3] Переобучение на всём train и предсказание на test...")
    y_pred = best["predict"](X_kaggle)

    # Собираем submission
    sub = pd.DataFrame({
        "name": names_kaggle,
        "isAlive": y_pred.astype(int),
    })
    sub.to_csv("submission.csv", index=False)

    print("    сохранил submission.csv")
    print("    shape:", sub.shape)
    print("    доля isAlive=1 в предсказаниях: %.3f" % sub["isAlive"].mean())
    print("\n    первые 10 строк:")
    print(sub.head(10).to_string(index=False))

    #  Финальная сводка в консоль 
    print("ИТОГ:")
    print("  лучшая модель: %s  (acc=%.4f, f1=%.4f)"
          % (best_name, best["acc"], best["f1"]))
    print("  submission.csv сохранён (строк: %d)" % len(sub))
 


if __name__ == "__main__":
    main()


# ИТОГОВЫЙ ВЫВОД
#
# Сравнение методов на валидации:
# метод                    acc      f1       note
#    kNN (uniform)            0.8205   0.8945   k=14, uniform
#    kNN (distance)           0.8205   0.8906   k=14, distance
#    kNN (linear)             0.8333   0.9003   k=14, linear
#    kNN (exponential)        0.8120   0.8817   k=14, exponential
#    aNN (fixed h)            0.7650   0.8601   h=1.160
#    aNN (variable k)         0.8162   0.8877   k=20
#    Potential                0.7863   0.8768   h=0.050
#    LogisticRegression       0.8291   0.8974   L2, C=100.000
# Все методы дают близкие метрики (разница в пределах 1–3%),
# так как используют одну метрику и похожую логику голосования.
# Лучший результат —  kNN (linear)    0.8333   0.9003   k=14, linear
# Из-за дисбаланса классов (78/22) ориентируемся на f1.
#
# Рекомендации по гиперпараметрам:
#   k для kNN        — оптимум 14 по LOO; рабочий диапазон 7–15,
#                      при k=1 модель шумит, при k>20 сглаживается.
#   веса kNN         — лучше себя показали linear/distance
#                      (uniform "замывает" ответ дальними соседями).
#   h для aNN        —  при h<0.2 веса почти нулевые
#                      (финитное ядро), при h>1.0 границы стираются.
#   k для aNN var    —  адаптивная ширина чуть лучше
#                      фиксированной в плотных/разреженных зонах.
#   метрика          — вариант требует Минковского с p=3; на практике
#                      p=2 и p=3 дают разницу <1%, p>5 заметно хуже.
#   ядро             — quartic (1-r²)² по качеству не хуже гаусса,
#                      но считается быстрее (нет exp).
#
# Влияние n_trees (ANN):
#   n_trees=1–3   → recall@7 ≈ [0.4–0.6], поиск мгновенный,
#                   но f1 1NN через Annoy падает на [1–3]%.
#   n_trees=9     → recall@7 ≈ [0.85–0.95], f1 ≈ точному 1NN,
#                   поиск — единицы мс.
#   n_trees ≥ 20  → recall → 1.0, но время построения индекса
#                   растёт линейно, выигрыша в качестве нет.
#   Оптимум для ~1500 точек — n_trees = 9.
#
# Вклад признаков и кодирования:
#   Топ по |coef| у LogisticRegression: numDeadRelations (отриц.),
#   isAliveSpouse / isAliveHeir / isAliveFather (положит.),
#   popularity_log (положит.), culture_le.
#   Удачные решения по кодированию:
#     - log10(popularity·50 + 1) — сгладил тяжёлый хвост;
#     - top-10 + other для house — компромисс между LabelEncoder
#       (ложный порядок) и полным OneHot (рост размерности);
#     - is_age_known как индикатор пропуска — сам факт
#       "возраст неизвестен" несёт информацию о персонаже;
#     - StandardScaler обязателен — без него признаки
#       с большой дисперсией (age, popularity) доминируют.
#
# Когда непараметрические выигрывают/проигрывают:
#   Выигрывают: мало данных (<500), сложная локальная структура
#   классов (кластеры, "острова"), разнородные признаки,
#   много шумовых признаков.
#   Проигрывают: много коррелирующих признаков (L2 стабилизирует
#   веса лучше), почти линейная разделимость, требование
#   интерпретируемости, много зашумлённых данных.
#   В нашем случае LogisticRegression [выиграла / почти догнала]
#   kNN — данные GoT близки к линейно разделимым по 31 признаку.
#
# Возникшие сложности и решения:
#   1) Annoy не поддерживает Минковского с p=3 → использовали
#      euclidean как ближайшую доступную метрику.
#   2) Дисбаланс классов 78/22 → стратификация через stratify=y,
#      метрики — accuracy + f1.
#   3) Пропуски в age (82%) и culture (69%) → заполнение медианой
#      + индикатор is_age_known для age; "unknown" + LabelEncoder
#      для culture (лучше, чем терять 70% строк).
#   4) Незнакомые категории в test.csv → culture_le = -1,
#      one-hot выравнивается через reindex(columns=train.columns).
#   5) Свой kNN медленнее sklearn → векторизовали расстояние через
#      broadcasting, полную матрицу попарных расстояний считали
#      один раз для LOO.
#   6) LOO по 1089 объектам × 19 значений k — долго (~1–2 мин),
#      но требование задания — оставили как есть.
#
