
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from config import SEED, KNN_P, H_MAX
from utils import load_data, save_artifact, load_artifact

np.random.seed(SEED)

# 1. Ядро quartic
def kernel_quartic(r):
    """K(r) = (1 - r^2)^2 при |r| <= 1, иначе 0."""
    r = np.asarray(r, dtype=float)
    mask = np.abs(r) <= 1
    out = np.zeros_like(r)
    out[mask] = (1 - r[mask] ** 2) ** 2
    return out

# 2. Метрика (Минковский p=3)
def minkowski_distance(X, x, p=KNN_P):
    diff = np.abs(X - x)
    return (diff ** p).sum(axis=1) ** (1.0 / p)

# 3. Метод потенциальных функций
def potential_classify(X_train, y_train, X_query, h):
    """
    Для каждого объекта x из X_query:
      U_0(x) = сумма K(d(x, x_j)/h) по всем x_j класса 0
      U_1(x) = сумма K(d(x, x_j)/h) по всем x_j класса 1
      предсказание = argmax(U_0, U_1)
    """
    preds = np.zeros(len(X_query), dtype=int)

    for i, x in enumerate(X_query):
        dists = minkowski_distance(X_train, x)
        # потенциал каждой обучающей точки
        potentials = kernel_quartic(dists / h)

        # суммарный потенциал по классам
        U0 = potentials[y_train == 0].sum()
        U1 = potentials[y_train == 1].sum()

        preds[i] = 1 if U1 >= U0 else 0

    return preds


# 4. Прогон по сетке h
def evaluate_h_grid(X_train, y_train, X_val, y_val, h_grid):
    """
    Возвращает массив accuracy и f1 для каждого h из h_grid на val.
    """
    accs = np.zeros(len(h_grid))
    f1s = np.zeros(len(h_grid))

    for j, h in enumerate(h_grid):
        preds = potential_classify(X_train, y_train, X_val, h=h)
        accs[j] = accuracy_score(y_val, preds)
        f1s[j] = f1_score(y_val, preds, average="binary")

    return accs, f1s


# 5. main
def main():
    data = load_data()
    X_train = data["X_train"].astype(float)
    y_train = data["y_train"].astype(int)
    X_val   = data["X_val"].astype(float)
    y_val   = data["y_val"].astype(int)

    print("Часть 5. Метод потенциальных функций (quartic, fixed h)")
    print("=" * 60)
    print("train:", X_train.shape, " val:", X_val.shape)

    #  Сетка h на валидации 
    print("\n[1] Прогон по сетке h ...")
    h_grid = np.linspace(0.05, H_MAX, 30)
    t0 = time.time()
    accs, f1s = evaluate_h_grid(X_train, y_train, X_val, y_val, h_grid)
    print("    время: %.1f сек" % (time.time() - t0))

    best_idx = int(np.argmax(accs))
    best_h = float(h_grid[best_idx])
    print("    лучший h (по val) = %.4f  acc = %.4f  f1 = %.4f"
          % (best_h, accs[best_idx], f1s[best_idx]))

    # График accuracy/f1 vs h 
    plt.figure(figsize=(9, 4.5))
    plt.plot(h_grid, accs, marker="o", label="accuracy")
    plt.plot(h_grid, f1s, marker="s", label="f1_score")
    plt.axvline(best_h, color="red", linestyle="--",
                label=f"best h = {best_h:.3f}")
    plt.xlabel("h")
    plt.ylabel("метрика на валидации")
    plt.title("Метод потенциальных функций (quartic): влияние h")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/potential_h_influence.png", dpi=120)
    plt.close()
    print("    сохранено в  artifacts/potential_h_influence.png")

    # Финальные метрики при лучшем h 
    print("\n[2] Финальные метрики (best h на val)")
    preds = potential_classify(X_train, y_train, X_val, h=best_h)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="binary")
    print("    potential (h=%.3f)  acc = %.4f  f1 = %.4f" % (best_h, acc, f1))

    # Сравнение с kNN и aNN 
    print("\n[3] Сравнение методов (метрики на val)")
    comparison = {}

    # свои результаты
    comparison["potential"] = {"acc": acc, "f1": f1}

    # kNN из сохранённого артефакта
    try:
        knn_res = load_artifact("knn_results")
        best_k = knn_res["best_k"]
        wr = knn_res["weight_results"]
        # лучший вес по accuracy
        best_w = max(wr, key=lambda w: wr[w]["acc"])
        comparison["kNN (best)"] = {
            "acc": wr[best_w]["acc"],
            "f1": wr[best_w]["f1"],
            "note": f"k={best_k}, weights={best_w}",
        }
        print("    kNN  : k=%d, weights=%s  acc=%.4f  f1=%.4f"
              % (best_k, best_w, wr[best_w]["acc"], wr[best_w]["f1"]))
    except FileNotFoundError:
        print("    [warn] knn_results.pkl не найден - пропускаем kNN")

    # aNN из сохранённого артефакта
    try:
        parz_res = load_artifact("parzen_results")
        comparison["aNN (fixed h)"] = {
            "acc": parz_res["acc_fixed"],
            "f1": parz_res["f1_fixed"],
            "note": f"h={parz_res['best_h']:.3f}",
        }
        comparison["aNN (variable k)"] = {
            "acc": parz_res["acc_var"],
            "f1": parz_res["f1_var"],
            "note": f"k={parz_res['best_k']}",
        }
        print("    aNN fixed   : h=%.3f  acc=%.4f  f1=%.4f"
              % (parz_res["best_h"], parz_res["acc_fixed"], parz_res["f1_fixed"]))
        print("    aNN variable: k=%d     acc=%.4f  f1=%.4f"
              % (parz_res["best_k"], parz_res["acc_var"], parz_res["f1_var"]))
    except FileNotFoundError:
        print("    [warn] parzen_results.pkl не найден - пропускаем aNN")

    # столбчатая диаграмма сравнения
    methods = [m for m in comparison if "note" in comparison[m]]
    acc_vals = [comparison[m]["acc"] for m in methods]
    f1_vals = [comparison[m]["f1"] for m in methods]

    x_pos = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(9, 4.5))
    plt.bar(x_pos - width / 2, acc_vals, width, label="accuracy")
    plt.bar(x_pos + width / 2, f1_vals, width, label="f1_score")
    plt.xticks(x_pos, methods, rotation=15)
    plt.ylim(0, 1.0)
    plt.ylabel("метрика на val")
    plt.title("Сравнение методов: kNN vs Parzen vs Potential (quartic)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("artifacts/potential_comparison.png", dpi=120)
    plt.close()
    print("    сохранено в artifacts/potential_comparison.png")

    results = {
        "best_h": best_h,
        "h_grid": h_grid,
        "accs": accs,
        "f1s": f1s,
        "acc_best": acc,
        "f1_best": f1,
        "comparison": comparison,
    }
    save_artifact("potential_results", results)
    print("\nСохранил artifacts/potential_results.pkl")


if __name__ == "__main__":
    main()
    