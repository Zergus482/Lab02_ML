
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, f1_score

from config import SEED, H_MAX, KNN_P
from utils import load_data, save_artifact

np.random.seed(SEED)

N_H = 30
H_GRID = np.linspace(0.05, H_MAX, N_H)


#  Ядро quartic
def kernel_quartic(r):
    """
    K(r) = (1 - r^2)^2 при |r| <= 1, иначе 0.
    """
    r = np.asarray(r, dtype=float)
    mask = np.abs(r) <= 1
    out = np.zeros_like(r)
    out[mask] = (1 - r[mask] ** 2) ** 2
    return out


#  Метрика Минковского
def minkowski_distance(X, x, p=KNN_P):
    diff = np.abs(X - x)
    return (diff ** p).sum(axis=1) ** (1.0 / p)


#  Классификатор Парзена
def parzen_classify(X_train, y_train, X_query, h=None,
                    mode="fixed", k=None):
    """
    mode = "fixed"    -> h - константа для всех точек
    mode = "variable" -> для каждой точки h = расстояние до k-го соседа
    """
    preds = np.zeros(len(X_query), dtype=int)

    for i, x in enumerate(X_query):
        dists = minkowski_distance(X_train, x)

        if mode == "fixed":
            h_i = h
        else:  # variable (aNN)
            sorted_d = np.sort(dists)
            h_i = sorted_d[k - 1] if k - 1 < len(sorted_d) else sorted_d[-1]
            if h_i <= 0:
                h_i = 1e-6

        # веса соседей: K(d / h)
        w = kernel_quartic(dists / h_i)

        score_0 = w[y_train == 0].sum()
        score_1 = w[y_train == 1].sum()

        preds[i] = 1 if score_1 >= score_0 else 0

    return preds


# 4. LOO для fixed h
def loo_parzen_fixed(X, y, h_grid):
    """Для каждого h из h_grid считаем LOO accuracy."""
    n = len(X)
    accs = np.zeros(len(h_grid))

    # заранее посчитаем полную матрицу попарных расстояний (n x n)
    diff = np.abs(X[:, None, :] - X[None, :, :])
    D = (diff ** KNN_P).sum(axis=2) ** (1.0 / KNN_P)

    for j, h in enumerate(h_grid):
        correct = 0
        for i in range(n):
            dists = np.delete(D[i], i)   # исключаем сам объект
            y_tr = np.delete(y, i)

            w = kernel_quartic(dists / h)
            s0 = w[y_tr == 0].sum()
            s1 = w[y_tr == 1].sum()
            pred = 1 if s1 >= s0 else 0
            if pred == y[i]:
                correct += 1
        accs[j] = correct / n

    return accs


#  LOO для variable h (перебор k)
def loo_parzen_variable(X, y, k_grid):
    n = len(X)
    accs = np.zeros(len(k_grid))

    diff = np.abs(X[:, None, :] - X[None, :, :])
    D = (diff ** KNN_P).sum(axis=2) ** (1.0 / KNN_P)

    for j, k in enumerate(k_grid):
        correct = 0
        for i in range(n):
            dists = np.delete(D[i], i)
            y_tr = np.delete(y, i)

            sorted_d = np.sort(dists)
            h_i = sorted_d[k - 1] if k - 1 < len(sorted_d) else sorted_d[-1]
            if h_i <= 0:
                h_i = 1e-6

            w = kernel_quartic(dists / h_i)
            s0 = w[y_tr == 0].sum()
            s1 = w[y_tr == 1].sum()
            pred = 1 if s1 >= s0 else 0
            if pred == y[i]:
                correct += 1
        accs[j] = correct / n

    return accs


# 6. Визуализация решающих границ на make_moons
def plot_decision_boundaries():
    X, y = make_moons(n_samples=300, noise=0.25, random_state=SEED)
    X = X.astype(float)

    h_values = [0.1, 0.3, 0.8]

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 120),
                         np.linspace(y_min, y_max, 120))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, h in zip(axes, h_values):
        preds_grid = parzen_classify(X, y, grid, h=h, mode="fixed")
        preds_grid = preds_grid.reshape(xx.shape)

        ax.contourf(xx, yy, preds_grid, alpha=0.4, cmap="coolwarm")
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", s=15, label="class 0")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c="red", s=15, label="class 1")
        ax.set_title(f"h = {h}")
        ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("Решающие границы Parzen (quartic) на make_moons")
    plt.tight_layout()
    plt.savefig("artifacts/parzen_moons.png", dpi=120)
    plt.close()
    print("  сохранено в artifacts/parzen_moons.png")


# 7. main
def main():
    data = load_data()
    X_train = data["X_train"].astype(float)
    y_train = data["y_train"].astype(int)
    X_val   = data["X_val"].astype(float)
    y_val   = data["y_val"].astype(int)

    print("Часть 3. Parzen (вариант 15: quartic, h_max = %.2f)" % H_MAX)
    print("=" * 60)
    print("train:", X_train.shape, " val:", X_val.shape)

    # LOO для fixed h 
    print("\n[1] LOO-подбор h (fixed window) ...")
    t0 = time.time()
    loo_fixed = loo_parzen_fixed(X_train, y_train, H_GRID)
    print("    время: %.1f сек" % (time.time() - t0))

    best_idx = int(np.argmax(loo_fixed))
    best_h = float(H_GRID[best_idx])
    print("    лучший h = %.4f (LOO acc = %.4f)" % (best_h, loo_fixed[best_idx]))

    plt.figure(figsize=(8, 4))
    plt.plot(H_GRID, loo_fixed, marker="o")
    plt.axvline(best_h, color="red", linestyle="--", label=f"best h = {best_h:.3f}")
    plt.xlabel("h")
    plt.ylabel("LOO accuracy")
    plt.title("Parzen (quartic): LOO accuracy vs h")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/parzen_loo_h.png", dpi=120)
    plt.close()
    print("    сохранил artifacts/parzen_loo_h.png")

    # LOO для variable h (aNN)
    print("\n[2] LOO-подбор k (variable window / aNN) ...")
    k_grid = np.arange(1, 21)
    t0 = time.time()
    loo_var = loo_parzen_variable(X_train, y_train, k_grid)
    print("    время: %.1f сек" % (time.time() - t0))

    best_k_idx = int(np.argmax(loo_var))
    best_k = int(k_grid[best_k_idx])
    print("    лучший k = %d (LOO acc = %.4f)" % (best_k, loo_var[best_k_idx]))

    plt.figure(figsize=(8, 4))
    plt.plot(k_grid, loo_var, marker="o")
    plt.axvline(best_k, color="red", linestyle="--", label=f"best k = {best_k}")
    plt.xlabel("k")
    plt.ylabel("LOO accuracy")
    plt.title("Parzen (quartic): LOO accuracy vs k (variable window)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/parzen_loo_k.png", dpi=120)
    plt.close()
    print("    сохранил artifacts/parzen_loo_k.png")

    #  Метрики на валидации 
    print("\n[3] Метрики на валидации")

    preds_fixed = parzen_classify(X_train, y_train, X_val,
                                  h=best_h, mode="fixed")
    acc_fixed = accuracy_score(y_val, preds_fixed)
    f1_fixed = f1_score(y_val, preds_fixed, average="binary")
    print("    fixed h=%.3f     acc=%.4f  f1=%.4f" % (best_h, acc_fixed, f1_fixed))

    preds_var = parzen_classify(X_train, y_train, X_val,
                                mode="variable", k=best_k)
    acc_var = accuracy_score(y_val, preds_var)
    f1_var = f1_score(y_val, preds_var, average="binary")
    print("    variable k=%d    acc=%.4f  f1=%.4f" % (best_k, acc_var, f1_var))

    # Решающие границы
    print("\n[4] Решающие границы на make_moons ...")
    plot_decision_boundaries()

    #  Сохраняем 
    results = {
        "best_h": best_h,
        "best_k": best_k,
        "loo_fixed": loo_fixed,
        "loo_var": loo_var,
        "h_grid": H_GRID,
        "k_grid": k_grid,
        "acc_fixed": acc_fixed,
        "f1_fixed": f1_fixed,
        "acc_var": acc_var,
        "f1_var": f1_var,
    }
    save_artifact("parzen_results", results)
    print("\nСохранено в  artifacts/parzen_results.pkl")


if __name__ == "__main__":
    main()