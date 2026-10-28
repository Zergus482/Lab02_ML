
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score

from config import SEED, KNN_P
from utils import load_data, save_artifact

np.random.seed(SEED)



#  Метрика (Минковский p=3)

def pairwise_distances(X):
    """Полная матрица расстояний (n x n)."""
    diff = np.abs(X[:, None, :] - X[None, :, :])
    return (diff ** KNN_P).sum(axis=2) ** (1.0 / KNN_P)


# Профиль компактности Pi_i(m)
def compactness_profile(X, y, m_max=10):
    """
    Для каждого объекта i и каждого m = 1..m_max:
        Pi_i(m) = (1/m) * сумма расстояний до m ближайших
                  соседей ТОГО ЖЕ класса.
    Возвращает матрицу (n, m_max).
    """
    n = len(X)
    D = pairwise_distances(X)
    np.fill_diagonal(D, np.inf)

    profile = np.zeros((n, m_max))

    for i in range(n):
        # маска "того же класса, кроме самого i"
        same_class = (y == y[i])
        same_class[i] = False

        # расстояния только до соседей своего класса
        d_same = D[i][same_class]

        # сортируем и берём первые m_max
        d_sorted = np.sort(d_same)

        # накапливаем сумму и делим на m
        cum = np.cumsum(d_sorted[:m_max])
        profile[i] = cum / np.arange(1, m_max + 1)

    return profile

# 3. LOO 1NN предсказания
def loo_1nn_predict(X, y):
    """
    Для каждого объекта возвращаем предсказание 1NN по остальным.
    """
    D = pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    nn_idx = np.argmin(D, axis=1)
    return y[nn_idx]


def loo_1nn_acc(X, y):
    preds = loo_1nn_predict(X, y)
    return (preds == y).mean()

# 4. Жадное удаление эталонных объектов
def greedy_remove(X, y, remove_frac=0.25, min_size=80, max_iter=40):
    """
    На каждой итерации:
      - считаем LOO 1NN предсказания
      - отбираем те, что классифицируются правильно
      - случайно удаляем часть из них (remove_frac)
      - если accuracy на новой выборке сильно падает - откат + стоп

    Возвращает маску "оставить" и историю (size, acc).
    """
    n = len(X)
    alive = np.ones(n, dtype=bool)
    history = []

    rng = np.random.default_rng(SEED)

    acc = loo_1nn_acc(X[alive], y[alive])
    history.append((int(alive.sum()), float(acc)))
    print("  старт: size = %d, acc = %.4f" % (alive.sum(), acc))

    for it in range(max_iter):
        if alive.sum() <= min_size:
            break

        X_cur = X[alive]
        y_cur = y[alive]
        preds = loo_1nn_predict(X_cur, y_cur)
        correct = (preds == y_cur)

        idx_correct_local = np.where(correct)[0]
        if len(idx_correct_local) == 0:
            break

        # сколько удалить
        n_remove = max(1, int(remove_frac * len(idx_correct_local)))
        remove_local = rng.choice(idx_correct_local, size=n_remove, replace=False)

        # мапим локальные индексы в глобальные
        alive_indices = np.where(alive)[0]
        remove_global = alive_indices[remove_local]

        # сохраняем состояние на случай отката
        alive_backup = alive.copy()

        alive[remove_global] = False

        new_acc = loo_1nn_acc(X[alive], y[alive])
        size = int(alive.sum())

        # если accuracy не упала больше чем на 1 процентный пункт - принимаем
        if new_acc >= acc - 0.01:
            acc = new_acc
            history.append((size, float(new_acc)))
            print("  итерация %2d: size = %4d, acc = %.4f" % (it + 1, size, new_acc))
        else:
            # откатываем и останавливаемся
            alive = alive_backup
            print("  итерация %2d: откат (acc упала до %.4f)" % (it + 1, new_acc))
            break

    history.append((int(alive.sum()), float(acc)))
    return alive, history


# 5. Визуализация PCA: эталоны vs удалённые
def plot_pca_before_after(X, y, alive):
    pca = PCA(n_components=2, random_state=SEED)
    X2 = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # до 
    ax = axes[0]
    for cls, color in [(0, "tab:blue"), (1, "tab:red")]:
        m = (y == cls)
        ax.scatter(X2[m, 0], X2[m, 1], c=color, s=8, alpha=0.5, label=f"class {cls}")
    ax.set_title(f"До отбора (n = {len(X)})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()

    # после 
    ax = axes[1]
    # удалённые - серые маленькие
    ax.scatter(X2[~alive, 0], X2[~alive, 1], c="lightgray", s=6, alpha=0.5,
               label="удалённые")
    # оставшиеся
    for cls, color in [(0, "tab:blue"), (1, "tab:red")]:
        m = (y == cls) & alive
        ax.scatter(X2[m, 0], X2[m, 1], c=color, s=20, label=f"эталон class {cls}")

    ax.set_title(f"После отбора (n = {alive.sum()})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()

    plt.tight_layout()
    plt.savefig("artifacts/compactness_pca.png", dpi=120)
    plt.close()
    print("  сохранено в  artifacts/compactness_pca.png")


# 6. t-SNE (бонус)
def plot_tsne_before_after(X, y, alive):
    print("  считаю t-SNE (это ~20-40 сек) ...")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, init="pca")
    X2 = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for cls, color in [(0, "tab:blue"), (1, "tab:red")]:
        m = (y == cls)
        ax.scatter(X2[m, 0], X2[m, 1], c=color, s=8, alpha=0.5, label=f"class {cls}")
    ax.set_title(f"t-SNE: до отбора (n = {len(X)})")
    ax.legend()

    ax = axes[1]
    ax.scatter(X2[~alive, 0], X2[~alive, 1], c="lightgray", s=6, alpha=0.5,
               label="удалённые")
    for cls, color in [(0, "tab:blue"), (1, "tab:red")]:
        m = (y == cls) & alive
        ax.scatter(X2[m, 0], X2[m, 1], c=color, s=20, label=f"эталон class {cls}")
    ax.set_title(f"t-SNE: после отбора (n = {alive.sum()})")
    ax.legend()

    plt.tight_layout()
    plt.savefig("artifacts/compactness_tsne.png", dpi=120)
    plt.close()
    print("  сохранено в artifacts/compactness_tsne.png")


# 7. main
def main():
    data = load_data()
    X_train = data["X_train"].astype(float)
    y_train = data["y_train"].astype(int)
    X_val   = data["X_val"].astype(float)
    y_val   = data["y_val"].astype(int)

    print("Часть 4. Профиль компактности + отбор эталонов")
    print("=" * 60)
    print("train:", X_train.shape, " val:", X_val.shape)

    #  Профиль компактности 
    print("\n[1] Считаю профиль компактности Pi(m), m=1..10 ...")
    t0 = time.time()
    profile = compactness_profile(X_train, y_train, m_max=10)
    print("    время: %.1f сек" % (time.time() - t0))

    # сортируем объекты по Pi(10) - итоговое среднее расстояние
    pi_final = profile[:, -1]                      # (n,)
    order = np.argsort(pi_final)

    plt.figure(figsize=(9, 4))
    plt.plot(pi_final[order], linewidth=1.5)
    plt.xlabel("объекты (отсортированы по Pi)")
    plt.ylabel("Pi(10) - среднее расст. до 10 своих соседей")
    plt.title("Профиль компактности (по убыванию информативности)")
    plt.axhline(np.percentile(pi_final, 95), color="red", linestyle="--",
                label="95-й процентиль")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/compactness_profile.png", dpi=120)
    plt.close()
    print("  сохранил artifacts/compactness_profile.png")

    # сколько объектов "выбиваются" (выше 95-го процентиля)
    n_outliers = int((pi_final > np.percentile(pi_final, 95)).sum())
    print("  объектов выше 95-го процентиля Pi: %d из %d" % (n_outliers, len(X_train)))

    #  Жадное удаление эталонов 
    print("\n[2] Жадное удаление эталонных объектов ...")
    t0 = time.time()
    alive, history = greedy_remove(X_train, y_train,
                                   remove_frac=0.25,
                                   min_size=80, max_iter=40)
    print("    время: %.1f сек" % (time.time() - t0))

    X_ref = X_train[alive]
    y_ref = y_train[alive]
    print("\n  итоговый размер эталонной выборки: %d из %d (%.1f%%)"
          % (alive.sum(), len(X_train), 100 * alive.mean()))

    #  График size vs accuracy 
    sizes = [h[0] for h in history]
    accs = [h[1] for h in history]

    plt.figure(figsize=(8, 4))
    plt.plot(sizes, accs, marker="o", linewidth=1.5)
    plt.xlabel("размер эталонной выборки Omega")
    plt.ylabel("LOO 1NN accuracy")
    plt.title("Зависимость LOO-точности от размера эталонной выборки")
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig("artifacts/compactness_size_vs_acc.png", dpi=120)
    plt.close()
    print("  сохранено в  artifacts/compactness_size_vs_acc.png")

    # Проверка accuracy на val 
    # 1NN на полной выборке
    preds_full = np.array([
        y_train[np.argmin(((X_train - x) ** 2).sum(axis=1) ** 0.5)]
        for x in X_val
    ])
    acc_full = accuracy_score(y_val, preds_full)

    # 1NN на эталонной выборке
    preds_ref = np.array([
        y_ref[np.argmin(((X_ref - x) ** 2).sum(axis=1) ** 0.5)]
        for x in X_val
    ])
    acc_ref = accuracy_score(y_val, preds_ref)

    print("\n[3] Метрики на валидации (1NN)")
    print("    полная выборка   : acc = %.4f" % acc_full)
    print("    эталонная выборка: acc = %.4f (размер %d)" % (acc_ref, len(X_ref)))

    # Визуализация 
    print("\n[4] Визуализация PCA ...")
    plot_pca_before_after(X_train, y_train, alive)

    # t-SNE - опционально, закомментируй если не хочешь ждать
    print("\n[5] Визуализация t-SNE ...")
    plot_tsne_before_after(X_train, y_train, alive)

    # Сохраняем результаты 
    results = {
        "alive_mask": alive,
        "history": history,
        "pi_final": pi_final,
        "n_outliers_95": n_outliers,
        "acc_full_val": acc_full,
        "acc_ref_val": acc_ref,
        "size_full": len(X_train),
        "size_ref": int(alive.sum()),
    }
    save_artifact("compactness_results", results)
    print("\nСохранено в artifacts/compactness_results.pkl")


if __name__ == "__main__":
    main()