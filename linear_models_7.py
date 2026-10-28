
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, f1_score

from config import SEED
from utils import load_data, save_artifact, load_artifact

# sklearn ругается на сходимость - прячем предупреждения(подсказала нейронка)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

np.random.seed(SEED)

# 1. LOO-подбор gamma (= 1/C) для LogisticRegression
def loo_logreg(X, y, C_grid, penalty="l2"):
    """
    Для каждого C из C_grid считаем LOO-accuracy.
    C = 1/gamma - так в sklearn; в отчёте можно сказать,
    что gamma = 1/C.

    Важно: LOO по 1089 фолдам * len(C_grid) значений - это долго.
    Для ускорения можно уменьшить C_grid (например, до 6-8 значений).
    """
    loo = LeaveOneOut()
    accs = np.zeros(len(C_grid))

    for j, C in enumerate(C_grid):
        t0 = time.time()
        preds = np.zeros_like(y, dtype=int)

        for train_idx, test_idx in loo.split(X):
            if len(np.unique(y[train_idx])) < 2:
                preds[test_idx] = y[train_idx][0]
                continue

            model = LogisticRegression(
                C=C, penalty=penalty, solver="liblinear",
                max_iter=1000, random_state=SEED
            )
            model.fit(X[train_idx], y[train_idx])
            preds[test_idx] = model.predict(X[test_idx])

        acc = (preds == y).mean()
        accs[j] = acc
        print("    C=%.4f (gamma=%.2f): LOO acc=%.4f   [%.1f сек]"
              % (C, 1 / C, acc, time.time() - t0))

    return accs


# 2. main
def main():
    data = load_data()
    X_train = data["X_train"].astype(float)
    y_train = data["y_train"].astype(int)
    X_val   = data["X_val"].astype(float)
    y_val   = data["y_val"].astype(int)

    print("Часть 7. LogisticRegression (L2)")
    print("train:", X_train.shape, " val:", X_val.shape)

    # 7.1. Сетка C (C = 1/gamma)
    C_grid = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])

    print("\n[1] LOO-подбор C = 1/gamma ...")
    t0 = time.time()
    loo_accs = loo_logreg(X_train, y_train, C_grid)
    print("    общее время: %.1f сек" % (time.time() - t0))

    best_idx = int(np.argmax(loo_accs))
    best_C = float(C_grid[best_idx])
    best_gamma = 1.0 / best_C
    print("\n    лучший C = %.4f (gamma = %.4f)  LOO acc = %.4f"
          % (best_C, best_gamma, loo_accs[best_idx]))

    # 7.2. График LOO vs C 
    plt.figure(figsize=(8, 4))
    plt.plot(C_grid, loo_accs, marker="o")
    plt.axvline(best_C, color="red", linestyle="--",
                label=f"best C = {best_C:.4f}")
    plt.xscale("log")
    plt.xlabel("C (обратный gamma)")
    plt.ylabel("LOO accuracy")
    plt.title("LogisticRegression (L2): LOO accuracy vs C")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/logreg_loo_vs_C.png", dpi=120)
    plt.close()
    print("    сохранено в artifacts/logreg_loo_vs_C.png")

    #  Финальная модель на всём train
    print("\n[2] Обучение финальной модели на всём train...")
    final_model = LogisticRegression(
        C=best_C, penalty="l2", solver="liblinear",
        max_iter=1000, random_state=SEED
    )
    final_model.fit(X_train, y_train)

    preds_val = final_model.predict(X_val)
    acc_logreg = accuracy_score(y_val, preds_val)
    f1_logreg = f1_score(y_val, preds_val, average="binary")
    print("    LogisticRegression (L2, C=%.4f):" % best_C)
    print("    accuracy = %.4f   f1_score = %.4f" % (acc_logreg, f1_logreg))

    # коэффициенты модели 
    coefs = final_model.coef_[0]
    feature_names = data["feature_names"]
    top_idx = np.argsort(np.abs(coefs))[::-1][:10]
    print("\n    Топ-10 самых влиятельных признаков (по |coef|):")
    for i in top_idx:
        print("      %-25s coef = %+.4f" % (feature_names[i], coefs[i]))

    #  Сравнение с непараметрическими методами
    print("\n[3] Сравнение с непараметрическими методами...")
    comparison = {
        "LogisticRegression": {"acc": acc_logreg, "f1": f1_logreg,
                                "note": f"L2, C={best_C:.3f}"},
    }

    # kNN
    try:
        knn_res = load_artifact("knn_results")
        wr = knn_res["weight_results"]
        best_w = max(wr, key=lambda w: wr[w]["acc"])
        comparison["kNN"] = {
            "acc": wr[best_w]["acc"],
            "f1": wr[best_w]["f1"],
            "note": f"k={knn_res['best_k']}, {best_w}",
        }
    except FileNotFoundError:
        print("    [warn] knn_results.pkl не найден")

    # aNN
    try:
        parz_res = load_artifact("parzen_results")
        comparison["aNN (fixed)"] = {
            "acc": parz_res["acc_fixed"],
            "f1": parz_res["f1_fixed"],
            "note": f"h={parz_res['best_h']:.3f}",
        }
        comparison["aNN (variable)"] = {
            "acc": parz_res["acc_var"],
            "f1": parz_res["f1_var"],
            "note": f"k={parz_res['best_k']}",
        }
    except FileNotFoundError:
        print("    [warn] parzen_results.pkl не найден")

    # potential
    try:
        pot_res = load_artifact("potential_results")
        comparison["Potential"] = {
            "acc": pot_res["acc_best"],
            "f1": pot_res["f1_best"],
            "note": f"h={pot_res['best_h']:.3f}",
        }
    except FileNotFoundError:
        print("    [warn] potential_results.pkl не найден")

    # печатаем сводную таблицу
    print("\n    %-22s %-10s %-10s %s" % ("метод", "accuracy", "f1", "note"))
    print("    " + "-" * 65)
    for name, m in comparison.items():
        note = m.get("note", "")
        print("    %-22s %.4f     %.4f     %s" % (name, m["acc"], m["f1"], note))

    # 7.5. Столбчатая диаграмма сравнения
    methods = list(comparison.keys())
    accs_cmp = [comparison[m]["acc"] for m in methods]
    f1_cmp = [comparison[m]["f1"] for m in methods]

    x_pos = np.arange(len(methods))
    width = 0.38

    plt.figure(figsize=(11, 5))
    plt.bar(x_pos - width / 2, accs_cmp, width, label="accuracy")
    plt.bar(x_pos + width / 2, f1_cmp, width, label="f1_score")
    plt.xticks(x_pos, methods, rotation=20, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("метрика на валидации")
    plt.title("Сравнение: LogisticRegression vs непараметрические методы")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("artifacts/linear_comparison.png", dpi=120)
    plt.close()
    print("\n    сохранил artifacts/linear_comparison.png")

    # 7.6. Сохраняем результаты 
    results = {
        "best_C": best_C,
        "best_gamma": best_gamma,
        "loo_accs": loo_accs.tolist(),
        "C_grid": C_grid.tolist(),
        "acc_logreg": acc_logreg,
        "f1_logreg": f1_logreg,
        "comparison": comparison,
        "top_features": [(feature_names[i], float(coefs[i])) for i in top_idx],
    }
    save_artifact("linear_results", results)
    print("\nСохранено в artifacts/linear_results.pkl")


if __name__ == "__main__":
    main()