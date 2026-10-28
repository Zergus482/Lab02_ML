# Лабораторная работа №2 — Метрические методы классификации

**Студент:** Степанов Сергей  
**Группа:** Б.ПИН.ИИ.24.16
**Вариант:** 15  
**Дисциплина:** Методы и алгоритмы машинного обучения  
**Дата:** 2026

## О чём работа

Задача бинарной классификации на датасете Game of Thrones: предсказать `isAlive`
(жив ли персонаж). Сравниваются несколько метрических методов и логистическая
регрессия:

- **kNN** — собственная реализация без sklearn, метрика Минковского (p=3),
  4 типа весов (uniform / distance / linear / exponential)
- **aNN (окно Парзена)** — ядро quartic, фиксированная и переменная ширина окна
- **Метод потенциальных функций** — формулировка Парзена через «потенциалы»
- **Annoy (ANN)** — приближённый поиск ближайших соседей
- **LogisticRegression** — L2-регуляризация, подбор C по LOO

Дополнительно: профиль компактности, жадный отбор эталонных объектов для 1NN.

## Параметры варианта 15

| Параметр | Значение |
|---|---|
| seed | 15 |
| k_max | 19 |
| h_max | 1.2 |
| Метрика kNN | minkowski (p=3) |
| Веса kNN | linear |
| Ядро Parzen | quartic |
| Popularity | log10(pop * 50 + 1) |
| Culture | без группировки (LabelEncoder) |
| House | top-10 + other (OneHot) |
| n_trees (Annoy) | 9 |
| n_neighbors_ann | 7 |
| query_character | Brienne of Tarth |

## Структура проекта

```
lab_2/
├── config.py           # все гиперпараметры варианта
├── utils.py            # сохранение/загрузка артефактов
├── data_prep.py        # Часть 1: препроцессинг + сплит 70/15/15
├── knn.py              # Часть 2: свой kNN + LOO + сравнение весов
├── parzen.py           # Часть 3: окно Парзена (quartic)
├── compactness.py      # Часть 4: профиль компактности + отбор эталонов
├── potential.py        # Часть 5: метод потенциальных функций
├── ann_search.py       # Часть 6: Annoy + recall@k
├── linear_models.py    # Часть 7: LogisticRegression L2 + LOO по C
├── submission.py       # Часть 8: финальный submission.csv + итоговый вывод
├── requirements.txt
├── artifacts/          # .pkl с результатами + графики (создаётся автоматически)
├── submission.csv      # итоговый файл с предсказаниями
└── README.md
```

## Установка

Нужен Python 3.12. Удобнее всего ставить через Homebrew (macOS).

```bash
# Python 3.12 (если ещё нет)
brew install python@3.12

# клонируем репозиторий
git clone https://github.com/[username]/[repo-name].git
cd [repo-name]

# создаём виртуальное окружение
python3.12 -m venv venv
source venv/bin/activate

# ставим зависимости
pip install -r requirements.txt
```

Если `annoy` не ставится — на macOS понадобится компилятор C++:

```bash
xcode-select --install
```

## Зависимости (`requirements.txt`)

```
numpy>=1.26
pandas>=2.1
matplotlib>=3.8
scikit-learn>=1.4
annoy>=1.17
```

Обновить под свою версию:

```bash
pip freeze > requirements.txt
```

## Как запускать

Скрипты запускаются **последовательно**, каждый читает артефакты предыдущего
из папки `artifacts/`. `main.py` нет специально — чтобы можно было
перезапускать отдельные части без полного пайплайна.

```bash
# Шаг 1. Препроцессинг и сплит (создаст artifacts/data.pkl)
python data_prep.py

# Шаг 2. kNN: LOO по k, сравнение весов (artifacts/knn_results.pkl)
python knn.py

# Шаг 3. Окно Парзена (artifacts/parzen_results.pkl)
python parzen.py

# Шаг 4. Профиль компактности + жадный отбор
python compactness.py

# Шаг 5. Метод потенциальных функций
python potential.py

# Шаг 6. Annoy (приближённый поиск)
python ann_search.py

# Шаг 7. LogisticRegression
python linear_models.py

# Шаг 8. Финальный submission.csv
python submission.py
```

После последнего шага в корне появится `submission.csv` с колонками
`name, isAlive` — это ответ для сдачи.

**Время выполнения** (на MacBook Air M1, для справки):

| Скрипт | Время |
|---|---|
| data_prep.py | ~5 сек |
| knn.py | ~2–3 мин (LOO по 19 значениям k) |
| parzen.py | ~1.5 мин |
| compactness.py | ~40 сек (из них t-SNE ~30 сек) |
| potential.py | ~30 сек |
| ann_search.py | ~10 сек |
| linear_models.py | ~2 мин (LOO по 6 значениям C) |
| submission.py | ~30 сек |

Общее время полного прогона — **~7 минут**.

## Что появляется в `artifacts/`

После полного прогона там будут:

### Артефакты (`.pkl`)

- `data.pkl` — подготовленные данные (X_train/val/test, y, имена)
- `knn_results.pkl` — лучший k, accuracy/f1 по весам
- `parzen_results.pkl` — лучший h и k, метрики
- `compactness_results.pkl` — маска эталонов, история отбора
- `potential_results.pkl` — лучший h, сравнение методов
- `ann_results.pkl` — recall, времена, f1 через Annoy
- `linear_results.pkl` — лучший C, коэффициенты, сравнение

### Графики (`.png`)

- `knn_loo.png` — LOO accuracy vs k
- `parzen_loo_h.png` — LOO accuracy vs h (fixed window)
- `parzen_loo_k.png` — LOO accuracy vs k (variable window)
- `parzen_moons.png` — решающие границы на make_moons
- `compactness_profile.png` — кривая Πᵢ
- `compactness_size_vs_acc.png` — размер Ω vs accuracy
- `compactness_pca.png` — PCA: до/после отбора
- `compactness_tsne.png` — t-SNE: до/после отбора
- `potential_h_influence.png` — accuracy/f1 vs h
- `potential_comparison.png` — сравнение 4 методов
- `ann_recall_vs_trees.png` — recall@7 vs n_trees
- `ann_time_vs_trees.png` — время поиска vs n_trees
- `logreg_loo_vs_C.png` — LOO accuracy vs C
- `linear_comparison.png` — сравнение всех методов

Все графики используются в отчёте.

## Основные результаты

### Метрики на валидационной выборке (234 объекта)

| Метод | accuracy | f1_score | Примечание |
|---|---|---|---|
| kNN (uniform) | 0.8205 | 0.8945 | k=14 |
| kNN (distance) | 0.8205 | 0.8906 | k=14 |
| **kNN (linear)** | **0.8333** | **0.9003** | **k=14 — лучшая** |
| kNN (exponential) | 0.8120 | 0.8817 | k=14 |
| aNN (fixed h) | 0.7650 | 0.8601 | h=1.160 |
| aNN (variable k) | 0.8162 | 0.8877 | k=20 |
| Potential | 0.7863 | 0.8768 | h=0.050 |
| LogisticRegression (L2) | 0.8291 | 0.8974 | C=100 |

### Лучшая модель

**kNN с весами `linear`, k=14** — accuracy = 0.8333, f1 = 0.9003.

Она переобучена на полном train (1557 объектов) и применена к тестовой выборке.
`submission.csv` содержит 389 строк, доля предсказанных «живых» — **0.802**
(сравнимо с 0.78 в train, значит модель не сваливается в один класс).

### Интересные наблюдения

- Среди весов kNN лучший — `linear` (0.833), хуже — `exponential` (0.812).
  Равномерное (`uniform`) и обратное расстоянию (`distance`) работают одинаково
  по accuracy, но `uniform` даёт чуть выше f1.
- `aNN (fixed h)` при h=1.160 работает заметно хуже остальных — при большом `h`
  финитное ядро сглаживает всё до мажоритарного класса.
- `LogisticRegression` с C=100 (то есть почти без регуляризации) даёт результат
  всего на 0.4% хуже лучшего kNN. Значит данные близки к линейно разделимым.
- Метод потенциальных функций с малым `h=0.05` проигрывает — при таком узком
  окне quartic обнуляет почти все веса, и модель предсказывает мажоритарный класс.

### Annoy (Часть 6)

- `recall@7 ≈ 0.86` при `n_trees = 9`
- Время поиска — единицы миллисекунд
- f1 для 1NN через Annoy ≈ точному 1NN (разница < 1%)
- При `n_trees < 3` recall падает до ~0.5, при `n_trees ≥ 20` — растёт до 1.0,
  но время построения индекса увеличивается линейно

## Ограничения и сложности

- **Annoy не поддерживает Минковского с p=3** — в библиотеке доступны только
  `angular`, `euclidean`, `manhattan`, `hamming`, `dot`. В Части 6 использован
  `euclidean` как ближайшая доступная метрика. Это ограничение библиотеки,
  не ошибка реализации.
- **Дисбаланс классов** 78/22 (alive/dead). Accuracy завышена — модель,
  всегда предсказывающая «жив», даёт 0.78. Поэтому ориентировались
  в первую очередь на f1.
- **Пропуски**: `age` — 82%, `culture` — 69%. Решение: заполнение медианой
  + индикатор `is_age_known`; строка `"unknown"` + LabelEncoder для `culture`.
- **Незнакомые категории в test** (`culture`, `house`): для `culture` маппинг
  в `-1`, для one-hot-колонок `house` — выравнивание через
  `reindex(columns=train.columns)`.

## Воспроизводимость

Все случайные операции (сплит, `AnnoyIndex`, `t-SNE`, `LogisticRegression`)
инициализируются `random_state = SEED = 15`. При повторном запуске результаты
полностью совпадают.

Проверить можно так:

```bash
python data_prep.py && python knn.py
```

Дважды подряд — вывод должен быть идентичным.

## Ссылки

- Train: https://raw.githubusercontent.com/EvADarya/Data_processing-DLS-/refs/heads/main/game_of_thrones_train.csv
- Test:  https://raw.githubusercontent.com/EvADarya/Data_processing-DLS-/refs/heads/main/game_of_thrones_test.csv

Датасеты скачиваются автоматически в `data_prep.py` по `pd.read_csv(URL)`.


