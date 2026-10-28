# Параметры варианта 15

SEED = 15

# kNN
K_MAX = 19
KNN_P = 3                     # метрика Минковского с p=3

# aNN (окно Парзена)
H_MAX = 1.2

# препроцессинг
POPULARITY_M = 50             # log10(popularity * 50 + 1)
HOUSE_TOP_K = 10              # top-10 домов + other/unknown

# Annoy
N_TREES = 9
N_NEIGHBORS_ANN = 7
QUERY_CHARACTER = "Brienne of Tarth"

# ссылки на данные
TRAIN_URL = ("https://raw.githubusercontent.com/EvADarya/"
             "Data_processing-DLS-/refs/heads/main/game_of_thrones_train.csv")
TEST_URL  = ("https://raw.githubusercontent.com/EvADarya/"
             "Data_processing-DLS-/refs/heads/main/game_of_thrones_test.csv")