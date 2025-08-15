# Пути к данным
TRAIN_DATA_PATH = "data/raw/train.csv"
TEST_DATA_PATH = "data/raw/test.csv"

# Путь для сохранения моделей и артефактов
MODEL_DIR = "models/"
MODEL_PATH = f"{MODEL_DIR}model.joblib"
VECTORIZER_PATH = f"{MODEL_DIR}vectorizer.joblib"

# Путь для сохранения предсказаний
SUBMISSION_PATH = "submission.csv"

# Параметры модели
RANDOM_STATE = 42
TEST_SIZE = 0.2