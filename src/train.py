import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

from preprocess import clean_text
import config

def train():
    """Основная функция для обучения модели."""
    print("Загрузка данных...")
    # Фейковые данные для примера
    # TODO: заменить на pd.read_csv(config.TRAIN_DATA_PATH)
    data = {
        'description': [
            'Оригинальные наушники SuperPods, гарантия 1 год',
            'Наушники SuprPods, реплика 1:1, лучшая копия',
            'Продаю свои SuperPods, состояние идеальное',
            'Дешевые наушники, аналог SuperPods, качество ААА',
            'Дешевые наушники, аналог SuperPods, качество ААА',
            'Дешевые наушники, аналог SuperPods, качество ААА'
        ],
        'target': [0, 1, 0, 1, 1, 1]
    }
    df = pd.DataFrame(data)

    print("Предобработка текста...")
    df['description_cleaned'] = df['description'].apply(clean_text)

    # Разделение данных на обучающую и валидационную выборки
    X = df['description_cleaned']
    y = df['target']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    print("Обучение модели...")
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(3, 5), analyzer='char')),
        ('model', LogisticRegression(random_state=config.RANDOM_STATE, class_weight='balanced'))
    ])

    pipeline.fit(X_train, y_train)

    score = pipeline.score(X_val, y_val)
    print(f"Точность на валидации: {score:.4f}")

    print("Сохранение модели и векторизатора...")
    joblib.dump(pipeline, config.MODEL_PATH)
    print(f"Модель сохранена в {config.MODEL_PATH}")


if __name__ == '__main__':
    train()