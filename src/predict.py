import pandas as pd
import joblib

from preprocess import clean_text
import config

def predict():
    """Функция для генерации предсказаний."""
    print("Загрузка обученной модели...")
    pipeline = joblib.load(config.MODEL_PATH)

    print("Загрузка тестовых данных...")
    # Фейковые данные для примера
    # TODO: заменить на pd.read_csv(config.TEST_DATA_PATH)
    test_data = {
        'id': [101, 102, 103],
        'description': [
            'Наушники SuperPods, почти новые',
            '100% копия, не отличить от оригинала',
            'Беспроводные наушники, полный комплект'
        ]
    }
    df_test = pd.DataFrame(test_data)

    print("Предобработка текста...")
    df_test['description_cleaned'] = df_test['description'].apply(clean_text)

    print("Создание предсказаний...")
    # Используем .predict_proba() для получения вероятностей
    predictions_proba = pipeline.predict_proba(df_test['description_cleaned'])[:, 1]
    # Используем .predict() для получения меток класса
    predictions_class = pipeline.predict(df_test['description_cleaned'])

    # Формирование файла для сабмита
    submission = pd.DataFrame({
        'id': df_test['id'],
        'target': predictions_class,
        'probability': predictions_proba
    })

    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Файл с предсказаниями сохранен в {config.SUBMISSION_PATH}")


if __name__ == '__main__':
    predict()