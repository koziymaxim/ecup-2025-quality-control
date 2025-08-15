import pandas as pd
import joblib
import argparse

from preprocess import clean_text

def predict(data_path: str, model_path: str, output_path: str):
    """
    Основная функция для генерации предсказаний.
    """
    print(f"Загрузка обученной модели из: {model_path}")
    try:
        pipeline = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Ошибка: Файл модели не найден по пути: {model_path}")
        return

    print(f"Загрузка тестовых данных из: {data_path}")
    try:
        df_test = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Ошибка: Файл с данными не найден по пути: {data_path}")
        return

    print("Предобработка текста...")
    if 'description' not in df_test.columns:
        print("Ошибка: В тестовых данных отсутствует колонка 'description'")
        return
        
    df_test['description_cleaned'] = df_test['description'].apply(clean_text)

    print("Создание предсказаний...")
    predictions_proba = pipeline.predict_proba(df_test['description_cleaned'])[:, 1]
    predictions_class = pipeline.predict(df_test['description_cleaned'])

    submission = pd.DataFrame({
        'id': df_test['id'],
        'target': predictions_class,
        'probability': predictions_proba
    })

    submission.to_csv(output_path, index=False)
    print(f"Файл с предсказаниями успешно сохранен в: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скрипт для предсказания контрафактных товаров.")

    parser.add_argument('--data_path', default='input_data/test.csv', help='Путь к тестовому CSV файлу')
    parser.add_argument('--model_path', default='models/model.joblib', help='Путь к сохраненной модели')
    parser.add_argument('--output_path', default='output_data/submission.csv', help='Путь для сохранения результата')

    args = parser.parse_args()

    predict(
        data_path=args.data_path,
        model_path=args.model_path,
        output_path=args.output_path
    )