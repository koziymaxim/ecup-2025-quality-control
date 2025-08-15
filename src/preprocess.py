import re
import unicodedata

def clean_text(text: str) -> str:
    """
    Функция для базовой очистки текста.
    - Приведение к нижнему регистру
    - Удаление HTML-тегов
    - Юникод-нормализация
    - Удаление всего, что не является буквами или цифрами
    - Удаление лишних пробелов
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[^а-яa-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text