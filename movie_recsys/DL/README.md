# DL сервис рекомендательной системы

## Описание

DL-сервис реализует рекомендательную систему на базе современных трансформеров (BERT, RoBERTa, DistilBERT) для генерации эмбеддингов текстов постов. Кластеризация эмбеддингов и вычисление кластерных расстояний используются как дополнительные признаки для модели CatBoost. Сервис предоставляет API для получения рекомендаций через FastAPI.

## Этапы пайплайна

1. **Загрузка и анализ данных**  
   - Импорт данных из PostgreSQL через общий модуль `common/db_connect.py`.
   - Подготовка текстов, анализ структуры и пропусков.

2. **Генерация эмбеддингов**  
   - Использование предобученных моделей: BERT, RoBERTa, DistilBERT из HuggingFace Transformers.
   - Построение датасета и загрузчика данных (PyTorch DataLoader), автоматический паддинг (DataCollator).

3. **Кластеризация эмбеддингов**  
   - Снижение размерности (PCA).
   - Кластеризация KMeans, добавление кластерных расстояний и меток к постам.

4. **Сохранение признаков**  
   - Сохраняются признаки постов (эмбеддинги, кластеры, расстояния) в PostgreSQL для дальнейшего использования в моделях.

5. **Обучение модели CatBoost**  
   - Объединение фичей эмбеддингов с пользовательскими признаками.
   - Обучение CatBoost на полном наборе признаков.

6. **API сервис**  
   - Реализация FastAPI для онлайн-рекомендаций.
   - Эндпоинт `/post/recommendations/` возвращает топ-N постов для пользователя.

## Полный стек технологий

- **Python 3.8+**
- **PyTorch**
- **Transformers (HuggingFace)**
- **CatBoost**
- **scikit-learn**
- **Pandas, NumPy**
- **FastAPI**
- **SQLAlchemy**
- **PostgreSQL**
- **Docker**
- **tqdm (инференс)**

## Метрики качества

- **ROC-AUC** — основная метрика модели CatBoost (train/test).
- **Hitrate** — финальная метрика для LMS.
- **Feature Importance** — анализ влияния эмбеддингов и кластерных признаков.

## Как запустить

### Локально

1. Установить зависимости:
    ```bash
    pip install -r ../common/requirements.txt
    ```

2. Запустить сервис:
    ```bash
    python service_dl.py
    ```

### Через Docker

```bash
docker build -t ml-dl-ab --build-arg PROJECT=dl --build-arg SERVICE_FILE=service_dl.py .
docker run --env-file ../common/.env -p 8001:8000 ml-dl-ab
```

## Пример запроса к API

```python
import requests
r = requests.get("http://localhost:8001/post/recommendations/", params={"id": 1000, "time": "2021-12-20T00:00:00", "limit": 5})
print(r.json())
```

## Тестирование

- Юнит-тесты для сервиса лежат в `test_dl.py`.
- Для проверки API используйте FastAPI TestClient или curl.

## Контакты

Автор: Анастасия Шарина  
GitHub: [anastasia-sharina](https://github.com/anastasia-sharina)