# Запуск теста для проверки работы рекомендатеьной системы с использованием ML

import app
from fastapi.testclient import TestClient
from datetime import datetime

# Инициализируем тестовый клиент для приложения FastAPI
client = TestClient(app.app)

# Задаем параметры пользователя и времени запроса
user_id = 1000
time = datetime(2021, 12, 20)

try:
    # Выполняем GET-запрос к эндпоинту рекомендаций постов
    r = client.get(
        f"/post/recommendations/",
        params={"id": user_id, "time": time, "limit": 5},
    )
except Exception as e:
    # В случае ошибки выводим информацию о типе и тексте исключения
    raise ValueError(f"Ошибка при выполнении запроса {type(e)} {str(e)}")

# Выводим полученные рекомендации в формате JSON
print(r.json())