# Тестирование API-сервиса рекомендаций с помощью FastAPI TestClient.
# Такой подход позволяет проверять корректность работы эндпоинта без запуска внешнего сервера,
# что удобно для юнит-тестирования и валидации бизнес-логики рекомендательной системы.

import service
from fastapi.testclient import TestClient
from datetime import datetime

client = TestClient(service.app)

user_id = 1000
time = datetime(2021, 12, 20)

try:
    r = client.get(
        "/post/recommendations/",
        params={"id": user_id, "time": time, "limit": 5},
    )
except Exception as e:
    raise ValueError(f"Ошибка при выполнении запроса {type(e)} {str(e)}")

print(r.json())