# Сервис для получения рекомендаций постов

import sys
sys.path.append('../common')

import os
import pandas as pd
from typing import List
from catboost import CatBoostClassifier
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from loguru import logger

from db_connect import get_engine

# Описываем модель данных для поста
class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    # Включаем поддержку работы с ORM для интеграции с базой данных
    class Config:
        orm_mode = True

app = FastAPI()

def batch_load_sql(query: str):
    # Загружаем данные из SQL-запроса большими чанками для экономии памяти
    engine = get_engine()
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=200000):
        chunks.append(chunk_dataframe)
        logger.info(f"Имеем чанк: {len(chunk_dataframe)}")
    conn.close()
    # Объединяем все чанки в один датафрейм
    return pd.concat(chunks, ignore_index=True)

def get_model_path(path: str) -> str:
    # Определяем путь к модели CatBoost в зависимости от окружения
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_features():
    # Загружаем признаки для работы модели
    logger.info("Загружаем залайканные посты")
    liked_posts_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        WHERE action = 'like'
    """
    liked_posts = batch_load_sql(liked_posts_query)

    engine = get_engine()

    logger.info("Загружаем признаки постов")
    posts_features = pd.read_sql(
        """SELECT * FROM public.anastasia_sharina_77""",
        con=engine
    )

    logger.info("Загружаем признаки юзеров")
    user_features = pd.read_sql(
        """SELECT * FROM public.user_data""",
        con=engine
    )    

    return [liked_posts, posts_features, user_features]

def load_models():
    # Загружаем обученную CatBoost модель из файла
    model_path = get_model_path("/Users/anastasiasharina/Documents/ML_Engineer_karpov/Lessons/Module_3_Deep_Learning/final_project/model.cbm")
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    return loaded_model

# При запуске сервиса сразу загружаем модель и признаки
logger.info("Загрузка модели")
model = load_models()
logger.info("Загрузка признаков")
features = load_features()
logger.info("Сервис инициализирован и работает")

# Список категориальных признаков как при обучении
object_cols = [
    'topic', 'TextCluster', 'gender', 'country',
    'city', 'exp_group', 'hour', 'month',
    'os', 'source'
]

# Итоговый порядок признаков по обучающему X
features_order = [
    "hour",
    "month",
    "gender",
    "age",
    "country",
    "city",
    "exp_group",
    "os",
    "source",
    "topic",
    "TextCluster",
    "DistanceToCluster_0",
    "DistanceToCluster_1",
    "DistanceToCluster_2",
    "DistanceToCluster_3",
    "DistanceToCluster_4",
    "DistanceToCluster_5",
    "DistanceToCluster_6",
    "DistanceToCluster_7",
    "DistanceToCluster_8",
    "DistanceToCluster_9",
    "DistanceToCluster_10",
    "DistanceToCluster_11",
    "DistanceToCluster_12",
    "DistanceToCluster_13",
    "DistanceToCluster_14"
]

def get_recommended_feed(id: int, time: datetime, limit: int):
    # Формируем рекомендации для пользователя
    logger.info(f"user_id: {id}")

    # 1. Признаки пользователя
    user_features = features[2].loc[features[2].user_id == id].drop('user_id', axis=1)
    if user_features.empty:
        logger.warning(f"User {id} не найден в user_features!")
        return []

    # 2. Признаки постов
    posts_features = features[1].copy()
    content = features[1][['post_id', 'topic']]

    # 3. Добавляем к каждому посту фичи пользователя (broadcast)
    user_dict = dict(zip(user_features.columns, user_features.values[0]))
    user_post_features = posts_features.assign(**user_dict)
    user_post_features = user_post_features.set_index('post_id')

    # 4. Добавляем временные фичи
    user_post_features['hour'] = time.hour
    user_post_features['month'] = time.month

    # 5. Приведение категориальных признаков к строке
    for col in object_cols:
        if col in user_post_features.columns:
            user_post_features[col] = user_post_features[col].astype(str)

    # формируем X в нужном порядке
    X = user_post_features[features_order]

    # 6. Предсказание вероятности лайка для всех постов
    predicts = model.predict_proba(X)[:, 1]
    user_post_features["predicts"] = predicts

    # 7. Убираем уже залайканные посты
    liked_posts = features[0]
    liked_post_ids = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_post_features[~user_post_features.index.isin(liked_post_ids)]

    # 8. Ранжируем по вероятности и выдаём top-5
    recommended_posts = filtered_.sort_values('predicts', ascending=False).head(limit).index

    return [
        PostGet(
            id=int(i),
            topic=content[content.post_id == i].topic.values[0]
        )
        for i in recommended_posts
    ]

# Эндпоинт для получения рекомендованных постов пользователю
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 10) -> List[PostGet]:
    return get_recommended_feed(id, time, limit)