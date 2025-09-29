# Сервис для получения рекомендаций постов по фильмам

import sys
sys.path.append('../common')

import os
import pandas as pd
from typing import List
from catboost import CatBoostClassifier
from fastapi import FastAPI
from datetime import datetime
from loguru import logger

from db_connect import get_engine

from pydantic import BaseModel

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
    # Объединяем все чанки в один DataFrame
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
        """SELECT * FROM public.posts_info_features""",
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
    model_path = get_model_path("/Users/anastasiasharina/Documents/ML_Engineer_karpov/Lessons/Module_2_Machine_Learning/final_project/model")
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    return loaded_model

# При запуске сервиса сразу загружаем модель и признаки
logger.info("Загрузка модели")
model = load_models()
logger.info("Загрузка признаков")
features = load_features()
logger.info("Сервис инициализирован и работает")

def get_recommended_feed(id: int, time: datetime, limit: int):
    # Формируем рекомендации для пользователя
    logger.info(f"user_id: {id}")
    logger.info("Чтение признаков")
    user_features = features[2].loc[features[2].user_id == id]
    user_features = user_features.drop('user_id', axis=1)

    logger.info("Удаление столбцов")
    posts_features = features[1].drop(['index', 'text'], axis=1)
    content = features[1][['post_id', 'text', 'topic']]

    logger.info("Объединение всего")
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    logger.info("Присваивание всего")
    user_post_features = posts_features.assign(**add_user_features)
    user_post_features = user_post_features.set_index('post_id')

    logger.info("Добавление информации о времени")
    user_post_features['hour'] = time.hour
    user_post_features['month'] = time.month

    logger.info("Предсказывание")
    predicts = model.predict_proba(user_post_features)[:, 1]
    user_post_features["predicts"] = predicts

    logger.info("Удаление уже лайкнутых постов")
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_post_features[~user_post_features.index.isin(liked_posts)]

    # Выбираем топ-N лучших постов по вероятности лайка
    recommended_posts = filtered_.sort_values('predicts')[-limit:].index

    return [
        PostGet(**{
            "id": i,
            "text": content[content.post_id == i].text.values[0],
            "topic": content[content.post_id == i].topic.values[0]
        }) for i in recommended_posts
    ]

# Эндпоинт для получения рекомендованных постов пользователю
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 10) -> List[PostGet]:
    return get_recommended_feed(id, time, limit)