import os
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from catboost import CatBoostClassifier

# --- TensorFlow (понадобится, когда добавите ML6)
# import tensorflow as tf


# Настройки приложения
st.set_page_config(
    page_title="Smoke Sensors: Fire Alarm — ML Dashboard",
    layout="wide"
)

DATA_PATH = "smokeEDA.csv"
TARGET_COL = "Fire Alarm"

MODELS_DIR = "models"
ASSETS_DIR = ""

MODEL_PATHS = {
    "ML1: LogisticRegression (pkl)": os.path.join(MODELS_DIR, "ML1_LogisticRegression.pkl"),
    "ML2: GradientBoosting (pkl)": os.path.join(MODELS_DIR, "ML2_GradientBoostingClassifier.pkl"),
    "ML3: CatBoost (cbm)": os.path.join(MODELS_DIR, "ML3_CatBoostClassifier.cbm"),
    "ML4: Bagging (pkl)": os.path.join(MODELS_DIR, "ML4_BaggingClassifier.pkl"),
    "ML5: Stacking (pkl)": os.path.join(MODELS_DIR, "ML5_StackingClassifier.pkl"),

    # --- ML6 (добавите позже)
    # Вариант A (SavedModel папка):
    # "ML6: Dense NN (TensorFlow)": os.path.join(MODELS_DIR, "ML6_DenseNN"),
    # Вариант B (один файл .keras):
    # "ML6: Dense NN (TensorFlow)": os.path.join(MODELS_DIR, "ML6_DenseNN.keras"),
}


# Загрузка данных
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Целевая колонка '{TARGET_COL}' не найдена. Колонки: {list(df.columns)}")
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    return feature_cols


# Загрузка моделей
@st.cache_resource
def load_sklearn_model(path: str):
    return joblib.load(path)


@st.cache_resource
def load_catboost_model(path: str) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(path)
    return model


# --- ML6: TensorFlow Dense NN (заготовка; включите когда модель появится)
# @st.cache_resource
# def load_tf_model(path: str):
#     # Поддерживает и SavedModel (папка), и .keras файл
#     return tf.keras.models.load_model(path)


def is_catboost(model_key: str) -> bool:
    return "CatBoost" in model_key


def is_tf_nn(model_key: str) -> bool:
    return "Dense NN" in model_key or "TensorFlow" in model_key


def get_model(model_key: str):
    path = MODEL_PATHS[model_key]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл/папка модели не найден(а): {path}")

    if is_tf_nn(model_key):
        # --- ML6: включите, когда будет TF модель
        # return load_tf_model(path)
        raise FileNotFoundError(
            "TF-модель (ML6) ещё не подключена. "
            "Добавьте файл/папку модели и раскомментируйте строки в коде."
        )

    if is_catboost(model_key):
        return load_catboost_model(path)

    return load_sklearn_model(path)


# Предсказание: 
# Возвращает вероятность класса 1 (Fire Alarm = 1)
def predict_proba(model, X: pd.DataFrame, model_key: str) -> np.ndarray:
    if is_tf_nn(model_key):
        # --- ML6: когда подключите TF модель
        # probs = model.predict(X.to_numpy(), verbose=0).reshape(-1)
        # return probs
        raise RuntimeError("TF-модель не подключена (см. комментарии в коде).")

    # sklearn / catboost
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    preds = model.predict(X)
    return np.asarray(preds).astype(float)


def predict_label_from_proba(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (proba >= threshold).astype(int)


# UI: ввод одной строки
def build_single_row_input(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    desc = df[feature_cols].describe(include="all").T
    defaults = df[feature_cols].mean(numeric_only=True)

    with st.form("single_row_form"):
        st.write("Заполните признаки (1 объект):")
        cols = st.columns(3)
        values = {}

        for i, c in enumerate(feature_cols):
            col = cols[i % 3]

            if pd.api.types.is_numeric_dtype(df[c]):
                c_min = float(desc.loc[c, "min"]) if "min" in desc.columns else float(df[c].min())
                c_max = float(desc.loc[c, "max"]) if "max" in desc.columns else float(df[c].max())
                c_def = float(defaults.get(c, (c_min + c_max) / 2.0))

                step = (c_max - c_min) / 100.0 if c_max > c_min else 1.0
                step = float(step) if np.isfinite(step) and step > 0 else 1.0

                values[c] = col.number_input(
                    label=f"{c}",
                    value=c_def,
                    min_value=c_min,
                    max_value=c_max,
                    step=step,
                    format="%.6f"
                )
            else:
                values[c] = col.text_input(f"{c}", value=str(df[c].mode().iloc[0]))

        submitted = st.form_submit_button("Сделать предсказание")
        if not submitted:
            return None

    return pd.DataFrame([values], columns=feature_cols)


# Страницы
def page_about():
    st.title("Стр. 1 — Информация о разработчике")

    left, right = st.columns([1, 2])

    with left:
        photo_path = os.path.join(ASSETS_DIR, "photo.jpg")
        if os.path.exists(photo_path):
            st.image(photo_path, caption="Фото", use_container_width=True)
        else:
            st.info("Добавьте фото в assets/photo.jpg (необязательно).")

    with right:
        st.markdown(
            """
**ФИО:** Смаилов Тимур  
**Группа:** ФИТ-231  
**Дисциплина:** Машинное обучение и большие данные  
**Тема РГР:** Разработка Web‑приложения (дашборда) для инференса моделей ML и анализа данных  
**Датасет:** smokeEDA.csv (датчики дыма)  
**Цель:** классификация целевого признака `Fire Alarm` (Бинарный)
"""
        )


def page_dataset_info(df: pd.DataFrame):
    st.title("Стр. 2 — Информация о наборе данных и признаках")

    st.subheader("Первые строки")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Размерность")
    st.write(f"Строк: **{df.shape[0]}**, столбцов: **{df.shape[1]}**")

    st.subheader("Пропуски")
    na = df.isna().sum().sort_values(ascending=False)
    st.dataframe(na[na > 0].to_frame("missing_count"), use_container_width=True)
    if (na == 0).all():
        st.success("Пропусков не обнаружено.")

    st.subheader("Типы данных")
    st.dataframe(df.dtypes.to_frame("dtype"), use_container_width=True)

    if TARGET_COL in df.columns:
        st.subheader("Распределение целевого класса")
        vc = df[TARGET_COL].value_counts(dropna=False)
        st.write(vc)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=TARGET_COL, data=df, ax=ax)
        ax.set_title("Target distribution")
        st.pyplot(fig, clear_figure=True)

    st.subheader("Описательная статистика (числовые признаки)")
    st.dataframe(df.describe().T, use_container_width=True)

    """st.info(
        "Важно: для корректного инференса препроцессинг должен совпадать с обучением. "
        "Лучше всего, если в .pkl сохранён sklearn Pipeline (например, StandardScaler + модель)."
    )"""


def page_visuals(df: pd.DataFrame):
    st.title("Стр. 3 — Визуализации зависимостей")

    feature_cols = get_feature_columns(df)

    # 1) Heatmap корреляций
    st.subheader("1) Корреляционная матрица (heatmap)")
    corr = df[feature_cols + [TARGET_COL]].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    st.pyplot(fig, clear_figure=True)

    # 2) Boxplot по одному признаку относительно класса
    st.subheader("2) Boxplot признака по классам")
    box_feature = st.selectbox("Выберите признак для boxplot", feature_cols, index=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x=TARGET_COL, y=box_feature, ax=ax)
    ax.set_title(f"{box_feature} by {TARGET_COL}")
    st.pyplot(fig, clear_figure=True)

    # 3) Scatterplot (2 признака) с hue=target
    st.subheader("3) Scatterplot (2 признака) с раскраской по классу")
    c1, c2 = st.columns(2)
    with c1:
        x_feat = st.selectbox("X", feature_cols, index=0, key="scatter_x")
    with c2:
        y_feat = st.selectbox("Y", feature_cols, index=min(1, len(feature_cols) - 1), key="scatter_y")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_feat, y=y_feat, hue=TARGET_COL, alpha=0.6, ax=ax)
    ax.set_title(f"{x_feat} vs {y_feat}")
    st.pyplot(fig, clear_figure=True)

    # 4) Гистограмма распределения выбранного признака
    st.subheader("4) Гистограмма распределения признака")
    hist_feature = st.selectbox("Признак для гистограммы", feature_cols, index=0, key="hist")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x=hist_feature, hue=TARGET_COL, bins=30, kde=True, ax=ax)
    ax.set_title(f"Distribution of {hist_feature}")
    st.pyplot(fig, clear_figure=True)


def page_inference(df: pd.DataFrame):
    st.title("Стр. 4 — Инференс (предсказание модели)")

    feature_cols = get_feature_columns(df)

    st.sidebar.subheader("Параметры инференса")
    model_key = st.sidebar.selectbox("Выберите модель", list(MODEL_PATHS.keys()))
    threshold = st.sidebar.slider("Порог класса 1 (Fire Alarm)", 0.0, 1.0, 0.5, 0.01)

    try:
        model = get_model(model_key)
        st.success(f"Модель загружена: {model_key}")
    except Exception as e:
        st.error(str(e))
        st.stop()

    tab1, tab2 = st.tabs(["Ручной ввод (1 объект)", "Загрузка CSV (batch)"])

    with tab1:
        X_one = build_single_row_input(df, feature_cols)
        if X_one is not None:
            proba = predict_proba(model, X_one, model_key=model_key)
            pred = predict_label_from_proba(proba, threshold=threshold)

            st.subheader("Результат")
            st.write(f"Вероятность Fire Alarm=1: **{float(proba[0]):.4f}**")
            st.write(f"Предсказанный класс: **{int(pred[0])}**")

            if int(pred[0]) == 1:
                st.warning("Сработает пожарная сигнализация (Fire Alarm = 1).")
            else:
                st.info("Пожарная сигнализация не сработает (Fire Alarm = 0).")

    with tab2:
        st.write("Загрузите CSV с теми же признаками, что и в обучении (без колонки целевого класса).")
        up = st.file_uploader("CSV файл", type=["csv"])

        if up is not None:
            batch = pd.read_csv(up)

            if TARGET_COL in batch.columns:
                batch = batch.drop(columns=[TARGET_COL])

            missing = [c for c in feature_cols if c not in batch.columns]
            extra = [c for c in batch.columns if c not in feature_cols]

            if missing:
                st.error(f"В загруженном файле не хватает колонок: {missing}")
                st.stop()

            if extra:
                st.warning(f"Лишние колонки будут проигнорированы: {extra}")

            Xb = batch[feature_cols].copy()

            proba = predict_proba(model, Xb, model_key=model_key)
            pred = predict_label_from_proba(proba, threshold=threshold)

            out = batch.copy()
            out["proba_fire_alarm_1"] = proba
            out["pred_fire_alarm"] = pred

            st.subheader("Предсказания (первые строки)")
            st.dataframe(out.head(50), use_container_width=True)

            st.download_button(
                "Скачать результаты (CSV)",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )


def main():
    if not os.path.exists(DATA_PATH):
        st.error(f"Файл датасета не найден: {DATA_PATH}")
        st.stop()

    df = load_data(DATA_PATH)

    st.sidebar.title("Навигация")
    page = st.sidebar.radio(
        "Перейти к странице",
        ["1) О разработчике", "2) О датасете", "3) Визуализации", "4) Инференс"],
        index=0
    )

    if page == "1) О разработчике":
        page_about()
    elif page == "2) О датасете":
        page_dataset_info(df)
    elif page == "3) Визуализации":
        page_visuals(df)
    elif page == "4) Инференс":
        page_inference(df)


main()
