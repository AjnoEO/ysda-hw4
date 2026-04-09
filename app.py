"""
Приложение для предсказания меток

Dual Recurrent Attention Units for Visual Question Answering
We propose an architecture for VQA which utilizes recurrent layers to generate visual and textual attention. The memory characteristic of the proposed recurrent attention units offers a rich joint embedding of visual and textual features and enables the model to reason relations between several parts of the image and question. Our single model outperforms the first place winner on the VQA 1.0 dataset, performs within margin to the current state-of-the-art ensemble model. We also experiment with replacing attention mechanisms in other state-of-the-art models with our implementation and show increased accuracy. In both cases, our recurrent attention mechanism improves performance in tasks requiring sequential or relational reasoning on the VQA dataset
"""
import data
from model import create_model

import streamlit as st
import pandas as pd

@st.cache_resource(show_spinner=False)
def get_model():
    return create_model()

st.session_state.setdefault('threshold_value', 0.62)

st.title("Тэггер статей")
st.markdown("Введите заголовок и абстракт статьи для определения подходящих тэгов тематик! "
            "**Модель работает с английскими статьями.**")

st.text_input("Заголовок статьи", key="article_title")
st.text_area("Абстракт статьи", key="article_abstract")
st.slider("Порог вероятности", min_value=0.3, max_value=0.9, key='threshold_value')
if st.button("Определить тэги"):
    with st.spinner("Подгружаю модель..."):
        model = get_model()
    with st.spinner("Подбираю тэги..."):
        probabilities = model.get_sample_proba(st.session_state["article_title"], st.session_state["article_abstract"])
    st.session_state["class_proba"] = probabilities

if "class_proba" in st.session_state:    
    df = pd.DataFrame({"Код тэга": data.TOPICS["List"], "Вероятность": st.session_state["class_proba"]})
    df = df[df["Вероятность"] > st.session_state['threshold_value']]
    if (len(df) == 0):
        st.warning("Подходящего тэга не нашлось. Ваша статья уникальна! Попробуйте понизить порог")
    else:
        df.sort_values("Вероятность", inplace=True, ascending=False)
        df["tag"] = df["Код тэга"].map(data.TOPICS["Translations"])
        mask = ~df["tag"].str.contains(", ")
        df["tag"][mask] = df["tag"][mask] + ", " + df["tag"][mask]
        df[["Макротема", "Тема"]] = df["tag"].str.split(", ", n=1, expand=True)
        st.bar_chart(df, x="Код тэга", y="Вероятность", color="Orange")
        st.dataframe(df[["Макротема", "Тема", "Вероятность"]], hide_index=True)

with st.spinner("Подгружаю модель..."):
    get_model()
