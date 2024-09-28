import os
import tempfile

import pandas as pd
import streamlit as st

from model_loader import load_tokenizer, load_model, predict_tags
from handler import convert_mp4_to_txt
from utils import download_video, convert_video

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''


@st.cache_resource
def get_tokenizer():
    return load_tokenizer()


@st.cache_resource
def get_model():
    return load_model()


tokenizer = get_tokenizer()
model = get_model()


def get_tag(video_path, tags_df):
    return tags_df

st.title("Users Jokers. Rutube | Tagging. Привязка тегов к видео")

upload_option = st.radio("Выберите способ загрузки видео", ("Загрузить файл", "Загрузить по ссылке", "Предсказать по описанию"))

video_path = None

if upload_option == "Загрузить файл":
    video_file = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        # video_path = tfile.name
        st.session_state.video_path = tfile.name

elif upload_option == "Загрузить по ссылке":
    video_url = st.text_input("Введите ссылку на видео")
    if st.button("Скачать видео"):
        download_video_path = download_video(video_url)
        if download_video_path:
            st.session_state.video_path = convert_video(download_video_path)

elif upload_option == "Предсказать по описанию":
    text = st.text_area("Введите описание")
    threshold = st.select_slider("Порог", options=[0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    if st.button("Предсказать тэги"):
        tags = ', '.join(predict_tags(text, model, tokenizer, threshold=threshold))
        st.write(tags)

if hasattr(st.session_state, 'video_path') and st.session_state.video_path:
    st.video(st.session_state.video_path)

    st.write("Транскрибация видео:")

    with st.spinner('Идет транскрибация, пожалуйста подождите...'):
        transcript = convert_mp4_to_txt(st.session_state.video_path)

    st.write(transcript)

tags_file = st.file_uploader("Загрузите CSV файл с тегами", type=["csv"])
if tags_file is not None:
    tags_df = pd.read_csv(tags_file)
    st.write("Загруженные теги:")
    st.dataframe(tags_df)

if st.button("Привязать теги к видео"):
    if st.session_state.video_path is not None and tags_file is not None:
        result_tags = get_tag(st.session_state.video_path, tags_df)
        st.write("Результат привязки тегов:")
        st.dataframe(result_tags)
    else:
        st.warning("Пожалуйста, загрузите и видео, и CSV файл с тегами.")
