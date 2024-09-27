import os
import tempfile

import pandas as pd
import streamlit as st

from utils import download_video, convert_video

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''


def get_tag(video_path, tags_df):
    return tags_df


st.title("Users Jokers. Rutube | Tagging. Привязка тегов к видео")

upload_option = st.radio("Выберите способ загрузки видео", ("Загрузить файл", "Загрузить по ссылке"))

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

if hasattr(st.session_state, 'video_path') and st.session_state.video_path:
    st.video(st.session_state.video_path)

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
