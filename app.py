import os
import time

import streamlit as st

from category_mapping import get_iab_tag_by_category
from model_loader import load_tokenizer, load_model, predict_tags
from utils import get_video_info

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

st.title("Привязка тегов к видео")
st.subheader('Users Jokers')

video_path = None

video_url = st.text_input("Введите ссылку на видео")
if video_url:
    video_info = get_video_info(video_url)
    st.header(video_info.title)
    st.image(video_info.thumbnail_url)
    st.write(video_info.description)
    st.caption(video_info.category)

    st.subheader('Теги')
    start = time.time()
    predicted_tags_by_description = predict_tags(video_info.description, model, tokenizer)
    prediction_by_description_time = time.time() - start
    st.write(f'По описанию: {", ".join(predicted_tags_by_description)}')
    st.caption(f'Время работы: {prediction_by_description_time}с')

    start = time.time()
    predicted_tags_by_category = get_iab_tag_by_category(video_info.category)
    prediction_by_category_time = time.time() - start
    st.write(f'По категории: {predicted_tags_by_category}')
    st.caption(f'Время работы: {prediction_by_category_time}с')
