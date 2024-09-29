import subprocess
import tempfile
from dataclasses import dataclass

import requests
import streamlit as st
import yt_dlp


@dataclass
class VideoInfo:
    media_id: str
    title: str
    description: str
    thumbnail_url: str
    category: str | None


def get_video_info(url: str) -> VideoInfo:
    media_id: str = url.rstrip('/').rsplit('/', 1)[-1]
    data: dict = requests.get(f'https://rutube.ru/api/play/options/{media_id}/').json()
    return VideoInfo(
        media_id=media_id,
        title=data['title'],
        description=data['description'],
        thumbnail_url=data['thumbnail_url'],
        category=data['category']['name'],
    )


def download_video(url: str) -> str | None:
    """
    Функция для скачивания видео с Rutube по ссылке
    :param url: ссылка
    :return: str - путь к файлу, куда скачалось видео
    """
    progress_bar = st.progress(0)  # Инициализация прогресс-бара в интерфейсе Streamlit
    status_text = st.empty()  # Инициализация пустого текстового поля для статуса

    def progress_hook(d):
        if d['status'] == 'downloading':
            total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate')
            downloaded_bytes = d.get('downloaded_bytes', 0)
            if total_bytes:
                progress_percentage = downloaded_bytes / total_bytes
                progress_bar.progress(progress_percentage)
                status_text.text(f"Скачано: {progress_percentage:.2%}")

        if d['status'] == 'finished':
            progress_bar.progress(1)
            status_text.text("Скачано")
            progress_bar.empty()
            status_text.empty()

    ydl_opts = {
        'format': 'best',  # Скачивание в лучшем доступном качестве
        'outtmpl': tempfile.mktemp(suffix=".mp4"),  # Временный файл для сохранения видео
        'quiet': True,  # Отключение вывода логов в консоль
        'progress_hooks': [progress_hook],  # Хук для обновления прогресса скачивания
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)  # Извлечение и скачивание видео
            video_file_path = ydl.prepare_filename(info_dict)  # Получение пути к скачанному файлу
            return video_file_path
    except Exception as e:
        st.error(f"Произошла ошибка при скачивании видео: {e}")
        return None


def convert_video(input_path: str) -> str | None:
    """
    Функция ковертации видео из формата mp4 непонятного системе в формат mp4 для юзера и системы
    :param input_path:
    :return:
    """
    output_path = tempfile.mktemp(suffix=".mp4")
    try:
        subprocess.run(['ffmpeg', '-i', input_path, '-c', 'copy', output_path],
                       check=True)  # Запуск команды ffmpeg для конвертации
        return output_path
    except subprocess.CalledProcessError as e:
        st.error(f"Произошла ошибка при конвертации видео: {e}")
        return None


def get_flatten_iab_tags(csv_path: str) -> list[str]:
    """
    Функция получения списка тегов из csv файл
    :param csv_path:
    :return: list[str]
    """
    iab_tags = list(map(str.strip, open(csv_path).readlines()[1:]))  # пропускаем header
    flatten_tags = [
        ': '.join(map(str.strip, filter(bool, line.split(','))))  # сплитим с ', ', нормализуем и объединяем с ': '
        for line in iab_tags
    ]
    return list(filter(bool, flatten_tags))  # фильтруем пустые строки


def print_red(text) -> None:
    """
    Функция вывода текста в цветовой форме в красный цвет
    :param text:
    :return:
    """
    red = "\033[31m"
    reset = "\033[0m"
    print(f"{red}{text}{reset}")


def print_green(text) -> None:
    """
    Функция вывода текста в цветовой форме в зеленый цвет
    :param text:
    :return:
    """
    green = "\033[32m"
    reset = "\033[0m"
    print(f"{green}{text}{reset}")
