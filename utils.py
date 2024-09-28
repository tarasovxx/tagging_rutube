import subprocess
import tempfile

import streamlit as st
import yt_dlp


def download_video(url: str) -> str | None:
    progress_bar = st.progress(0)
    status_text = st.empty()

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
        'format': 'best',
        'outtmpl': tempfile.mktemp(suffix=".mp4"),
        'quiet': True,
        'progress_hooks': [progress_hook],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_file_path = ydl.prepare_filename(info_dict)
            return video_file_path
    except Exception as e:
        st.error(f"Произошла ошибка при скачивании видео: {e}")
        return None


def convert_video(input_path: str) -> str | None:
    output_path = tempfile.mktemp(suffix=".mp4")
    try:
        subprocess.run(['ffmpeg', '-i', input_path, '-c', 'copy', output_path], check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        st.error(f"Произошла ошибка при конвертации видео: {e}")
        return None


def get_flatten_iab_tags(csv_path: str) -> list[str]:
    iab_tags = list(map(str.strip, open(csv_path).readlines()[1:]))  # skip header
    flatten_tags = [
        ': '.join(map(str.strip, filter(bool, line.split(','))))  # split with ', ', strip and join with ': '
        for line in iab_tags
    ]
    return list(filter(bool, flatten_tags))  # filter empty strings


def print_red(text):
    red = "\033[31m"
    reset = "\033[0m"
    print(f"{red}{text}{reset}")


def print_green(text):
    green = "\033[32m"
    reset = "\033[0m"
    print(f"{green}{text}{reset}")
