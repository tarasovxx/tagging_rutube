# utils.py

import re
import nltk
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from transformers import T5ForConditionalGeneration, GPT2Tokenizer
import torch
import ffmpeg
from pydub import AudioSegment
import srt
from datetime import timedelta
import whisper
import numpy as np
import av
import tiktoken

nltk.download('stopwords')

def process_text(text, stop_phrases):
    """
    Очистка текста от стоп-слов, пунктуации, лемматизация, удаление тематического мусора и суммаризация.
    """
    # Приведение к нижнему регистру
    text = text.lower()

    # Удаление пунктуации
    text = re.sub(r'[^\w\s]', '', text)

    # Удаление цифр
    text = re.sub(r'\d+', '', text)

    # Удаление стоп-фраз
    for phrase in stop_phrases:
        text = text.replace(phrase.lower(), '')

    # Токенизация
    tokens = nltk.word_tokenize(text, language='russian')

    # Удаление стоп-слов
    stop_words = set(stopwords.words('russian'))
    tokens = [word for word in tokens if word not in stop_words]

    # Лемматизация
    morph = MorphAnalyzer()
    tokens = [morph.normal_forms(word)[0] for word in tokens]

    # Объединение в строку
    processed_text = ' '.join(tokens)

    return processed_text

def get_summarize_text(text: str) -> str:
    """
    Суммаризация текста
    """
    tokenizer = GPT2Tokenizer.from_pretrained("RussianNLP/FRED-T5-Summarizer", eos_token="</s>")
    model = T5ForConditionalGeneration.from_pretrained("RussianNLP/FRED-T5-Summarizer")

    input_text = rf"""<LM> Сократи текст.\n ({text})"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs,
        max_length=600,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
        top_p=0.9
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary

def convert_mp4_to_txt(input_file: str, output_file="cache/temp.mp3") -> str | None:
    """
    Конвертация MP4 видео в текст с использованием Whisper
    """
    if not os.path.isfile(input_file):
        print(f"Файл {input_file} не найден.")
        return None

    # Извлечение аудио
    audio_stream = ffmpeg.input(input_file).audio
    aac_file = os.path.join(os.getcwd(), "cache", "temp_audio.aac")
    if os.path.isfile(aac_file):
        os.remove(aac_file)

    ffmpeg.output(audio_stream, aac_file).run()

    # Конвертация в MP3
    audio = AudioSegment.from_file(aac_file, format='aac')
    audio.export(output_file, format='mp3')

    output_audio = output_file

    # Загрузка модели Whisper
    model_name = 'medium'
    model = whisper.load_model(model_name)

    # Транскрибирование
    result = model.transcribe(output_audio, language='russian')

    # Сбор транскрибированного текста
    raw_text = [segment['text'].strip() for segment in result['segments']]
    transcribed_text = " ".join(raw_text)

    return transcribed_text

def read_video_pyav(container, indices):
    """
    Декодирование видео с помощью PyAV.
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def num_tokens_from_string(string, encoding_name: str) -> int:
    """
    Подсчет количества токенов в строке.
    """
    string = str(string)
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def extract_tags_from_response(response):
    """
    Извлечение тегов из ответа ассистента в формате [тег1, тег2, ...]
    """
    matches = re.findall(r'\[(.*?)\]', response)
    tags = []
    for match in matches:
        split_tags = re.split(r',|\n', match)
        split_tags = [tag.strip() for tag in split_tags if tag.strip()]
        tags.extend(split_tags)
    return tags
