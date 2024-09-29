# main.py

import os
import json
import numpy as np
import torch
import pandas as pd
from transformers import BitsAndBytesConfig, LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor
from utils import (read_video_pyav, num_tokens_from_string, get_summarize_text,
                   convert_mp4_to_txt, process_text, extract_tags_from_response)
import av

# Определение всех путей к файлам
DATA_FOLDER = '/kaggle/input/test_tag_video'
TAG_LIST_FILE = os.path.join(DATA_FOLDER, 'IAB_tags_list.json')
SAMPLE_SUBMISSION_FILE = os.path.join(DATA_FOLDER, 'sample_submission.csv')
VIDEO_FOLDER = os.path.join(DATA_FOLDER, 'videos')

# Загрузка списка тегов
with open(TAG_LIST_FILE, 'r', encoding='utf-8') as file:
    iab_tags = json.load(file)

tags_lvl1 = list(iab_tags.keys())
tags_lvl2 = {}
tags_lvl3 = {}

for lvl1_tag in tags_lvl1:
    tags_lvl2[lvl1_tag] = []
    for lvl2_tag, lvl3_tags in iab_tags[lvl1_tag].items():
        tags_lvl2[lvl1_tag].append(lvl2_tag)
        if isinstance(lvl3_tags, list):
            tags_lvl3[lvl2_tag] = lvl3_tags
        else:
            tags_lvl3[lvl2_tag] = []

# Загрузка данных из sample_submission.csv
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_FILE)

# Инициализация модели LLava
processor = LlavaOnevisionProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
processor.tokenizer.padding_side = "left"

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    torch_dtype="float16",
    device_map='auto',
)

generate_kwargs = {"max_new_tokens": 22, "do_sample": True, "top_p": 0.9}

# Список тематических "мусорных" фраз
stop_phrases = [
    'в этом выпуске',
    'по вопросам рекламы пишите на email',
    'в новом выпуске',
    'байкальская миля',
    '20 00',
    'такой расклад',
    'выпуске шоу',
    'и по',
    'том что',
    'в котором вы узнаете',
    # Добавьте дополнительные фразы по необходимости
]

# Обработка каждого видео
for idx, row in sample_submission.iterrows():
    video_id = row['video_id']
    title = row['title']
    description = row['description']

    # Обработка заголовка и описания
    processed_title = process_text(title, stop_phrases)
    processed_description = process_text(description, stop_phrases)

    # Суммаризация заголовка и описания
    summarized_title = get_summarize_text(processed_title)
    summarized_description = get_summarize_text(processed_description)

    # Путь к видеофайлу
    video_path = os.path.join(VIDEO_FOLDER, f"{video_id}.mp4")

    # Транскрибирование и суммаризация аудио из видео
    transcribed_text = convert_mp4_to_txt(video_path, output_file=f"cache/{video_id}.mp3")
    if transcribed_text:
        processed_transcription = process_text(transcribed_text, stop_phrases)
        summarized_transcription = get_summarize_text(processed_transcription)
    else:
        summarized_transcription = ''

    # Комбинация обработанных и суммированных текстов
    combined_text = f"{summarized_title} {summarized_description} {summarized_transcription}"

    # Создание подсказки для уровня 1
    tags_lvl1s = '\n'.join(tags_lvl1)

    PROMPT = f"""
Вы являетесь экспертом по тегированию видео в соответствии с IAB Content Taxonomy. На основе предоставленного видео и следующей информации: {combined_text}, выберите наиболее подходящие теги из списка ниже.

**Возможные теги**:
{tags_lvl1s}

Ответьте на русском языке в формате: [теги]. Выберите один или несколько тегов, соответствующих содержанию видео и информации.
"""

    # Подготовка данных для модели LLava
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "video"},
                ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Получение кадров из видео
    N = 8  # Количество кадров для выборки
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / N).astype(int)
    clip = read_video_pyav(container, indices)

    if len(clip) < N:
        print(f"Пропускаем видео {video_id} из-за недостаточного количества кадров.")
        continue

    inputs = processor(text=prompt, videos=[clip], padding=True, return_tensors="pt").to(model.device, torch.float16)

    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)

    # Извлечение ответа ассистента
    assistant_index = generated_text[0].find('assistant\n')
    if assistant_index != -1:
        response = generated_text[0][assistant_index + len('assistant\n'):]
    else:
        response = generated_text[0]

    # Извлечение предсказанных тегов уровня 1
    predicted_lvl1_tags = extract_tags_from_response(response)

    # Предсказание тегов уровней 2 и 3
    predicted_tags = predicted_lvl1_tags.copy()
    for lvl1_tag in predicted_lvl1_tags:
        lvl2_tags = tags_lvl2.get(lvl1_tag, [])
        if lvl2_tags:
            # Подготовка подсказки для уровня 2
            lvl2_tags_str = '\n'.join(lvl2_tags)
            PROMPT_LVL2 = f"""
На основе выбранного тега "{lvl1_tag}" и предоставленной информации: {combined_text}, выберите наиболее подходящие подкатегории из списка ниже.

**Возможные подкатегории**:
{lvl2_tags_str}

Ответьте на русском языке в формате: [подкатегории]. Выберите один или несколько подкатегорий, соответствующих содержанию видео и информации.
"""

            conversation_lvl2 = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT_LVL2},
                        {"type": "video"},
                        ],
                },
            ]

            prompt_lvl2 = processor.apply_chat_template(conversation_lvl2, add_generation_prompt=True)

            inputs_lvl2 = processor(text=prompt_lvl2, videos=[clip], padding=True, return_tensors="pt").to(model.device, torch.float16)

            output_lvl2 = model.generate(**inputs_lvl2, **generate_kwargs)
            generated_text_lvl2 = processor.batch_decode(output_lvl2, skip_special_tokens=True)

            assistant_index = generated_text_lvl2[0].find('assistant\n')
            if assistant_index != -1:
                response_lvl2 = generated_text_lvl2[0][assistant_index + len('assistant\n'):]
            else:
                response_lvl2 = generated_text_lvl2[0]

            predicted_lvl2_tags = extract_tags_from_response(response_lvl2)

            predicted_tags.extend(predicted_lvl2_tags)

            # Предсказание тегов уровня 3
            for lvl2_tag in predicted_lvl2_tags:
                lvl3_tags = tags_lvl3.get(lvl2_tag, [])
                if lvl3_tags:
                    lvl3_tags_str = '\n'.join(lvl3_tags)
                    PROMPT_LVL3 = f"""
На основе выбранной подкатегории "{lvl2_tag}" и предоставленной информации: {combined_text}, выберите наиболее подходящие подподкатегории из списка ниже.

**Возможные подподкатегории**:
{lvl3_tags_str}

Ответьте на русском языке в формате: [подподкатегории]. Выберите один или несколько подподкатегорий, соответствующих содержанию видео и информации.
"""

                    conversation_lvl3 = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": PROMPT_LVL3},
                                {"type": "video"},
                                ],
                        },
                    ]

                    prompt_lvl3 = processor.apply_chat_template(conversation_lvl3, add_generation_prompt=True)

                    inputs_lvl3 = processor(text=prompt_lvl3, videos=[clip], padding=True, return_tensors="pt").to(model.device, torch.float16)

                    output_lvl3 = model.generate(**inputs_lvl3, **generate_kwargs)
                    generated_text_lvl3 = processor.batch_decode(output_lvl3, skip_special_tokens=True)

                    assistant_index = generated_text_lvl3[0].find('assistant\n')
                    if assistant_index != -1:
                        response_lvl3 = generated_text_lvl3[0][assistant_index + len('assistant\n'):]
                    else:
                        response_lvl3 = generated_text_lvl3[0]

                    predicted_lvl3_tags = extract_tags_from_response(response_lvl3)

                    predicted_tags.extend(predicted_lvl3_tags)

    # Сохранение предсказанных тегов
    row['predicted_tags'] = ', '.join(predicted_tags)
    sample_submission.loc[idx] = row

# Сохранение результатов
sample_submission.to_csv('submission.csv', index=False)
