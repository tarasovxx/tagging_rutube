import locale
import os
from datetime import timedelta

# if not already_processed:
#   !pip uninstall -y ffmpeg-python (maybe sudo apt install ffmpeg | brew install ffmpeg)
#   !pip uninstall ffmpeg
#   !pip install ffmpeg-python
import ffmpeg
import srt
import whisper
# @title Обработка входного файла и перевод в mp3
from pydub import AudioSegment

from utils import print_red

# @markdown Файл был предобработан (т.е. сейчас в формате Mp3)?
already_processed_ = "\u0414\u0430"  # @param ["Да", "Нет"]
already_processed = already_processed_ == "Нет"


def getpreferredencoding(do_setlocale=True) -> str:
    """
    Функция для получения предпочтительной кодировки (устанавливаем UTF-8)
    :param do_setlocale:
    :return: str
    """
    return "UTF-8"


locale.getpreferredencoding = getpreferredencoding


def convert_mp4_to_txt(input_file: str, output_file="cache/temp.mp3") -> str | None:
    """
    Основная функция для конвертации видеофайла MP4 в текст (сохраняется .txt файл, а также возвращется строка
    :param input_file: str (path)
    :param output_file: str (path)
    :return: str | None
    """
    # Проверка наличия входного файла
    if os.path.isfile(input_file):
        print(f"\nФайл с местоположением `{input_file}` найден")
    else:
        print_red(f"\nФайл с местоположением `{input_file}` не найден")
        return

    print(f"output_file = {output_file}")
    print(f"input_file = {input_file}")

    # Извлечение аудиопотока (AAC) из видеофайла
    audio_stream = ffmpeg.input(input_file).audio
    aac_file = os.path.join(os.getcwd(), "cache", "temp_audio.aac")
    if os.path.isfile(aac_file):
        os.remove(aac_file)

    # Сохранение извлеченного аудиопотока в формате AAC
    ffmpeg.output(audio_stream, aac_file).run()

    # Конвертация AAC в MP3 с использованием библиотеки pydub
    audio = AudioSegment.from_file(aac_file, format='aac')
    audio.export(output_file, format='mp3')
    # audio.export(output_file, format='mp3')

    output_audio = output_file

    # Загрузка модели Whisper для транскрибации
    model_name = 'medium'  # @param ["tiny", "small", "medium", "large", "tiny.en", "small.en", "medium.en"]
    # Disable SSL verification globally
    # import ssl
    # ssl._create_default_https_context = ssl._create_unverified_context  # Отключение проверки SSL (для локального использования)
    # os.environ['CURL_CA_BUNDLE'] = ''  # Отключение проверки сертификатов (для локального использования)
    # os.environ['REQUESTS_CA_BUNDLE'] = ''

    cache_dir = os.path.join(os.getcwd(), "models", "weights", "whisper")
    model_path = os.path.join(cache_dir, f"{model_name}.pt")

    if not os.path.exists(model_path):
        print(f"Модель не найдена по пути {model_path}. Загрузка модели...")

        # Загружаем модель
        model = whisper.load_model(model_name)

        # Сохраняем модель в кэш
        os.makedirs(cache_dir, exist_ok=True)
        model.save(model_path)
        print(f"Модель сохранена по пути {model_path}")
    else:
        print(f"Модель найдена по пути {model_path}. Загружаем модель из файла...")
        model = whisper.load_model(model_path)

    audio_file_name = output_audio

    audio_file_language = 'russian'  # @param ['автоматически', 'english', 'chinese', 'german', 'spanish', 'russian', 'korean', 'french', 'japanese', 'afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'assamese', 'azerbaijani', 'bashkir', 'basque', 'belarusian', 'bengali', 'bosnian', 'breton', 'bulgarian', 'burmese', 'cantonese', 'castilian', 'catalan', 'croatian', 'czech', 'danish', 'dutch', 'estonian', 'faroese', 'finnish', 'flemish', 'galician', 'georgian', 'greek', 'gujarati', 'haitian', 'haitian creole', 'hausa', 'hawaiian', 'hebrew', 'hindi', 'hungarian', 'icelandic', 'indonesian', 'italian', 'javanese', 'kannada', 'kazakh', 'khmer', 'lao', 'latin', 'latvian', 'letzeburgesch', 'lingala', 'lithuanian', 'luxembourgish', 'macedonian', 'malagasy', 'malay', 'malayalam', 'maltese', 'mandarin', 'maori', 'marathi', 'moldavian', 'moldovan', 'mongolian', 'myanmar', 'nepali', 'norwegian', 'nynorsk', 'occitan', 'panjabi', 'pashto', 'persian', 'polish', 'portuguese', 'punjabi', 'pushto', 'romanian', 'sanskrit', 'serbian', 'shona', 'sindhi', 'sinhala', 'sinhalese', 'slovak', 'slovenian', 'somali', 'sundanese', 'swahili', 'swedish', 'tagalog', 'tajik', 'tamil', 'tatar', 'telugu', 'thai', 'tibetan', 'turkish', 'turkmen', 'ukrainian', 'urdu', 'uzbek', 'valencian', 'vietnamese', 'welsh', 'yiddish', 'yoruba']
    if audio_file_language == 'автоматически':
        audio_file_language = None

    # @title Name of the transcribed srt to generate, should be ending with `.srt`
    transcribed_srt_name = f"{output_file}.srt"  # @param {type:"string"}

    # @markdown Далее идут опциональные параметры

    # @markdown Если нет явных проблем с качеством транскрибации, то можно их не менять

    # @markdown --------------------

    # @markdown Чувствительность к тишине (иногда лучше работает значение 0.2)
    no_speech_threshold = 0.2  # @param {type:"slider", min:0, max:1, step:0.1}

    # @markdown Использовать контекст для распознования (логичнее использовать)
    condition_on_previous_text = "\u0414\u0430"  # @param = ["Да", "Нет"]
    condition_on_previous_text = condition_on_previous_text == "Да"

    def run_transcribe() -> str:
        """
        Функция для выполнения транскрибации
        :return: str - транскрибированный текст
        """
        result = model.transcribe(audio_file_name, language=audio_file_language,
                                  verbose=True, no_speech_threshold=no_speech_threshold,
                                  suppress_tokens="",
                                  initial_prompt="Текст на русском. Смешарки, Крош, Копатыч, Лосяш, Пин, Карыч, Совунья, Ёжик, Пин код, На старой железяке далеко не улетишь, На старой железяке далеко не уплывёшь, Хорош сидеть в овраге и ворочать железяки, Наука-штука хитрая мозги прокипятишь, Наука-штука крепкая все зубы обдерёшь, Вся жизнь твоя изменится, как только подберёшь самый пин код",
                                  condition_on_previous_text=condition_on_previous_text)

        # Создание списка для хранения сегментов транскрипции
        result_srt_list = []
        raw_text = []

        # Проход по каждому сегменту результата транскрипции
        for i in result['segments']:
            result_srt_list.append(
                srt.Subtitle(index=i['id'], start=timedelta(seconds=i['start']), end=timedelta(seconds=i['end']),
                             content=i['text'].strip()))
            raw_text.append(i['text'].strip())

        # Компоновка транскрипции в формат SRT
        composed_transcription = srt.compose(result_srt_list)

        # Печать транскрипции в формате SRT
        print(composed_transcription)

        # Запись транскрипции в файл SRT
        with open(transcribed_srt_name, 'w', encoding='utf-8') as f:
            f.write(composed_transcription)

        # Запись транскрипции в текстовый файл
        with open(transcribed_srt_name.split('.srt')[0] + ".txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(raw_text))

        res_string = " ".join(raw_text)
        return res_string

    # Проверка наличия MP3 файла для транскрибации
    if not os.path.isfile(audio_file_name):
        print_red('mp3 файл для транскрибации не найден')
    else:
        return run_transcribe()


# Для тестирования
if __name__ == "__main__":
    # input_video = "cache/0a7a288165c6051ebd74010be4dc9aa8.mp4"
    input_video = "cache/0b73cea7b2dfdd7048dd84b95f8b9b0e.mp4"
    output_audio = "0a7a288165c6051ebd74010be4dc9aa8_transcribe.mp3"
    output_file = "res.txt"
    print(convert_mp4_to_txt(input_video))
