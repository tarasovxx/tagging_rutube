from transformers import T5ForConditionalGeneration, GPT2Tokenizer
import torch

def get_summarize_text(text:str) -> str:
    '''
    Функция выдачи суммаризации по тексту
    :param text:
    :return:
    '''
    # Загрузка токенизатора и модели
    tokenizer = GPT2Tokenizer.from_pretrained("RussianNLP/FRED-T5-Summarizer", eos_token="</s>")
    model = T5ForConditionalGeneration.from_pretrained("RussianNLP/FRED-T5-Summarizer")

    # Форматирование текста для модели
    input_text = rf'''<LM> Сократи текст.\n ({text})'''

    # Определение устройства для выполнения (GPU или CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Токенизация входного текста
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Генерация суммаризации
    outputs = model.generate(
        inputs,
        max_length=600,  # Максимальная длина выходного текста
        num_beams=5,  # Количество лучей для поиска (beam search)
        early_stopping=True,  # Остановка, когда достаточно информации для суммаризации
        no_repeat_ngram_size=3,  # Избегание повторения фраз
        top_p=0.9  # Стратегия выборки для улучшения разнообразия
    )

    # Декодирование сгенерированного текста
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Вывод суммаризации
    return summary