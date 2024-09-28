import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer


def load_tokenizer():
    return BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')


def load_model():
    model = BertForSequenceClassification.from_pretrained('./bert_model')
    model.eval()
    return model


# Уникальные теги первого уровня, полученные в ходе анализа тренировочной выборки
# (см. baseline/bert.ipynb)
np_iab_tags = np.array(['Спорт',
                        'Дом и сад',
                        'Изобразительное искусство',
                        'Игры и головоломки',
                        'Музыка и аудио',
                        'Наука',
                        'События и достопримечательности',
                        'Карьера',
                        'Хобби и стиль',
                        'Путешествия',
                        'Отношения знаменитостей',
                        'Хобби и интересы',
                        'Экономика',
                        'Красота',
                        'Семья и отношения',
                        'Nan',
                        'Создание контента',
                        'Книги и литература',
                        'Бизнес и финансы',
                        'Личные финансы',
                        'Здоровый образ жизни',
                        'Игры',
                        'Информационные технологии',
                        'Стиль и красота',
                        'Медицинские направления',
                        'Транспорт',
                        'Новости и политика',
                        'Телевидение',
                        'Медицина',
                        'Фильмы и анимация',
                        'Массовая культура',
                        'Образование',
                        'Еда и напитки',
                        'Недвижимость',
                        'Компьютеры и цифровые технологии',
                        'Религия и духовность'])


def prepare_input(description, tokenizer, max_len):
    inputs = tokenizer(description, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
    return inputs['input_ids'], inputs['attention_mask']


def predict_tags(description, model, tokenizer, threshold=0.2, max_len=256):
    input_ids, attention_mask = prepare_input(description, tokenizer, max_len=max_len)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        if isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs.logits

    probabilities = torch.sigmoid(logits)
    predicted_tags = (probabilities > threshold).int()

    return np_iab_tags[predicted_tags.cpu().numpy().flatten() == 1]
