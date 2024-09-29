# tagging_rutube

## Users Jokers 🌠

Мы предлагаем систему тегирования видео на основе видео контента, названия и описания видео. Тегирование происходит по
универсальному списку тегов для web платформ, широко затрагивающему различные тематики и подтематики. В решении
участников протегированное видео может иметь тег родительской категории и тег подкатегории, соответствующий
родительской. Видео может содержать несколько тегов из различных тематик.

![Демонстрация работы](docs/demo.gif)

### Технические особенности

- BERT для предсказания тегов по описанию/транскрибации (torch, BERT)
- Использование метаинформации о видео для предсказания тегов (Mock DB request)
- Разработаны модули для скачивания и транскрибации видео (Whisper)

# Установка
1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/tarasovxx/tagging_rutube
   cd tagging_rutube
   ```

2. Убедитесь, что у вас установлен Python версии 3.9 и выше. Создайте и активируйте виртуальное окружение:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Установите зависимости:
4. Базовые зависимости для работы веб-интерфейса, обучения и инференса модели

   ```bash
   pip install -r requirements_base.txt
   ```

5*. Все зависимости для дополнительных модулей с транскрибацией и скачиванием видео
   ```bash
   pip install -r requirements.txt
   ```

# Обучение модели

Обучение BERT-модели происходит в файле [bert.ipynb](baseline/bert.ipynb), 
где все подробно расписано по шагам, включая получение самого файла модели.

Папка `model` кладется в корень проекта, оттуда приложение будет брать модель

# Про файлы

- baseline - папка с бейзлайном и ноутбуком с обучением BERT
- app.py - веб-приложение на стримлите
- category_mapping.py - предсказание по категории
- handler.py - конвертация mp4 в текст
- model_loader.py - модуль по работе с BERT-моделью
- summarization.py - суммаризация текста
- utils.py - прочие функции по работе с Rutube-видео

# Запуск
```bash
streamlit run app.py
```

# Стек используемых технологий:

## Инструменты разработки
- Python3
- pytorch
- transformers
- Streamlit

## Пример использования технологий
В нашем проекте мы используем **Whisper** для распознавания речи и преобразования аудиофайлов в текст. Затем с помощью **Transformers** мы обрабатываем текст для выполнения задач суммаризации и тематического анализа. Все результаты визуализируются с помощью **Streamlit**.

Этот стек технологий позволяет нам эффективно обрабатывать и анализировать текстовые и аудиоданные, а также предоставлять пользователям удобные инструменты для взаимодействия с результатами анализа.

## Docker
Мы используем Docker для того, чтобы каждый мог легко развернуть наше приложение на любой платформе. Docker обеспечивает
кроссплатформенность и упрощает процесс развертывания.

### Локальный запуск с использованием Docker

1. Склонируйте репозиторий:
    ```bash
    git clone https://github.com/tarasovxx/tagging_rutube
   ```
2. Перейдите в директорию проекта:
   ```bash
   cd tagging_rutube
   ```
3. Постройте Docker-образ:
   ```bash
   docker build -t tagging_rutube .
   ```
4. Запустите контейнер:
   ```bash
   docker run -p 8501:8501 tagging_rutube
   ```

# Разработчики

| Имя                | Роль             | Контакт                                  |
|--------------------|------------------|------------------------------------------|
| Константин Балцат  | Data Analysis    | [t.me/baltsat](https://t.me/baltsat)     |
| ---                | ---              | ---                                      |
| Артем Тарасов      | Full Stack       | [t.me/tarasovxx](https://t.me/tarasovxx) |
| ---                | ---              | ---                                      |
| Харламов Александр | Machine Learning | [t.me/Wignorbo](https://t.me/@Wignorbo)  |
| ---                | ---              | ---                                      |
| Даниил Галимов     | Data Analysis    | [t.me/Dan_Gan](https://t.me/Dan_Gan)     |
| ---                | ---              | ---                                      |
