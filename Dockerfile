FROM python:3.10-slim

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV APP_HOME=/home/app

WORKDIR $APP_HOME

# Установка ffmpeg и очистка apt cache
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Создание appuser и выдача прав пользователю
RUN useradd -m -s /bin/bash -G sudo appuser && echo "appuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Создание директории и выдача прав
RUN mkdir -p /home/app/cache

# Копирование requirements.txt и установка зависимостей
COPY requirements.txt $APP_HOME
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . $APP_HOME

# Обеспечение право собственности на все файлы пользователя приложения
RUN chown -R appuser:appuser /home/app && chmod -R 755 /home/app

# Смена на пользователя appuser
USER appuser

# Запуск приложения
CMD ["streamlit", "run", "app.py"]