# Распознавание Русской Речи
[![Heard by 🎙️](https://a.b-b.top/badge.svg?repo=voice_to_text&label=Heard&background_color=ff5722&background_color2=ff7043&utm_source=github&utm_medium=readme&utm_campaign=badge)](https://a.b-b.top)

Приложение для распознавания русской речи в реальном времени с использованием искусственного интеллекта.

![Russian Speech Recognition](./2025-06-04%2012_01_41.png)


## Описание

Это приложение позволяет преобразовывать русскую речь в текст в реальном времени. Оно использует модель Wav2Vec2 для точного распознавания речи и предоставляет удобный графический интерфейс для управления процессом записи.

## Требования

- Python 3.8 или выше
- PyQt5
- PyTorch
- Transformers
- SoundDevice
- NumPy
- SciPy

## Установка

### Вариант 1: Установка через pip

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/russian-speech-recognition.git
cd russian-speech-recognition
```

2. Установите необходимые зависимости:
```bash
pip install -r requirements.txt
```

### Вариант 2: Установка через Conda

1. Создайте новое окружение Conda:
```bash
conda create -n speech_rec python=3.8
conda activate speech_rec
```

2. Установите PyTorch через Conda:
```bash
conda install pytorch cpuonly -c pytorch
```

3. Установите остальные зависимости:
```bash
conda install -c conda-forge pyqt
conda install -c conda-forge transformers
conda install -c conda-forge sounddevice
conda install numpy scipy
```

4. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/russian-speech-recognition.git
cd russian-speech-recognition
```

## Использование

1. Запустите приложение:
```bash
python run.py
```

2. Основные функции:
   - Нажмите кнопку "Record (R)" или клавишу R для начала записи
   - Нажмите кнопку "Pause (P)" или клавишу P для паузы
   - Нажмите кнопку "Stop (S)" или клавишу S для остановки записи
   - Используйте "Copy Text (Ctrl+C)" для копирования текста
   - Используйте "Clear Text (Ctrl+L)" для очистки текста

## Горячие клавиши

- R - Начать запись
- P - Пауза/Продолжить
- S - Остановить запись
- Ctrl+C - Копировать текст
- Ctrl+L - Очистить текст
- Esc - Выход

## Особенности

- Распознавание речи в реальном времени
- Поддержка длинных записей
- Возможность паузы и возобновления записи
- Автоматическое сохранение текста
- Удобный интерфейс в стиле Windows
- Поддержка горячих клавиш

## Технические детали

Приложение использует модель Wav2Vec2 (jonatasgrosman/wav2vec2-large-xlsr-53-russian) для распознавания речи. Модель оптимизирована для работы на CPU и обеспечивает высокую точность распознавания русской речи.

## Устранение неполадок

1. Если приложение не запускается:
   - Убедитесь, что все зависимости установлены
   - Проверьте версию Python (должна быть 3.8 или выше)
   - Проверьте наличие микрофона и его работоспособность
   - При использовании Conda убедитесь, что окружение активировано

2. Если распознавание работает неточно:
   - Говорите четко и в нормальном темпе
   - Убедитесь, что микрофон правильно настроен
   - Проверьте уровень шума в помещении

## Лицензия

MIT License

## Поддержка

Если у вас возникли проблемы или есть предложения по улучшению, пожалуйста, создайте issue в репозитории проекта. 
