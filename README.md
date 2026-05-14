# Sign Language Classification (PJM)

Проект по классификации жестов польского жестового языка (PJM)
с использованием координат точек руки (hand landmarks) и архитектуры Bidirectional LSTM.

---

## Описание задачи

**Вход**: координаты 21 точки кисти (x, y, z) × 3 фрейма, полученные с помощью Google MediaPipe  
**Выход**: класс жеста (буква/знак PJM)  
**Тип задачи**: многоклассовая классификация  
**Метрики**: Accuracy, F1-macro  

Архитектура модели универсальна и может быть адаптирована для других жестовых языков
(РЖЯ, ASL и др.) при наличии аналогичных данных.

---

## Структура проекта

```
sign-language-classification/
│
├── data/
│ └── landmarks.csv # исходный датасет (PJM hand landmarks)
│
├── results/ # артефакты обучения (графики, метрики, модель)
│ ├── class_distribution.png
│ ├── landmark_example.png
│ ├── training_history.png
│ ├── cm_random_forest.png
│ ├── cm_bilstm.png
│ ├── model_comparison.png
│ ├── metrics_report.csv
│ └── best_model.keras
│
├── utils.py # вспомогательные функции (предобработка, модель, оценка)
├── main.ipynb # основной ноутбук с полным пайплайном
├── README.md
└── requirements.txt
```

---

## Как запустить

**1. Клонировать репозиторий**
```bash
git clone https://github.com/username/sign-language-classification.git
cd sign-language-classification
```

**2. Установить зависимости**
```bash
pip install -r requirements.txt
```

**3. Поместить датасет**

Скачать датасет с [Kaggle](https://www.kaggle.com/datasets/kacperjarosik1/polish-sign-language-google-landmarks-csv-data)
и положить файл `landmarks.csv` в папку `data/`.

**4. Запустить ноутбук**
```bash
jupyter notebook main.ipynb
```

Запускать ячейки **последовательно сверху вниз** по секциям.

---

## Результаты

| Модель                   | F1-macro |
|--------------------------|----------|
| Random Forest (baseline) | —        |
| SVM (baseline)           | —        |
| **BiLSTM (наша модель)** | —        |

> Значения будут заполнены после обучения.

---

## Архитектура BiLSTM
```
Input (3, 63)
→ Bidirectional LSTM (units=64)
→ Dropout (0.3)
→ Dense (128, ReLU)
→ Dropout (0.3)
→ Dense (n_classes, Softmax)
```

---

## Датасет

- **Источник**: [Kaggle — Polish Sign Language (PJM) Hand Landmarks](https://www.kaggle.com/datasets/kacperjarosik1/polish-sign-language-google-landmarks-csv-data)
- **Язык**: Польский жестовый язык (PJM)
- **Признаки**: координаты 21 точки руки × 3 фрейма × 3 оси (x, y, z)

---

## Стек технологий

- Python 3.10+
- TensorFlow / Keras
- scikit-learn
- pandas, numpy
- matplotlib, seaborn