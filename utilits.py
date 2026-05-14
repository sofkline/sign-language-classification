import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Bidirectional, LSTM, Dense, Dropout, Input
)
from tensorflow.keras.optimizers import Adam

#================ PREPROCESSING =====================

def load_data(path: str) -> pd.DataFrame:
    """Загрузка датасета из CSV."""
    return pd.read_csv(path)

def encode_labels(df: pd.DataFrame, label_col: str):
    """
    Кодирует строковые метки классов в числа.
    Возвращает encoded labels и сам encoder (нужен для обратного декодирования).
    """
    le = LabelEncoder()
    encoded = le.fit_transform(df[label_col])
    return encoded, le

def normalize_landmarks(X: np.ndarray) -> np.ndarray:
    """
    Нормализация координат landmarks по каждому признаку (MinMax).
    Вход: (samples, timesteps * features)
    """
    scaler = MinMaxScaler()
    X_flat = X.reshape(X.shape[0], -1)
    X_scaled = scaler.fit_transform(X_flat)
    return X_scaled, scaler

def reshape_for_lstm(X: np.ndarray, timesteps: int = 3) -> np.ndarray:
    """
    Преобразует плоский вектор признаков в формат (samples, timesteps, features).
    63 признака = 21 точка × 3 координаты (x, y, z).
    """
    features = X.shape[1] // timesteps
    return X.reshape(X.shape[0], timesteps, features)

def get_splits(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """Разбивка на train / val / test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_relative,
        random_state=random_state, stratify=y_train
    )
    return X_train, X_val, X_test, y_train, y_val, y_test



#================ MODEL =====================

def build_bilstm(
    input_shape: tuple,   # (timesteps, features) — например (3, 21)
    n_classes: int,
    lstm_units: int = 64,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001
):
    """
    Bidirectional LSTM для классификации жестов.
    input_shape: (timesteps, features)
    n_classes: количество жестов/букв в датасете
    """
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(lstm_units, return_sequences=False)),
        Dropout(dropout_rate),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_baseline_mlp(input_dim: int, n_classes: int):
    """
    Простой MLP — альтернативный нейросетевой baseline
    для сравнения с BiLSTM (без учёта временно́й структуры).
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model



#================ EVALUATE =====================

def plot_training_history(history, save_path='results/training_history.png'):
    """Графики loss и accuracy по эпохам (train vs val)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['loss'], label='Train')
    axes[0].plot(history.history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history.history['accuracy'], label='Train')
    axes[1].plot(history.history['val_accuracy'], label='Val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names,
                          save_path='results/confusion_matrix.png'):
    """Матрица ошибок с названиями классов."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def print_metrics(y_true, y_pred, class_names):
    """Полный отчёт по метрикам + macro F1."""
    print(classification_report(y_true, y_pred, target_names=class_names))
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Macro F1-score: {f1:.4f}")
    return f1