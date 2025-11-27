from config import SAMPLING_RATE, N_MELS
import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix, precision_recall_curve

def plot_mel_spectrogram(y, sr=SAMPLING_RATE, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax);

def plot_mel_spectrogram_from_dB(S_dB, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    librosa.display.specshow(S_dB, sr=SAMPLING_RATE, x_axis='time', y_axis='mel', ax=ax);

def show_precision_recall_curve(y_true, y_proba, punto_corte=0.5, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    prec, rec, thr = precision_recall_curve(y_true, y_proba)

    ax.plot(thr, rec[:-1], label='recall')
    ax.plot(thr, prec[:-1], label='precision')
    ax.set_xlabel('Punto de corte')
    ax.vlines(punto_corte, 0, 1, colors='red', linestyles='dashed', label='corte')
    ax.grid()
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.legend()
    
    return ax

def show_confusion_matrix(y_true, y_pred, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Matriz de Confusión')
    return ax

def show_importances_heatmap(importances, n_mels=N_MELS, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    n_frames = importances.shape[0] // n_mels

    importance_img = importances.reshape((n_mels, n_frames))
    importance_img = np.flipud(importance_img)

    sns.heatmap(importance_img, cmap='inferno', ax=ax, cbar=False)
    ax.set_xlabel('Tiempo (frames)')
    ax.set_ylabel('Bandas Mel')
    ax.set_title('Mapa de importancia de features del modelo')
    return ax
