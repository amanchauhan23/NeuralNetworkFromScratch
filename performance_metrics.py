import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def calculate_metrics(y_true, y_pred, loss_vs_e, show_cm=True):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    F1 = f1_score(y_true, y_pred, average="macro")

    # plot loss vs epocs
    loss_vs_e = np.array(loss_vs_e)
    fig = px.line(x=loss_vs_e[:, 0], y=loss_vs_e[:, 1], labels={'x': "Epochs", 'y': 'Error'}, title="Error vs Epochs")
    fig.show()

    # confusion matrix plot
    fig = px.imshow(cm, text_auto=True)
    fig.show()
    return cm, acc, F1