import cv2


def horizontal_flip(img, flag):
    '''
    The function cv::flip flips a 2D array around vertical, horizontal, or both axes.
    In questo caso viene effettuato un flip dell'immagine di tipo orizzontale
    '''
    if flag:
        return cv2.flip(img, 1)
    else:
        return img


def vertical_flip(img, flag):
    '''
    The function cv::flip flips a 2D array around vertical, horizontal, or both axes.
    In questo caso viene effettuato un flip dell'immagine di tipo verticale
    '''
    if flag:
        return cv2.flip(img, 0)
    else:
        return img


import random


def rotation(img, angle):
    '''
    L'immagine viene ruotata di un grado pari a: angle
    '''
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img


import numpy as np


def brightness(img, low, high):
    '''
    La luuminosità dell'immagine viene fatta variare a seconda dei valori dei parametri: low e high
    '''
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def zoom(img, value):
    '''
    Viene effettuato uno "zoom" dell'immagine
    '''
    if value > 1 or value < 0:
        print('Value for zoom should be less than 1 and greater than 0')
        return img
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    return img


from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_score,f1_score
import matplotlib.pyplot as plt
import seaborn as sns


def performance(y_val, y_pred, model, isDummy):
    '''
    Vengono calcolate le performance del modello model e vengono generati:
        - classification report
        - calcolo metriche (acc, prec, rec, f1)
        - Groundtruth vs Predicted
        - Confusion matrix
        - Best parameters
        - ROC value
    '''
    # Classification report
    print(classification_report(y_val, y_pred))

    # Salva le metriche
    acc = str(accuracy_score(y_val, y_pred))
    prec = str(precision_score(y_val, y_pred))
    rec = str(recall_score(y_val, y_pred))
    f1 = str(f1_score(y_val, y_pred))

    print('\n')
    print('-------------------------------------')

    # Groundtruth vs Predicted
    print('GROUDTRUTH')
    print(np.array(y_val))
    print('PREDICTED')
    print(y_pred)

    print('\n')
    print('-------------------------------------')

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
    plt.xticks(ticks=np.arange(2) + 0.5, labels=['Non_difettose', 'Difettose'])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=['Non_difettose', 'Difettose'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print('\n')
    print('-------------------------------------')

    # Best parameters
    if isDummy:
        best_est = 0
        best_score = 0
        best_para = 0
    else:
        best_est = model.best_estimator_
        best_score = model.best_score_
        best_para = model.best_params_

    print("\n The best estimator across ALL searched params:\n", best_est)
    print("\n The best score across ALL searched params:\n", best_score)
    print("\n The best parameters across ALL searched params:\n", best_para)

    print('\n')
    print('-------------------------------------')

    # ROC value
    roc = roc_auc_score(y_val, y_pred)
    print("Valore della ROC:", roc)

    return acc, prec, rec, f1, best_est, best_score, best_para, roc, cm

import pandas as pd

def generate_df(image_dir, label):
    filepaths = pd.Series(list(image_dir.glob(r'*.bmp')), name='Filepath').astype(str)
    labels = pd.Series(label, name='Label', index=filepaths.index)
    df = pd.concat([filepaths, labels], axis=1)

    return df


from sklearn.metrics import ConfusionMatrixDisplay


def plot_conf_matrix(names, classifiers, nrows, ncols, fig_a, fig_b):
    '''
    Plots confusion matrices in a subplots.

    Args:
        names : list of names of the classifier
        classifiers : list of classification algorithms
        nrows, ncols : number of rows and rows in the subplots
        fig_a, fig_b : dimensions of the figure size
    '''

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_a, fig_b))

    i = 0
    for cm, ax in zip(classifiers, axes.flatten()):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues, ax = ax, colorbar=False)
        ax.title.set_text(names[i])
        i = i + 1

    plt.tight_layout()
    plt.show()


from sklearn.metrics import recall_score, accuracy_score,roc_curve, auc


def roc_auc_curve(names, pred_list, y_val):
    '''
    Given a list of classifiers, this function plots the ROC curves
    '''
    plt.figure(figsize=(12, 8))

    for name, prd in zip(names, pred_list):

        fpr, tpr, thresholds = roc_curve(y_val, prd)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=3, label= name +' ROC curve (area = %0.2f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (ROC) curves', fontsize=20)
        plt.legend(loc="lower right")


def score_summary(names, pred_list, time_duration, y_val):
    '''
    Given a list of classifiers, this function calculates:
        - Recall
        - Precision
        - Accuracy
        - F1
        - ROC_AUC
        - Time
    '''

    cols=["Classifier", "Recall", "Precision", "Accuracy", "F1", "ROC_AUC", "Time"]
    data_table = pd.DataFrame(columns=cols)

    for name, prd, time in zip(names, pred_list, time_duration):

        accuracy = accuracy_score(y_val, prd)


        fpr, tpr, thresholds = roc_curve(y_val, prd)
        roc_auc = auc(fpr, tpr)

        # confusion matric, cm
        cm = confusion_matrix(y_val, prd)

        # recall: TP/(TP+FN)
        recall = cm[1,1]/(cm[1,1] +cm[1,0])

        # precision: TP/(TP+FP)
        precision = cm[1,1]/(cm[1,1] +cm[0,1])

        # F1 score: TP/(TP+FP)
        f1 = 2*recall*precision/(recall + precision)

        df = pd.DataFrame([[name, recall, precision, accuracy*100, f1, roc_auc, time]], columns=cols)
        data_table = data_table.append(df)

    return(np.round(data_table.reset_index(drop=True), 2))


def metrics(df_metrics):

    """
        Calcolo delle metriche di accuracy, precision, f1, recall e auc per ciascun modello.

    """

    # accuracy
    accuracy = df_metrics[df_metrics['rank_test_accuracy'] == 1]['mean_test_accuracy'].values[0]

    # precision
    precision = df_metrics[df_metrics['rank_test_precision'] == 1]['mean_test_precision'].values[0]

    # f1
    f1 = df_metrics[df_metrics['rank_test_f1'] == 1]['mean_test_f1'].values[0]

    # recall
    recall = df_metrics[df_metrics['rank_test_recall'] == 1]['mean_test_recall'].values[0]

    # roc_auc
    roc_auc = df_metrics[df_metrics['rank_test_roc_auc'] == 1]['mean_test_roc_auc'].values[0]

    return accuracy, precision, f1, recall, roc_auc



import pickle


def pickle_dump(model, filename):
    '''
    Salva il modello in path
    '''
    path = "../Models/"
    with open(path + str(filename), "wb") as file:
        pickle.dump(model, file)