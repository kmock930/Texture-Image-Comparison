import pandas as pd;
import constants;
import numpy as np;
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score;
import matplotlib.pyplot as plt;
from setup import displayImage;
import random;

'''
@summary Function that displays the prediction performance in a well-formatted confusion matrix
@param y_pred numpy.array: an array of predicted labels
@param y_test numpy.array: an array of actual labels
@return pandas.core.frame.DataFrame The resulting confusion matrix in a Pandas DataFrame format
'''
def printConfMtx(y_pred: np.ndarray, y_test: np.ndarray):
    # Convert contents in y_pred
    # 0: YOR; 1: CAL
    y_new_pred = [];
    for pred in y_pred:
        if pred == 0:
            y_new_pred.append(constants.YOR);
        elif pred == 1:
            y_new_pred.append(constants.CAL);

    # Convert contents in y_test 
    # 0: YOR; 1: CAL
    y_new_test = []
    for test in y_test:
        if test == 0:
            y_new_test.append(constants.YOR);
        elif test == 1:
            y_new_test.append(constants.CAL);
    
    y_new_test = np.asarray(y_new_test); # convert it to a numpy array

    # Construct the confusion matrix
    confMtx = confusion_matrix(
        y_true=y_new_test,
        y_pred=y_new_pred
    )

    class_names = [constants.YOR, constants.CAL]
    
    # Convert to DataFrame with class names
    confMtx_df = pd.DataFrame(confMtx, index=class_names, columns=class_names)
    
    return confMtx_df


def getAccuracy(y_pred: np.ndarray, y_test: np.ndarray):
    return accuracy_score(
        y_true=y_test,
        y_pred=y_pred
    );

def getPrecision(y_pred: np.ndarray, y_test: np.ndarray):
    return precision_score(
        y_true=y_test,
        y_pred=y_pred
    );

def getRecall(y_pred: np.ndarray, y_test: np.ndarray):
    return recall_score(
        y_true=y_test,
        y_pred=y_pred
    );

def getF1(y_pred: np.ndarray, y_test: np.ndarray):
    return f1_score(
        y_true=y_test,
        y_pred=y_pred
    );

def visualizeMisclassified(X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray):
    # assume the lengths of all the numpy arrays from the parameters are all equal
    # assume each image samples in X_test are in the same order as their actual labels in y_test and in the predictions y_pred arrays
    evalArr = y_test == y_pred;
    misclassifiedInds = [];
    for evalResInd in range(0, len(evalArr)):
        evalRes = evalArr[evalResInd];
        if (evalRes == False):
            misclassifiedInds.append(evalResInd);
    if (len(misclassifiedInds) == 0):
        print("No misclassified images.");
    else:
        randMisclassifiedInd = random.randint(0, len(misclassifiedInds)-1);
        category: str = "";
        if (y_test[randMisclassifiedInd] == 0):
            category = constants.YOR;
        else:
            category = constants.CAL;
        displayImage(
            img_array=X_test[randMisclassifiedInd],
            category=category,
            isBefore=False
        );
        print(f"Number of misclassified images for this model: {len(misclassifiedInds)}");