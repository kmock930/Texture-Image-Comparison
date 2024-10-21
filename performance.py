import pandas as pd;
'''
@summary Function that displays the prediction performance in a well-formatted confusion matrix
@param y_pred numpy.array: an array of predicted labels
@param y_test numpy.array: an array of actual labels
@return pandas.core.frame.DataFrame The resulting confusion matrix in a Pandas DataFrame format
'''
def printConfMtx(y_pred, y_test):
    pandas_y_actual = pd.Series(y_test, name='Actual');
    pandas_y_pred = pd.Series(y_pred, name='Predicted');
    confMtx = pd.crosstab(
        pandas_y_actual, 
        pandas_y_pred, 
        rownames=['Actual'], 
        colnames=['Predicted'], 
        margins=True);
    confMtx.fillna(0, inplace=True);
    return confMtx;