import data_sets
import cascade_classifier
import cv2
import numpy as np
import pandas as pd

X_train,Y_train,X_test,Y_test = data_sets.loadDataATT()




clf = cascade_classifier.train_classifer(X_train, Y_train)

test_Image = 5
pred_id, _ = clf.predict(X_test[test_Image])

print("predicted target :" + str(pred_id))
print("prediction confidence of :" + str(_))
print("============")
print("actual target :" + str(Y_test[test_Image]))

#---------------------
print("============")


Y_score = np.empty(len(X_test))
for i in range(len(X_test)):
    pred_id, _ = clf.predict(X_test[i])
    Y_score[i] = pred_id
    
print(Y_test)
print(Y_score)
    
#from sklearn.metrics import average_precision_score
#average_precision = average_precision_score(Y_test, Y_score)

#print('Average precision-recall score: {0:0.2f}'.format(
      #average_precision))
      
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

#disp = plot_precision_recall_curve(clf, X_test, Y_test)
#disp.ax_.set_title('2-class Precision-Recall curve: '
                   #'AP={0:0.2f}'.format(average_precision))

def print_comparison_result(y_test, y_predict):
    # Re-align numbering for result printing
    y_test = np.array(y_test)
    y_test = y_test - 1
    y_predict = y_predict - 1

    cm = confusion_matrix(y_test, y_predict, labels=range(len(y_test)))
    #df = pd.DataFrame(cm, columns = Y_train, index = Y_train)
    print ("\n==================RESULT==================")
    print ("Confusion Matrix: ")
    #print (df)
    print ("Classification Report: ")

    #print(classification_report(y_test, y_predict, target_names=TARGET_NAMES))
    print(classification_report(y_test, y_predict))
    
    print( """How to comprehend the report:
    - Recall value: "Given a true face of person X, how likely does the classifier detect it is X?
    - Precision value: "If the classifier predicted a face person X, how likely is it to be correct?
    """)
    
print_comparison_result(Y_test,Y_score)