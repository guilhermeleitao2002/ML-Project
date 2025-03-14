#######         Importing required libraries          #######
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.io.arff import loadarff
import warnings

def warn(*args, **kwargs): pass
warnings.warn = warn
 

#######            Reading the ARFF file              #######
# Load the data
data = loadarff('pd_speech.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')


#######            Folding and Classifiers            #######
X, y = df.drop('class', axis=1), df['class']
cv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
# Creating the classifiers
predictor_kNN = KNeighborsClassifier(weights='uniform', n_neighbors=5, metric='euclidean')
predictor_nb = GaussianNB()


#######   Running classifier and attesting results    #######
cm_kNN, cm_nb, kNN_acc, nb_acc = [], [], [], []

for train_k, test_k in cv.split(X, y):
    # Getting the training and testing splits
    X_train, X_test = X.iloc[train_k], X.iloc[test_k]
    y_train, y_test = y.iloc[train_k], y.iloc[test_k]
    # Training the classifiers
    predictor_kNN.fit(X_train, y_train)
    predictor_nb.fit(X_train, y_train)
    # Predicting the classes
    y_pred_kNN = predictor_kNN.predict(X_test)
    y_pred_nb = predictor_nb.predict(X_test)
    # Computing the confusion matrices
    cm_kNN.append(np.array(confusion_matrix(y_test, y_pred_kNN, labels=['0', '1'])))
    cm_nb.append(np.array(confusion_matrix(y_test, y_pred_nb, labels=['0', '1'])))
    # Computing the accuracy
    kNN_acc.append(round(metrics.accuracy_score(y_test, y_pred_kNN), 3))
    nb_acc.append(round(metrics.accuracy_score(y_test, y_pred_nb), 3))

cm_kNN = np.sum(cm_kNN, axis=0)
cm_nb = np.sum(cm_nb, axis=0)

# Creating the confusion matrices' plot
confusion_knn = pd.DataFrame(cm_kNN, index=['Healthy', 'Sick', ], columns=['Predicted Healthy', 'Predicted Sick'])
confusion_nb = pd.DataFrame(cm_nb, index=['Healthy', 'Sick', ], columns=['Predicted Healthy', 'Predicted Sick'])


#######                     Ex 5                      #######
# KNN confusion matrix
sns.heatmap(confusion_knn, annot=True, fmt='g')
plt.title('KNN Confusion Matrix')
plt.show()

# Naive Bayes confusion matrix
sns.heatmap(confusion_nb, annot=True, fmt='g')
plt.title('Naive Bayes Confusion Matrix')
plt.show()


#######                     Ex 6                      #######
# Performing the t-test
res = stats.ttest_rel(kNN_acc, nb_acc, alternative='greater')
print("knn accuracy: ", kNN_acc)
print("nb accuracy: ", nb_acc)
# Outputting the p-value
print("p1>p2? pval=",np.round(res.pvalue, 3))