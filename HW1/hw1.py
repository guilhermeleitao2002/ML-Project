#######         Importing required libraries          #######
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn import tree, metrics
import matplotlib.pyplot as plt


#######            Reading the ARFF file              #######
data = loadarff('pd_speech.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')


#######      Creating the training-testing split      #######
X, y = df.drop('class', axis=1), df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=1)


####### Feature ranking based on discriminative power #######
dp = mutual_info_classif(X, y, random_state=1)
dict1, count = {}, 0

for feature in X_train.columns.values:
    dict1[feature] = dp[count]
    count += 1

dict2 = dict(sorted(dict1.items(), key=lambda item: item[1], reverse=True))
features = list(dict2.keys())


#######   Running classifier and attesting results    #######
train, test = [], []
predictor = tree.DecisionTreeClassifier(random_state=1)

for n in [5, 10, 40, 100, 250, 700]:
    predictor.fit(X_train[features[0:n]], y_train)
    y_pred1 = predictor.predict(X_test[features[0:n]])
    y_pred2 = predictor.predict(X_train[features[0:n]])
    test.append(round(metrics.accuracy_score(y_test, y_pred1), 3))
    train.append(round(metrics.accuracy_score(y_train, y_pred2), 3))


#######                Creating plot                  #######
plt.plot([5, 10, 40, 100, 250, 700], test)
plt.plot([5, 10, 40, 100, 250, 700], train)
plt.xlabel('nº features')
plt.ylabel('accuracy')
plt.show()