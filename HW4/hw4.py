#######         Importing required libraries          #######
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics, cluster
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


#######           Defining Purity Function            #######
def purity_score(y_true, y_pred):
    # compute contingency/confusion matrix
    confusion_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return round(np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix), 4)


#######             Reading the ARFF file             #######
data = loadarff('pd_speech.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')
X, y = df.drop('class', axis=1), df['class']


#######                  Normalizing                  #######
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)


#######                    K-Means                    #######
kmeans_algo0 = cluster.KMeans(n_clusters=3, random_state=0)
kmeans_algo1 = cluster.KMeans(n_clusters=3, random_state=1)
kmeans_algo2 = cluster.KMeans(n_clusters=3, random_state=2)

# learning the model
kmeans_model0 = kmeans_algo0.fit(X_normalized)
kmeans_model1 = kmeans_algo1.fit(X_normalized)
kmeans_model2 = kmeans_algo2.fit(X_normalized)

# getting the predicted labels
y_pred0 = kmeans_model0.labels_
y_pred1 = kmeans_model1.labels_
y_pred2 = kmeans_model2.labels_


#######                     Ex 1                      #######
print("Silhouette score of Solution 1:", round(metrics.silhouette_score(X_normalized, y_pred0), 4))
print("Purity of Solution 1:", purity_score(y, y_pred0))
print("Silhouette score of Solution 2:", round(metrics.silhouette_score(X_normalized, y_pred1), 4))
print("Purity of Solution 2:", purity_score(y, y_pred1))
print("Silhouette score of Solution 3:", round(metrics.silhouette_score(X_normalized, y_pred2), 4))
print("Purity of Solution 3:", purity_score(y, y_pred2))


#######                     Ex 3                      #######
# variance by feature of normalized data
variance = X_normalized.var(axis=0)
# get the two highest features based on variance
two_highest_variance = variance.argsort()[-2:][::-1]

# plotting
plt.figure(figsize=(14, 5))

plt.subplot(121)
y_values = np.array([int(i) for i in y.values])
plt.scatter(X_normalized[:, two_highest_variance[0]], X_normalized[:, two_highest_variance[1]], c=y_values)
plt.legend(handles=[plt.scatter([], [], label='Healthy', c='#7F00FF'),
                    plt.scatter([], [], label='Sick', c='#FFF333')])
plt.title("Original Parkinson diagnoses")
plt.xlabel(X.columns[two_highest_variance[0]])
plt.ylabel(X.columns[two_highest_variance[1]])

plt.subplot(122)
plt.scatter(X_normalized[:, two_highest_variance[0]], X_normalized[:, two_highest_variance[1]], c=y_pred0)
plt.legend(handles=[plt.scatter([], [], label='Cluster 0', c='#7F00FF'),
                    plt.scatter([], [], label='Cluster 1', c='#006666'),
                    plt.scatter([], [], label='Cluster 2', c='#FFF333')])
plt.title("K-Means clustering (random_state = 0)")
plt.xlabel(X.columns[two_highest_variance[0]])
plt.ylabel(X.columns[two_highest_variance[1]])

plt.show()


#######                     Ex 4                      #######
pca = PCA()
pca.fit(X_normalized)
i = 1
for var in np.cumsum(pca.explained_variance_ratio_):
    if var > 0.8:
        break
    i += 1
print("Number of principal components:", i)