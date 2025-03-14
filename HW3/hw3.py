#######         Importing required libraries          #######
from scipy.io.arff import loadarff
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import warnings

def warn(*args, **kwargs): pass
warnings.warn = warn


#######            Reading the ARFF file              #######
data = loadarff('kin8nm.arff')
df = pd.DataFrame(data[0])


#######      Creating the training-testing split      #######
X, y = df.drop('y', axis=1), df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)


#######     Creating and asserting the regressors     #######

# Creating the regressors
ridge = Ridge(alpha=0.1)
mlp1 = MLPRegressor(hidden_layer_sizes=(10,10), max_iter=500, early_stopping=True, random_state=0, activation='tanh')
mlp2 = MLPRegressor(hidden_layer_sizes=(10,10), max_iter=500, early_stopping=False, random_state=0, activation='tanh')

# Training the regressors
ridge.fit(X_train, y_train)
mlp1.fit(X_train, y_train)
mlp2.fit(X_train, y_train)

# Predicting the values
ridge_pred = ridge.predict(X_test)
mlp1_pred = mlp1.predict(X_test)
mlp2_pred = mlp2.predict(X_test)


#######     Calculating the MAE for each regressor    #######
ridge_mae = metrics.mean_absolute_error(y_test, ridge_pred)
mlp1_mae = metrics.mean_absolute_error(y_test, mlp1_pred)
mlp2_mae = metrics.mean_absolute_error(y_test, mlp2_pred)


#######                      Ex 4                     #######
print("MAE for Ridge Regression: ", round(ridge_mae, 3))
print("MAE for MLP with early stopping: ", round(mlp1_mae, 3))
print("MAE for MLP without early stopping: ", round(mlp2_mae, 3))


#######                      Ex 5                     #######
residues = [abs(y_test - ridge_pred), abs(y_test - mlp1_pred), abs(y_test - mlp2_pred)]

# Boxplot
sns.boxplot(residues)
plt.xticks([0, 1, 2], ['Ridge', 'MLP1', 'MLP2'])
plt.show()

# Histogram
plt.hist(residues, bins=10, label=['Ridge', 'MLP1', 'MLP2'])
plt.legend()
plt.xlabel('Residues')
plt.ylabel('Frequency')
plt.show()


#######                      Ex 6                     #######
print("MLP1 iterations: ", mlp1.n_iter_)
print("MLP2 iterations: ", mlp2.n_iter_)