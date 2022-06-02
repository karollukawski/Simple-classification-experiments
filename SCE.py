from scipy import rand
import sklearn as sk
from sklearn import datasets
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

#1.1

X, y = datasets.make_classification(
    weights=[0.8,0.2],
    n_samples=400,
    n_features=2,
    n_informative=2,
    n_repeated=0,
    n_redundant=0,
    flip_y=.08,
    random_state=1024,
    n_clusters_per_class=2
    )

plt.scatter(X[:,0], X[:,1],c=y, cmap='bwr')
plt.show()

#Writing to file

with open('figure.csv', 'w', encoding='UTF8')as f:
    writer=csv.writer(f)
    writer.writerow(X)
    writer.writerow(y)

#1.2
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=1024)

clf=GaussianNB()
clf.fit(X_train, y_train)
class_prob = clf.predict_proba(X_test)
#test = np.argmax(class_prob, axis = 1)
predict = np.argmax(class_prob, axis = 1)
score = accuracy_score(y_test, predict)

#Drawing graphics

fig, ax = plt.subplots(1,2,figsize=(15,5))

ax[0].scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='bwr')
ax[0].set_xlabel('feature 0')
ax[0].set_ylabel('feature 1')
ax[0].set_title('Real score')
ax[0].set_xlim(-4,4)
ax[0].set_ylim(-4,4)

ax[1].scatter(X_test[:,0], X_test[:,1], c=predict, cmap='bwr')
ax[1].set_xlabel('feature 0')
ax[1].set_ylabel('feature 1')
ax[1].set_title('Prediction score: %.3f' % score)
ax[1].set_xlim(-4,4)
ax[1].set_ylim(-4,4)

plt.tight_layout()
plt.savefig('plot2.png')
plt.show()

print("True labels: \n   ", y_test)
print("Prediction: \n", predict )
print("Accuracy score:\n %.2f" % score)

#1.3

clf = GaussianNB()

skf = StratifiedKFold (n_splits = 5)
skf.get_n_splits(X,y)
print (skf)

StratifiedKFold(n_splits=5, random_state = None, shuffle = False)

scores = []

for train_index, test_index, in skf.split(X,y):
    clf = GaussianNB()
    print ("TRAIN \n", train_index, "\n", "TEST \n", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y [test_index]
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    scores.append(accuracy_score(y_test, predict))



mean_score = np.mean(scores)
std_score = np.std(scores)

print ("Accuracy score %.3f (%.3f)" % (mean_score, std_score))