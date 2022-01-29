import pandas as pd
import numpy as np
import time

ini = time.time()

har_smartphone_Xtrain = pd.read_csv('/content/X_train.txt', header=None)
X_train = np.float64(np.asarray([x[0].split() for x in har_smartphone_Xtrain.values]))
har_smartphone_ytrain = pd.read_csv('/content/y_train.txt', header=None)
y_train = np.float64(np.asarray(har_smartphone_ytrain))-1

har_smartphone_Xtest = pd.read_csv('/content/X_test.txt', header=None)
X_test = np.float64(np.asarray([x[0].split() for x in har_smartphone_Xtest.values]))
har_smartphone_ytest = pd.read_csv('/content/y_test.txt', header=None)
y_test = np.float64(np.asarray(har_smartphone_ytest))-1

from sklearn.model_selection import train_test_split

xTrain, x_validation, yTrain, y_validation = train_test_split(X_train, y_train, train_size=0.99, random_state=0)
X_train = np.append(xTrain, x_validation, axis=0)
y_train = np.append(yTrain, y_validation, axis=0)

y_train = y_train.ravel()
y_test = y_test.ravel()

from sklearn.linear_model import LogisticRegressionCV
from sklearn.multiclass import OneVsOneClassifier

logisticModel_OneVsOne = OneVsOneClassifier(LogisticRegressionCV(cv = 10, max_iter=10000, solver='newton-cg' ,n_jobs=-1)).fit(X_train, y_train)

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

fig, ax = plt.subplots(figsize=(10, 10))
disp = plot_confusion_matrix(logisticModel_OneVsOne, X_test, y_test.ravel(), cmap=plt.cm.Purples,values_format='d', ax=ax)
plt.title("Matriz de confusão: regressão logística multiclasse com OneVsOne")
plt.savefig('Matriz de confusão: regressão logística multiclasse com OneVsOne.png', dpi=1080, facecolor='w', edgecolor='w', orientation='landscape', papertype=None, format=None,transparent=False, bboxinches=None, padinches=0.1,frameon=None, bbox_inches='tight')
plt.show()

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

predictions_OneVsOne = logisticModel_OneVsOne.predict(X_test)

print(classification_report(y_test,predictions_OneVsOne, digits=6))

print("precisão micro: %0.16f" % precision_score(y_test, predictions_OneVsOne, average='micro'))
print("precisão macro: %0.16f" % precision_score(y_test, predictions_OneVsOne, average='macro'))
print("precisão weighted: %0.16f" % precision_score(y_test, predictions_OneVsOne, average='weighted'))
print("recall micro: %0.16f" % recall_score(y_test, predictions_OneVsOne, average='micro'))
print("recall macro: %0.16f" % recall_score(y_test, predictions_OneVsOne, average='macro'))
print("recall weighted: %0.16f" % recall_score(y_test, predictions_OneVsOne, average='weighted'))
print("f1-score micro: %0.16f" % f1_score(y_test, predictions_OneVsOne, average='micro'))
print("f1-score macro: %0.16f" % f1_score(y_test, predictions_OneVsOne, average='macro'))
print("f1-score weighted: %0.16f" % f1_score(y_test, predictions_OneVsOne, average='weighted'))


from sklearn.multiclass import OneVsRestClassifier

logisticModel_OneVsRest = OneVsRestClassifier(LogisticRegressionCV(cv = 10, max_iter=10000, solver='newton-cg' ,n_jobs=-1)).fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(10, 10))
disp = plot_confusion_matrix(logisticModel_OneVsRest, X_test, y_test, cmap=plt.cm.Purples,values_format='d', ax=ax)
plt.title("Matriz de confusão: regressão logística multiclasse com OneVsRest")
plt.savefig('Matriz de confusão: regressão logística multiclasse com OneVsRest.png', dpi=1080, facecolor='w', edgecolor='w', orientation='landscape', papertype=None, format=None,transparent=False, bboxinches=None, padinches=0.1,frameon=None, bbox_inches='tight')
plt.show()

predictions_OneVsRest = logisticModel_OneVsRest.predict(X_test)

print(classification_report(y_test, predictions_OneVsRest, digits=6))

print("precisão micro: %0.16f" % precision_score(y_test, predictions_OneVsRest, average='micro'))
print("precisão macro: %0.16f" % precision_score(y_test, predictions_OneVsRest, average='macro'))
print("precisão weighted: %0.16f" % precision_score(y_test, predictions_OneVsRest, average='weighted'))
print("recall micro: %0.16f" % recall_score(y_test, predictions_OneVsRest, average='micro'))
print("recall macro: %0.16f" % recall_score(y_test, predictions_OneVsRest, average='macro'))
print("recall weighted: %0.16f" % recall_score(y_test, predictions_OneVsRest, average='weighted'))
print("f1-score micro: %0.16f" % f1_score(y_test, predictions_OneVsRest, average='micro'))
print("f1-score macro: %0.16f" % f1_score(y_test, predictions_OneVsRest, average='macro'))
print("f1-score weighted: %0.16f" % f1_score(y_test, predictions_OneVsRest, average='weighted'))

from sklearn.neighbors import KNeighborsClassifier

array_f1_weighted = []
impares = np.arange(1, 200, 2)
for k in impares:
  KNN = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
  predictions_KNN = KNN.predict(X_test)
  array_f1_weighted.append(f1_score(y_test, predictions_KNN, average='weighted'))

plt.title(r"$Curva \ da \ medida-F_{1} \ em \ função \ de \ K$")
plt.xlabel(r'$ \ K \in ímpares  \ (1\  \leq K \leq 199 \ ) $')
plt.ylabel(r"$Medida \ -\ F_{1}\ ponderada$")
plt.xlim(min(impares), max(impares))
plt.ylim(min(array_f1_weighted)-0.01, max(array_f1_weighted)+0.01)
plt.grid()
plt.plot(impares, array_f1_weighted, color='b', label=r'$Medida-F_{1}$')
plt.plot([min(impares)-0.01, max(impares)+0.01], [max(array_f1_weighted), max(array_f1_weighted)], color='y', linestyle='dashed', label='Valor Máximo medida-F1 ponderada: %0.3f' % max(array_f1_weighted))
plt.plot([impares[array_f1_weighted.index(max(array_f1_weighted))], impares[array_f1_weighted.index(max(array_f1_weighted))]], [min(array_f1_weighted)-0.01, max(array_f1_weighted)+0.01], color='g', linestyle='dashed', label='Melhor K : %d' % impares[array_f1_weighted.index(max(array_f1_weighted))])
plt.legend(loc='best')
plt.savefig('Curva da medida-F1 em função de K.png', dpi=1080, facecolor='w', edgecolor='w', orientation='landscape', papertype=None, format=None,transparent=False, bboxinches=None, padinches=0.1,frameon=None, bbox_inches='tight')
plt.show()

Kbest = impares[array_f1_weighted.index(max(array_f1_weighted))]
KNN = KNeighborsClassifier(n_neighbors=Kbest).fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(10, 10))
disp = plot_confusion_matrix(KNN, X_test, y_test, cmap=plt.cm.Purples,values_format='d', ax=ax)
plt.title("Matriz de confusão: classificação multiclasse com KNN")
plt.savefig('Matriz de confusão: classificação multiclasse com KNN.png', dpi=1080, facecolor='w', edgecolor='w', orientation='landscape', papertype=None, format=None,transparent=False, bboxinches=None, padinches=0.1,frameon=None, bbox_inches='tight')
plt.show()

predictions_KNN = KNN.predict(X_test)

print(classification_report(y_test, predictions_KNN, digits=6))

print("precisão micro: %0.16f" % precision_score(y_test, predictions_KNN, average='micro'))
print("precisão macro: %0.16f" % precision_score(y_test, predictions_KNN, average='macro'))
print("precisão weighted: %0.16f" % precision_score(y_test, predictions_KNN, average='weighted'))
print("recall micro: %0.16f" % recall_score(y_test, predictions_KNN, average='micro'))
print("recall macro: %0.16f" % recall_score(y_test, predictions_KNN, average='macro'))
print("recall weighted: %0.16f" % recall_score(y_test, predictions_KNN, average='weighted'))
print("f1-score micro: %0.16f" % f1_score(y_test, predictions_KNN, average='micro'))
print("f1-score macro: %0.16f" % f1_score(y_test, predictions_KNN, average='macro'))
print("f1-score weighted: %0.16f" % f1_score(y_test, predictions_KNN, average='weighted'))

fim = time.time()

print("Tempo de execução de código: %0.18f segundos" % (fim-ini))