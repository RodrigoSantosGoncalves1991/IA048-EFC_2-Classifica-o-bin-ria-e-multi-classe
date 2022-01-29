import pandas as pd

dataset_voz = pd.read_csv('/content/dados_voz_genero.csv', sep=';')
voz_masculina = dataset_voz.loc[dataset_voz['label']==1.0,:]
voz_feminina = dataset_voz.loc[dataset_voz['label']==0.0,:]

import numpy as np
import matplotlib.pyplot as plt

colunas_fonte = np.asarray(dataset_voz.columns)
figure_size=(10,6)
fig, axes = plt.subplots(5, 4, figsize=figure_size)
fig.canvas.set_window_title("Histograma dos atributos de entrada")
ax = axes.ravel()
plt.rcParams['legend.fontsize'] = 5
for i in range(19):
    _, bins = np.histogram(dataset_voz.loc[:,colunas_fonte[i]], bins=100)
    ax[i].hist(voz_masculina.loc[:,colunas_fonte[i]], bins=bins, color='navy', alpha=.7)
    ax[i].hist(voz_feminina.loc[:,colunas_fonte[i]], bins=bins, color='magenta', alpha=.7)
    ax[i].set_title(colunas_fonte[i],size = 16)
    ax[i].set_yticks(())
    ax[i].legend(["voz masculina", "voz feminina"], loc="best")
fig.tight_layout()
plt.show()

import seaborn as sns

#correlacao_entradas = dataset_voz.drop(columns=['label'])
correlacao_entradas = dataset_voz
f, ax = plt.subplots(figsize=(15, 10))
f.canvas.set_window_title("Matriz de correlação entre os atributos de entrada")
correlacao = correlacao_entradas.corr()
matriz_de_correlacao = sns.heatmap(round(correlacao, 3), annot=True, ax=ax, cmap="coolwarm", fmt='.3f', linewidths=.05)
f.subplots_adjust(top=0.95)
t= f.suptitle('Matriz de correlação entre os atributos de entrada', fontsize=14)
plt.show()

from sklearn.model_selection import train_test_split

x_train, x_validation, y_train, y_validation = train_test_split(dataset_voz.drop('label', axis=1), dataset_voz['label'], test_size=0.2, random_state=813)

from sklearn.linear_model import LogisticRegressionCV

logisticModel = LogisticRegressionCV(cv = 10, max_iter=10000, solver='newton-cg' ,n_jobs=-1).fit(x_train, y_train)
predictions = logisticModel.predict(x_validation)
probabilities_predictions = logisticModel.predict_proba(x_validation)

from sklearn.metrics import  classification_report, confusion_matrix

print(classification_report(y_validation,predictions))
print(confusion_matrix(y_validation, predictions))

yprob = probabilities_predictions[:, 1]

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_validation, yprob)

from sklearn.metrics import roc_curve

false_positives, true_positives, _ = roc_curve(y_validation, yprob)
f.canvas.set_window_title("Receiver operating characteristic")
plt.title("Receiver operating characteristic")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.rcParams['legend.fontsize'] = 8
plt.plot(false_positives, true_positives, color='blue', label='Curva ROC do modelo estimado (AUC = %0.3f)' % auc)
plt.plot([0, 1], [0, 1], color='y', linestyle='dashed', label='Classificador Aleatório')
#plt.plot([0-0.006, 1], [1+0.006, 1+0.006], color='g', linestyle='dotted', label='Classificador Perfeito')
#plt.plot([0-0.006, 0-0.006], [0-0.006, 1+0.006], linestyle='dotted', color='g')
plt.plot([0, 1], [1, 1], color='g', linestyle='dashed', label='Classificador Perfeito')
plt.plot([0, 0], [0, 1], linestyle='dashed', color='g')
plt.legend(loc='best')
plt.savefig('Receiver operating characteristic.png', dpi=1080, facecolor='w', edgecolor='w', orientation='landscape', papertype=None, format=None,transparent=False, bboxinches=None, padinches=0.1,frameon=None, bbox_inches='tight')
plt.show()


def funcao_degrau(v, threshold):
    if v >= threshold:
        return 1
    else:
        return 0

from sklearn.metrics import f1_score

medida_f1 = []

thresholds = np.arange(0.0, 1.001, 0.001)
for threshold in thresholds:
    ythreshold = np.array([funcao_degrau(x, threshold) for x in yprob])
    medida_f1.append(f1_score(y_validation, ythreshold))


f.canvas.set_window_title("Curva de evolução da medida-F1")
plt.title(r"$Curva \ de \ evolução \ da \ medida-F_{1}$")
plt.xlabel("threshold de decisão")
plt.ylabel(r"$Medida \ -\ F_{1}$")
plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)
plt.grid(True)
plt.plot(thresholds, medida_f1, color='b', label=r'$Medida-F_{1}$')
plt.plot([-0.01, 1.01], [max(medida_f1), max(medida_f1)], color='y', linestyle='dashed', label='Valor Máximo medida-F1: %0.3f' % max(medida_f1))
plt.plot([thresholds[medida_f1.index(max(medida_f1))], thresholds[medida_f1.index(max(medida_f1))]], [-0.01, 1.01], color='g', linestyle='dashed', label='Melhor threshold : %0.3f' % thresholds[medida_f1.index(max(medida_f1))])
plt.legend(loc='best')
plt.savefig('Curva de evolução da medida-F1.png', dpi=1080, facecolor='w', edgecolor='w', orientation='landscape', papertype=None, format=None,transparent=False, bboxinches=None, padinches=0.1,frameon=None, bbox_inches='tight')
plt.show()

best_threshold = thresholds[medida_f1.index(max(medida_f1))]
ybest  = np.array([funcao_degrau(x, best_threshold) for x in yprob])

f.canvas.set_window_title("Matriz de confusão do melhor threshold de decisão")
sns.heatmap(confusion_matrix(y_validation, ybest), annot=True, fmt="d", cmap="coolwarm")
plt.title("Matriz de confusão do melhor threshold de decisão")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.savefig('Matriz de confusão do melhor threshold de decisão.png', dpi=1080, facecolor='w', edgecolor='w', orientation='landscape', papertype=None, format=None,transparent=False, bboxinches=None, padinches=0.1,frameon=None, bbox_inches='tight')
plt.show()

print(classification_report(y_validation,ybest, digits=3))
print(confusion_matrix(y_validation, ybest))

print("Instâncias com rótulos de voz masculina: %d" % len(list(filter(lambda i: i == 1.0, y_validation))))
print("Instâncias com rótulos de voz feminina: %d" % len(list(filter(lambda i: i == 0.0, y_validation))))