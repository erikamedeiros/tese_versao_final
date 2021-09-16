from pandas import DataFrame                       # For dataframes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt                        # For plotting data
import seaborn as sns                                     # For plotting data
from sklearn.model_selection import train_test_split    # For train/test splits
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV      # For optimization
from matplotlib import pyplot
from numpy import mean
import time

#Classifeier
from sklearn.neighbors import KNeighborsClassifier    

#Feature selection
from sklearn.feature_selection import VarianceThreshold # Feature selector

#Feature extraction
from sklearn.decomposition import PCA  # Feature extraction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # Feature extraction
from sklearn.manifold import LocallyLinearEmbedding

# class balancing
from imblearn.over_sampling import SMOTE

# For setting up pipeline
from imblearn.pipeline import Pipeline                               

# Various pre-processing steps
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, MaxAbsScaler

#define dataset
dataset = pd.read_csv('dataset.csv', sep=';')  
del dataset["imgName"]

# The data matrix X
X = dataset
# The labels
y = pd.Series([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 2, 2, 1, 3, 0, 2, 1, 3, 2, 1, 3, 0, 0, 2, 2, 1, 3, 3, 1, 3, 0, 2, 3, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 2, 0, 2, 3, 2, 0, 0, 0, 0, 0, 2, 2, 0, 3, 0, 0, 0, 3, 3, 0, 3, 2, 3, 2, 3, 3, 0, 3, 3, 3, 3, 2, 3, 3, 0, 3, 3, 2])

def tuning(pipe, X_train, y_train):
        
    parameters = {
        'scaler': [StandardScaler(), MinMaxScaler(feature_range = (0, 1)), Normalizer(), MaxAbsScaler(), 'passthrough'],
        #'balancing': [SMOTE(random_state=0), BorderlineSMOTE(random_state=0), SVMSMOTE(random_state=0), RandomOverSampler(random_state=0)],
        #'balancing': [SMOTE(random_state=0)],
        'selector': [VarianceThreshold(), PCA(random_state=0), LinearDiscriminantAnalysis(), LocallyLinearEmbedding(random_state=0)],
   	    'classifier': [KNeighborsClassifier()],	
        'classifier__n_neighbors': [1, 3, 5, 7, 10],
	    'classifier__p': [1, 2],
	    'classifier__leaf_size': [1, 5, 10, 15]
    }
    
    cv = StratifiedKFold(2, shuffle=True, random_state=0)
    grid = GridSearchCV(pipe, parameters, cv=cv, scoring = 'accuracy', return_train_score=False)
    grid.fit(X_train, y_train)

    best_params_iteration.append(grid.best_params_)
     
    return grid

#Avaliacao do modelo em kfold de 10
def evaluate_model(model, X_train, y_train, n_folds):
	# prepare the cross-validation procedure
	cv = StratifiedKFold(n_folds, shuffle=True, random_state=0)
	# evaluate model
	scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores


best_params_iteration = []
results_train = []
accuracy_predict = []
maior_acuracia_treino = 0
oversample = SMOTE(random_state=0)
mean_fit_time = []

for seed in range(1, 31):
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print('Iteração: ', seed)
    #Divisão da base em treino (80%) e teste (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify = y, random_state=seed)
    

    X_train, y_train = oversample.fit_resample(X_train, y_train)
          
    pipe = Pipeline([('scaler', StandardScaler()),
                     #('balancing', SMOTE(random_state=0)),
                     ('selector', VarianceThreshold()),
                     ('classifier', KNeighborsClassifier())
                     ])               
    
    model_tuning = tuning(pipe, X_train, y_train)
    
    mean_fit_time.append((model_tuning.cv_results_['mean_fit_time']))

    #Acurácia do treino
    scores_train = evaluate_model(model_tuning, X_train, y_train, 10)
    results_train.append(mean(scores_train))
    
    #Acurácia de teste
    #Make predictions
    predict = model_tuning.predict(X_test)
    accuracy_predict.append(accuracy_score(y_test, predict))

    #Gravando a base de dados de treino da melhor predicao para a MC e plots
    aux = max(float(results_train) for results_train in results_train)
    if maior_acuracia_treino < aux:
        maior_acuracia_treino = aux
        best_X_train = X_train
        best_y_train = y_train
        best_X_test = X_test
        best_y_test = y_test
        best_model = model_tuning
    

#Dados de treino
print ("Média da acurácia do KNN nos dados de treino {:.4f}%.".format(mean(results_train)*100))
print("Desvio padrao do KNN no dados de treino: ", np.std(results_train))


#Dados de teste
print("--------------------------------------------------------------")
print ("A acurácia da predição do KNN foi de {:.4f}%.".format(mean(accuracy_predict)*100))
print("Desvio padrao na predição do KNN: ", np.std(accuracy_predict))

#Dados de teste
print("--------------------------------------------------------------")
print ("O mean_fit_time do KNN foi de", (mean(mean_fit_time)))
print("Desvio padrao do mean fit time do KNN: ", np.std(mean_fit_time))

#Criando a matriz de confusão de cada modelo
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(best_model, best_X_test, best_y_test, display_labels=["2-", "3", "2", "2+"])


# Access the best set of parameters of best model tunning
best_params = best_model.best_params_
print('Melhores parâmetros', best_params)
# Stores the optimum model in best_pipe
best_pipe = best_model.best_estimator_
print('Melhor pipeline', best_pipe)
best_features = best_model.best_estimator_.feature_importances_


result_df = dataset.from_dict(best_model.cv_results_, orient='columns')
#print('Resultados das colunas', result_df.columns)

sns.relplot(data=result_df,
	kind='line',
	x='param_classifier__n_neighbors',
	y='mean_test_score',
	hue='param_scaler',
	col='param_classifier__p')
plt.show()

sns.relplot(data=result_df,
            kind='line',
            x='param_classifier__n_neighbors',
            y='mean_test_score',
            hue='param_scaler',
            col='param_classifier__leaf_size')
plt.show()


#FEATURE IMPORTANCE
# perform permutation importance
from sklearn.inspection import permutation_importance

results = permutation_importance(best_model, best_X_train, best_y_train, scoring='accuracy')
# get importance
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
    

# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

##########
import eli5
numeric_features_list = ["medianR", "medianG", "medianB", "histR", "histG", "histB", "R", "G", "B", "saturationHSV", "hueHSV", "valueHSV", "saturationHSI", "hueHSI", "intensityHSI", "lLab", "aLab", "bLab"]
eli5.show_weights(best_model)

