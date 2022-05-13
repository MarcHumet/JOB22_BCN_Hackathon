#!/usr/bin/env python
# coding: utf-8

# ## Description

# **Enunciado**:
# 
# Los insectos nocturnos representan uno de los grupos más diversos de organismos, por lo que es de suma importancia estudiarlos.
# 
# Es por ello que un grupo de prestigiosos entomólogos han construido un ecosistema aislado con múltiples especies para poder estudiarlos en mayor detalle. Para este estudio están diseñando un sistema de sensores para poder trackear de forma automática las dinámicas y hábitos de estos insectos.
# 
# Este dataset contiene datos de las mediciones de los sensores, el tiempo de las mediciones y el tipo de insecto identificado.
# 
# 
# El dataset **'train.csv'** contiene las siguientes variables:
# 
# Hour: Hora a la que se ha hecho la medición.
# 
# Minutes: Minutos en los que se ha realizado la medición.
# 
# Sensor_alpha: Posición del insecto al sensor alpha.
# 
# Sensor_beta: Posición del insecto al sensor beta.
# 
# Sensor_gamma: Posición del insecto al sensor gamma.
# 
# Sensor_alpha_plus: Posición del insecto al sensor alpha+.
# 
# Sensor_beta_plus: Posición del insecto al sensor beta+.
# 
# Sensor_gamma_plus: Posición del insecto al sensor gamma+.
# 
# Insect: Categoría de insecto.
# 
#     0 -> Lepidoptero 
# 
#     1 -> Himenoptera
# 
#     2 -> Diptera
# 
# 
# 

# 
# 

# ## Task and target of the exercise

# Tienes que entregar el **link de tu repositorio de Github/Gitlab**.
# 
# Este tiene que tener:
# 
# El código con el que has realizado el EDA y el modelo predictivo.
# 
# Un **archivo 'results.csv'** con las predicciones de tu algoritmo al meterle como input el dataset test_x. Este archivo con los resultados ha de tener una columna con el índice y otra con las predicciones del 'Insect'.

# In[53]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from collections import Counter
from sklearn.decomposition import PCA
import lightgbm
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN 
from imblearn.combine import SMOTETomek
from lightgbm import LGBMClassifier
#import joblib

import dill as pickle


import pandas as pd
from pycaret.classification import * 
#from ngboost import NGBClassifier
#from ngboost.distns import Bernoulli
 
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.metrics import f1_score, confusion_matrix
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

get_ipython().run_line_magic('matplotlib', 'inline')


# # Exploratory Data Analysis

# ## Data acquisition & preliminary study 

# In[2]:


insect_df=pd.read_csv ('train.csv',index_col=0)


# In[3]:


insect_df.describe()


# In[4]:


insect_df.isnull().sum()


# There are no null values.

# In[5]:


#Hour and minute columns are unified by converting hour to minutes and adding it to the minute column:
insect_df['Minutes']=(insect_df['Hour']*60)+insect_df['Minutes']
insect_df.drop('Hour',axis=1,inplace=True)
insect_df.describe()


# In[6]:


insect_df.info()


# All columns are numerical values. However, the insect column should be consideres as categorical (as label column).

# ## Data Visualization

# In[7]:


sns.pairplot(insect_df,hue='Insect',corner=True)


# From previous graph, several information can be extracted from the original data set:
# * There is special patern for each type of insect showing differential graphical clusters.
# * Data has mostly a Gaussian distribution.
# * There is no need to transform time values to cyclical because data distribution (not allocated in the extremes of the range). 
#     

# To visualize the diferent insect's types in a graphical mode, the dimension of the parameters is reduced to 2 by Principal Component Analysis. 

# In[8]:


pca = PCA(n_components = 2)
insect_df_no_target= insect_df.copy()
insect_df_no_target.drop('Insect',inplace=True,axis=1)
pca_insects = pca.fit_transform(insect_df_no_target)
pca_insects_df = pd.DataFrame(data=pca_insects, 
                            columns = ['Component 1', 'Component 2']
)


#per donar color
labels=np.array(insect_df.Insect)
LABEL_COLOR_MAP = {0 : 'r', 1 : 'b', 2 : 'y', 3 : '%killbgscripts'}
label_color = [LABEL_COLOR_MAP[l] for l in labels]


# In[9]:


fig=plt.figure(figsize=(6,6))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('Component 1', fontsize=15)
ax.set_ylabel('Component 2', fontsize=15)
ax.set_title('Components Principals', fontsize=20)

ax.scatter(x= pca_insects_df['Component 1'], y=pca_insects_df['Component 2'],
           c=label_color,s=3,alpha=0.9)
plt.show()


# Clearly, there is paterns than define each class of instect, and it can be used to generate a model to predict which class of insect is based on his movement pattern.

# ## Data Engineering

# In[10]:


# Split the dataframe between the label column and  columns which contains the info to label the insects:
X=insect_df.copy()
X.drop('Insect', axis=1, inplace=True)
y=insect_df.Insect #label


# In[13]:


#Class distribution checking
counter = Counter(y)
dist_plot=sns.barplot(x=list(counter.keys()), y=list(counter.values()))
dist_plot.set_title('Distribution of instances per label')
dist_plot.set_xlabel('Label')
dist_plot.set_ylabel('# of instances')


# Number of instances for each class is unbalanced. It should be considered in the evaluation of model's performance. Specifically, it has to be checked the results with the minoritary class. F1 score is a good metric to check recall and accuracy

# To improve the model performance data will be standarized taking special care on avoiding information leakage from the test sample. Working with Pycaret and pipelines ensure a proper prepocessing of data.

# # Model development by Machine Learning

# ## Initial Screening of Classification Models by Pycaret

# In[14]:


def scoring_f1(y_test,X_test,method,model_name): #generation of different f1 metrics
    score_list=[]
    for i in [None, 'micro', 'macro', 'weighted']:
        score_list.append(i)
        
        f1score=f1_score(y_test, 
                         method.predict(X_test),
                         average= i
        )
        if i!=None:
            score_list.append(f1score)
        else:
            score_list.append(list(f1score))
    score_list.pop(0)
    df = f1_df(score_list,model_name)
    return df      


# In[15]:


def f1_df (score_list, model_name): #dataframe generator for f1 metrics
    df = pd.DataFrame({'Model': [model_name],
                    'F1_ind_L0':  [round(score_list[0][0], 3)], 
                    'F1_ind_L1':  [round(score_list[0][1], 3)], 
                    'F1_ind_L2':  [round(score_list[0][2], 3)],
                    'F1_averg_micro' : [round(score_list[2], 3)], 
                    'F1_averg_macro' : [round(score_list[4], 3)], 
                    'F1_averg_weighted' : [round(score_list[6],3)],
                    },
                     index = None)
    return df


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3,  
                                                    stratify=y,
)


# In[17]:


#setting the experiment

experiment = setup(insect_df, 
                   target='Insect'
)  


# In[18]:


#show the best model and their statistics

best_model = compare_models() 


# The model with best results in this first exploratory analysis is Light Gradient Boosting Machine. However, let's check the tunning for the top three models with 2 methodologies: Pycaret and GridSearchCV.

# ## Tuning hyperparameters

# ### Gradient Boosting Classifier

# #### Tuned with Pycaret

# In[22]:


gbc=create_model('gbc')	# Gradient Boosting Classifier


# In[23]:


gbc_tuned=tune_model(gbc, 
                     optimize = 'F1'
)


# In[24]:


gbc_PyC_df= scoring_f1(y_test,
                           X_test,
                           gbc,
                           'gbc_PyC')


# In[25]:


gbc_PyC_tuned_df= scoring_f1(y_test,
                           X_test,
                           gbc_tuned,
                           'gbc_PyC_tuned')


# In[26]:


gbc_PyC_df


# In[27]:


gbc_PyC_tuned_df


# #### Tuned with GridSearchCV

# In[ ]:


grid_gbt1_df


# In[28]:


param_test1 = {'n_estimators':range(20,300,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(),
                        param_grid = param_test1, 
                        n_jobs = 4, 
                        cv = 5,
                        verbose = 4,
)

ss = StandardScaler()
 # fit_transform (), adjust and standarize data 
X_train_t = ss.fit_transform(X_train)
 # transform (), standarize X_test  
X_test_t = ss.transform(X_test)

#cal fer standarització a X_train!!!
gsearch1.fit(X_train_t,y_train)


# In[29]:


gsearch1.best_params_, gsearch1.best_score_


# In[30]:


gbc1_df=scoring_f1(y_test,X_test_t,gsearch1,'grid_gbt1')


# In[31]:


# Using Pipeline and GridSearchCV to obtimize the Gradient Boosting Classifier Model
steps = [('scaler', StandardScaler()), ('gbc', GradientBoostingClassifier())]
pipeline = Pipeline(steps) # define the pipeline object


# In[32]:


parameters = {'gbc__max_depth': [60,80,100],
            'gbc__max_features': [3],
            'gbc__min_samples_leaf': [3],
            'gbc__min_samples_split': [8],
            'gbc__n_estimators': [100],
}

grid1 = GridSearchCV(pipeline, 
                     param_grid=parameters, 
                     cv=3, 
                     verbose=0)
grid1.fit(X_train, y_train)
print ("score = %3.3f" %(grid1.score(X_test,y_test)))
print (grid1.best_params_)


# In[33]:


parameters = {'gbc__max_depth': [80],
            'gbc__max_features': [3],
            'gbc__min_samples_leaf': [3],
            'gbc__min_samples_split': [8],
            'gbc__n_estimators': [60, 80, 100, 120]
}

grid2 = GridSearchCV(pipeline, 
                    param_grid=parameters, 
                    cv=3, 
                    verbose=0,
)
grid2.fit(X_train_t, y_train)
print ("score = %3.3f" %(grid2.score(X_test_t,y_test)))
print (grid2.best_params_)


# In[34]:


parameters = {'gbc__max_depth': [80],
            'gbc__max_features': [3],
            'gbc__min_samples_leaf': [3],
            'gbc__min_samples_split': [6,8,10],
            'gbc__n_estimators': [100],
}

grid3 = GridSearchCV(pipeline, 
                    param_grid=parameters, 
                    cv=3,
                    verbose=0,
)

grid3.fit(X_train_t, y_train)
print ("score = %3.3f" %(grid3.score(X_test_t,y_test)))
print (grid3.best_params_)


# In[35]:


print ("score = %3.3f" %(grid3.score(X_test_t, y_test)))
print (grid3.best_params_)

grid_gbt3_df = scoring_f1(y_test, X_test_t, 
                          grid3,
                          'grid_gbt3'
                         )


# In[36]:


grid_gbt2_df=scoring_f1(y_test,X_test_t,grid2,'grid_gbt2')


# In[37]:


grid_gbt1_df=scoring_f1(y_test,X_test_t,grid1,'grid_gbt1')


# In[38]:


parameters = { 'gbc__criterion':['friedman_mse', 'squared_error', 'mse'],
            'gbc__max_depth': [80],
            'gbc__max_features': [3],
            'gbc__min_samples_leaf': [3],
            'gbc__min_samples_split': [8],
            'gbc__n_estimators': [100]
}

grid_gbc1 = GridSearchCV(pipeline, 
                    param_grid=parameters, 
                    cv=3, 
                    verbose=0
)
grid_gbc1.fit(X_train_t, y_train)
print ("score = %3.3f" %(grid_gbc1.score(X_test_t,y_test)))
print (grid_gbc1.best_params_)


# In[39]:


parameters = {'gbc__loss':['deviance','exponential'],
            'gbc__criterion':['mse'],
            'gbc__max_depth': [80],
            'gbc__max_features': [3],
            'gbc__min_samples_leaf': [3],
            'gbc__min_samples_split': [8],
            'gbc__n_estimators': [100]
}

grid_gbc2 = GridSearchCV(pipeline, 
                    param_grid=parameters, 
                    cv=3, 
                    verbose=0
)
grid_gbc2.fit(X_train_t, y_train)
print ("score = %3.3f" %(grid_gbc2.score(X_test_t,y_test)))
print (grid_gbc2.best_params_)



# In[40]:


grid_gbc1_df=scoring_f1(y_test,X_test_t,grid_gbc1,'grid_gbc1' )


# In[41]:


grid_gbc2_df= scoring_f1(y_test,X_test_t,grid_gbc2,'grid_gbc2')


# ### Light Gradient Boosting Classifier

# #### Tuned with Pycaret

# In[42]:


lightgbm_met=create_model('lightgbm')	#Light Gradient Boosting Machine


# In[43]:


lightgbm_tuned=tune_model(lightgbm_met, 
                          optimize = 'F1'
)


# In[44]:


print(lightgbm_tuned)


# In[46]:


lightgbm_PyC_df=scoring_f1(y_test,
                           X_test,
                           lightgbm_met,
                           'lightgbm_PyC',
                          )


# In[47]:


lightgbm_PyC_tuned_df = scoring_f1(y_test,
                                   X_test,
                                   lightgbm_tuned,
                                   'lightgbm_PyC_tuned'
                                  )


# #### Tuned with GridSearchCV

# In[48]:


# Using Pipeline and GridSearchCV to obtimize the Random Forest Classifier Model
steps = [('scaler', StandardScaler()), 
         ('lgbmc', LGBMClassifier())
        ]
pipeline = Pipeline(steps) # define the pipeline object.


# In[49]:


parameters = {'lgbmc__boosting_type' : ['gbdt'], 
              'lgbmc__num_leaves' : [31,80,120], 
              'lgbmc__max_depth' : [3,5,9],
              'lgbmc__learning_rate' : [0.1], 
              'lgbmc__class_weight' : [None],
              'lgbmc__n_estimators' : [100], 
              'lgbmc__subsample_for_bin' : [200000], 
              'lgbmc__objective' : [None],
              'lgbmc__class_weight' : [None],
              'lgbmc__min_split_gain' : [0.0],
              'lgbmc__min_child_weight' : [0.001], 
              'lgbmc__min_child_samples' : [20], 
              'lgbmc__subsample':[1.0], 
              'lgbmc__subsample_freq' : [0],
              'lgbmc__colsample_bytree' : [1.0],
              'lgbmc__reg_alpha' : [0.0], 
              'lgbmc__reg_lambda' : [0.0], 
              'lgbmc__random_state' : [1526], 
              'lgbmc__n_jobs' : [-1],              
              'lgbmc__importance_type' : ['split'], 
             }
    


# In[55]:


grid_lgbmc1 = GridSearchCV (pipeline, 
                    param_grid = parameters, 
                    cv=3, 
                    verbose=0
                )
grid_lgbmc1.fit(X_train, y_train)
print ("score = %3.3f" %(grid_lgbmc1.score(X_test,y_test)))
print (grid_lgbmc1.best_params_)


# In[56]:


grid_lgbmc1_df=scoring_f1(y_test,X_test,grid_lgbmc1,'grid_lgbmc1' )
grid_lgbmc1_df


# In[57]:


grid_lgbmc1.best_params_


# lgbc
# https://machinelearningmastery.com/light-gradient-boosted-machine-lightgbm-ensemble/

# ### Random Forest

# #### Tuned with Pycaret

# In[58]:


random_forest=create_model('rf')


# In[59]:


random_forest_tuned=tune_model(random_forest, 
                          optimize='F1'
)


# In[60]:


print(random_forest_tuned)


# Tunning with Pycaret does not improve the model.

# In[61]:


rf_tuned_PyC_df=scoring_f1(y_test,X_test,random_forest_tuned, 'rf_tuned_PyC' )
rf_PyC_df=scoring_f1(y_test,X_test,random_forest, 'rf_PyC' )


# #### Tuned with GridSearchCV

# In[62]:


# Using Pipeline and GridSearchCV to obtimize the Random Forest Classifier Model
steps = [('scaler', StandardScaler()), ('clf', RandomForestClassifier())]
pipeline = Pipeline(steps) # define the pipeline object.


# In[63]:


parameters = {'clf__bootstrap': [True],    
              'clf__max_depth': [80, 90, 100, 110],    
              'clf__max_features': [2, 3],
              'clf__min_samples_leaf': [3, 4, 5],
              'clf__min_samples_split': [8, 10, 12],
              'clf__n_estimators': [50, 100, 300],
}


# In[64]:


grid_rf1 = GridSearchCV(pipeline, 
                    param_grid=parameters, 
                    cv=3, 
                    verbose=0
)
grid_rf1.fit(X_train, y_train)
print ("score = %3.3f" %(grid_rf1.score(X_test,y_test)))
print (grid_rf1.best_params_)


# In[65]:


grid_rf1_df=scoring_f1(y_test,X_test,grid_rf1,'grid_rf1' )


# ## Over/under Sampling study

# With the best parameters for lgbm, different over/under sampling technics were tested:

# In[66]:


steps = [('scaler', StandardScaler()), 
         ('lgbmc', lightgbm.LGBMClassifier())]

parameters = {'lgbmc__boosting_type': ['gbdt'],
             'lgbmc__class_weight': [None],
             'lgbmc__colsample_bytree': [1.0],
             'lgbmc__importance_type': ['split'],
             'lgbmc__learning_rate': [0.1],
             'lgbmc__max_depth': [9],
             'lgbmc__min_child_samples': [20],
             'lgbmc__min_child_weight': [0.001],
             'lgbmc__min_split_gain': [0.0],
             'lgbmc__n_estimators': [100],
             'lgbmc__n_jobs': [-1],
             'lgbmc__num_leaves': [31],
             'lgbmc__objective': [None],
             'lgbmc__random_state': [1526],
             'lgbmc__reg_alpha': [0.0],
             'lgbmc__reg_lambda': [0.0],
             'lgbmc__subsample': [1.0],
             'lgbmc__subsample_for_bin': [200000],
             'lgbmc__subsample_freq': [0],
            }

pipeline = Pipeline(steps) # define the pipeline object.

sampl_meth=['None','SMOTE','SMOTEENN','SMOTETomek']

sampling_metrics_df=pd.DataFrame(columns=['Sampling','F1_individual', 'F1_averg_micro', 'F1_averg_macro', 'F1_averg_weighted'])

F1_ind_L0,F1_ind_L1,F1_ind_L2 =[],[],[]
F1_averg_micro, F1_averg_macro ,F1_averg_weighted =[],[],[]

for sampling_method in [0,SMOTE,SMOTEENN,SMOTETomek]:
    if sampling_method==0:
        X_train_modified,y_train_modified = X_train,y_train 
        
    else:
        method = sampling_method (random_state = 42)
        X_train_modified,y_train_modified = method.fit_resample(X_train,y_train) 
    grid = GridSearchCV(pipeline,                     
                        param_grid = parameters, 
                        cv = 3, 
                        verbose = 0
    )
        
    grid.fit(X_train_modified, y_train_modified)
    list_scoring_df= scoring_f1(y_test, X_test,grid, 'grid')
    
    F1_ind_L0.append(round(list_scoring_df['F1_ind_L0'][0],3))
    F1_ind_L1.append(round(list_scoring_df['F1_ind_L1'][0],3))
    F1_ind_L2.append(round(list_scoring_df['F1_ind_L2'][0],3))
    F1_averg_micro.append(round(list_scoring_df['F1_averg_micro'][0],3))
    F1_averg_macro.append(round(list_scoring_df['F1_averg_macro'][0],3))
    F1_averg_weighted.append(round(list_scoring_df['F1_averg_weighted'][0],3))
    


# In[67]:


F1_metric_vs_sampling_df=pd.DataFrame({'Over/under Sampling': sampl_meth,
                                'F1_ind_L0':  F1_ind_L0 , 
                                'F1_ind_L1':  F1_ind_L1, 
                                'F1_ind_L2':  F1_ind_L2 , 
                                'F1_averg_micro' : F1_averg_micro, 
                                'F1_averg_macro' : F1_averg_macro, 
                                'F1_averg_weighted' : F1_averg_weighted
                                },index=None)
    


# In[68]:


F1_metric_vs_sampling_df


# There is no improvement by applying different technics of over/under samplig to treat unbalanced classes. So, no additional changes will be performed on the model.

# # Results comparison and Conclusions

# In[77]:


#concatenate of the dataframe with the metrics of models:
f1_df_metrics_df= pd.concat([gbc1_df,
                           lightgbm_PyC_df,
                           lightgbm_PyC_tuned_df,
                           grid_gbt2_df,
                           grid_gbt3_df,
                           grid_gbc1_df,
                           grid_gbc2_df,
                           rf_PyC_df,
                           rf_tuned_PyC_df,
                           gbc_PyC_df,
                           gbc_PyC_tuned_df,
                           grid_rf1_df,
                           grid_lgbmc1_df
                          ]).reset_index(drop=True)


# In[78]:


#Data Frame sorted by F1 results
sort_by_F1averg_weight = f1_df_metrics_df.sort_values('F1_averg_weighted', ascending=False).reset_index(drop=True)
sort_by_F1averg_weight


#  There is something wrong with Pycaret and creating models, in the web there is some comments of leakage of information in the feature selection along the setup process: https://github.com/pycaret/pycaret/issues/1874. 
#  
#  The predictions models created by Pycaret are discarded and only considered the ones tuned by gridsearch. Among them, the best method is Light Gradient Boosting Classifier (grid_lgbmc1). The best model is saved with dill extension of the pickle module:

# In[71]:


pkl_filename = "lightgbm.pkl"


with open(pkl_filename, 'wb') as file:
    pickle.dump(grid_lgbmc1, file)  


# It should be highlighted that the initial analysis with  Pycaret gave the same order of performance for the top three models as tuned by gridsearchCV with few lines of code. Although it has some drawbacks as working as a blackbox, it is very useful to adress efforts on the most promising methodoloies to be tuned for the proposed data set. Furthermore, next weeks the next update of Pycaret (3.0) will be released ant it looks like this and other issues are going be fixed : https://towardsdatascience.com/pycaret-3-0-is-coming-soon-whats-new-e890e6a69ff5.

# # Generating predictions from 'test_x.csv'

# In[72]:


test_x_df=pd.read_csv('test_x.csv',index_col='Unnamed: 0')


# In[73]:


#Hour and minute columns are unified by converting hour to minutes and adding it to the minute column:
test_x_df['Minutes']=(test_x_df['Hour']*60)+test_x_df['Minutes']
test_x_df.drop('Hour',axis=1,inplace=True)
test_x_df.describe()


# In[74]:


insect_df_no_target.describe()


# Comparing statistical data from training initial set and the testing set, shows that both are very similar. Prediction model should work properly with test_x dataframe. Next, 'results.csv' is generated:

# In[76]:


results_df=pd.DataFrame(grid_lgbmc1.predict(test_x_df),
                        columns=['Insect'],
                        index=test_x_df.index
                       )
results_df.to_csv('results.csv')
results_df 


# In[ ]:




