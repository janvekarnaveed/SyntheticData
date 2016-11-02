# -*- coding: utf-8 -*-
"""
Created on Sun May  8 10:04:41 2016

@author: naveedjanvekar
"""
#################Data Exploration and Preprocessing################################
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectFromModel, RFECV
from sklearn import cross_validation, metrics
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVR
#Read files:
train = pd.read_csv('codetest_train.txt', sep='\t', header=0) #Input to train predictive model
test = pd.read_csv('codetest_test.txt', sep='\t', header=0)   #Data to test the predictive model

#Its generally a good idea to combine both train and test data sets into one, perform feature engineering and then divide them later again
#combine test and train into a dataframe ‘data’ with a ‘source’ column specifying where each observation belongs
train['source']='train'
test['source']='test'

data = pd.concat([train, test],ignore_index=True) #Merged train and test dataset

#ID Column to assign a unique identifier to each record
data['index'] = data.index
print (train.shape, test.shape, data.shape) #To check dimensions of the data

#Check for null values
data.apply(lambda x: sum(x.isnull()))

#Describe for some statistics of the dataset data
data.describe()

#Viewing number of unique values in variables
data.apply(lambda x: len(x.unique()))


#Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
#Exclude ID cols and source:

#Print frequency of categories
for col in categorical_columns:
    print ('\nFrequency of Categories for varible %s'%col)
    print (data[col].value_counts())
    
#Impute Missing Values
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

data = DataFrameImputer().fit_transform(data)

#one-hot encoding to convert categorical variables to numeric values
var_mod = ['f_121','f_215','f_237','f_61']
ID_col = ['index']
target_col = ["target"]
cat_cols = ['f_121','f_215','f_237','f_61']
num_cols= list(set(list(data.columns))-set(cat_cols)-set(ID_col)-set(target_col))

  
#create label encoders for categorical features
for var in var_mod:
 number = LabelEncoder()
 data[var] = number.fit_transform(data[var].astype('str'))    
 
#One Hot Coding to creating dummy variables, one for each category of a categorical variable
data = pd.get_dummies(data, columns=['f_121','f_215','f_237','f_61']) 
 
#Checking datatypes of the variables in dataset 
data.dtypes 

#Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

#Drop unnecessary columns:
test.drop(['source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)


##################Feature Elimination using PCA################################
#convert it to numpy arrays
X=train.values

#Scaling the values
X = scale(X)

#Used PCA class with n_components == ‘mle’, Minka’s MLE is used to guess the dimension
pca = PCA(n_components='mle')
pca.fit(X)

#The amount of variance that each PC explains
#var= pca.explained_variance_ratio_
print ("Explained variance by component: %s" % pca.explained_variance_ratio_)

pcaDF = pd.DataFrame(pca.components_,columns=train.columns)
factor = FactorAnalysis(n_components=4,  random_state=101).fit(X)

factoranalysis = pd.DataFrame(factor.components_,columns=train.columns)
#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print (var1)

#Scree plot
plt.plot(var1)

##########################Model Building##########################################################################
#Define target and ID columns:
target = 'target'
IDcol = ['index']

def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    """
    This function trains on the train dataset with the given algorithm and then predicts the target variable
    on the test dataset and exports the result
    """
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print ("\nModel Report")
    print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    print ("Model MSE:", (metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
   
    
    print ("R-squared:", (metrics.r2_score(dtrain[target].values, dtrain_predictions))) 
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    dtest=dtest[predictors].join(dtest[target])
    
    #Export submission file:
    dtest.drop('index', axis=1, inplace=True)
    dtest.to_csv(filename, sep='\t', index=False)
    



#Lasso Regression Model
predictors = [x for x in train.columns if x not in [target]+[IDcol]]
alg1 = Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000, normalize=False, positive=False, precompute=False, random_state=None, selection='cyclic', tol=0.0001, warm_start=False)
modelfit(alg1, train, test, predictors, target,IDcol, 'testLasso.txt')

#Ridge Regression Model
predictors = [x for x in train.columns if x not in [target]+[IDcol]]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'testRidge.txt')

#ElasticNet Regression
predictors = [x for x in train.columns if x not in [target]+[IDcol]]
alg3 = ElasticNet(alpha=1,l1_ratio=0.5)
modelfit(alg3, train, test, predictors, target, IDcol, 'testElasticNet.txt')


#DecisionTreeRegressor
predictors = [x for x in train.columns if x not in [target]+[IDcol]]
alg4 = DecisionTreeRegressor(criterion="mse", splitter='best', max_features="auto", max_depth=15, min_samples_leaf=100)
modelfit(alg4, train, test, predictors, target, IDcol, 'testDecTree.txt')

#GradientBoostingRegressor
predictors = [x for x in train.columns if x not in [target]]
alg5 = ensemble.GradientBoostingRegressor(n_estimators= 10, max_depth= 4, min_samples_split= 1,learning_rate= 0.01, loss= 'ls')
modelfit(alg5, train, test, predictors, target,IDcol, 'testGB.txt')

#RandomForestRegressor
predictors = [x for x in train.columns if x not in [target]]
alg6 = RandomForestRegressor(n_estimators=100, max_features="auto", oob_score = True,n_jobs = 4,random_state =1)
modelfit(alg6, train, test, predictors, target,IDcol, 'testRF.txt')


#Linear Regression Model with Recursive Feature Elimination with Cross-Validation
predictors = [x for x in train.columns if x not in [target]+[IDcol]]
alg7 = LinearRegression()
selector = RFECV(alg7, step=1, cv=5)
modelfit(selector, train, test, predictors, target,IDcol, 'testOutLinReg.txt')

