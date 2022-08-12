# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# %%
train = pd.read_csv(r'F:/ai/train.csv')
test = pd.read_csv(r'F:/ai/test.csv')

# %%
print("Shape of train :", train.shape)
print("Shape of test :", test.shape)

# %%
train.head(2)


# %%
train.tail(2)

# %%
test.head(2)



# %%
test.tail(2)

# %%
train.shape

# %%
test.shape

# %%
train.describe()

# %%
train.nunique()

# %%
train['recruitment_channel'].unique()

# %%
train.isnull().sum()

# %%
test.isnull().sum()

# %%
plt.figure(figsize=(12,8))
sns.distplot(train['no_of_trainings'])
plt.show()

# %%
plt.figure(figsize=(12,8))
sns.distplot(train['age'],color='r')
plt.show()

# %%
plt.figure(figsize=(12,8))
sns.distplot(train['KPIs_met >80%'],color='g')
plt.show()

# %%
plt.figure(figsize=(12,8))
sns.distplot(train['avg_training_score'],color='y')
plt.show()

# %%
fig, axarr = plt.subplots(8, 1, figsize=(15, 15))

sns.boxplot(train['employee_id'],palette='Set1',ax=axarr[0])
sns.boxplot(train['no_of_trainings'],palette='Set2',ax=axarr[1])
sns.boxplot(train['age'],palette='Set3',ax=axarr[2])
sns.boxplot(train['previous_year_rating'],palette='icefire',ax=axarr[3])
sns.boxplot(train['length_of_service'],palette='Dark2',ax=axarr[4])
sns.boxplot(train['KPIs_met >80%'],palette='Set1',ax=axarr[5])
sns.boxplot(train['awards_won?'],palette='plasma',ax=axarr[6])
sns.boxplot(train['avg_training_score'],palette='Set1',ax=axarr[7])
plt.show()

# %%
sns.boxplot(train['age'])

# %%
train1=train.copy()
test1=test.copy()

# %%
from sklearn.impute import SimpleImputer

# %%
imp=SimpleImputer(strategy='median')

# %%
train1['previous_year_rating']=imp.fit_transform(train1[['previous_year_rating']])
imp1=SimpleImputer(strategy='most_frequent')

# %%
train1['education']=imp1.fit_transform(train1[['education']])

# %%
test1['previous_year_rating']=imp.fit_transform(test1[['previous_year_rating']])
test1['education']=imp1.fit_transform(test1[['education']])


# %%
sns.jointplot(y='avg_training_score',x='age',data=train1,color='y')
plt.show()

# %%
plt.figure(figsize=(12,8))
sns.countplot(train1['region'],hue=train['is_promoted'],palette='Set1')
plt.xticks(rotation=90)
plt.show()

# %%
plt.figure(figsize=(12,8))
sns.countplot(train1['department'],hue=train['is_promoted'],palette='plasma')
plt.show()

# %%
plt.figure(figsize=(12,8))
sns.countplot(train1['education'],hue=train['is_promoted'],palette='Set3')
plt.show()

# %%
plt.figure(figsize=(12,8))
sns.countplot(train1['recruitment_channel'],hue=train1['is_promoted'],palette='YlOrRd_r')
plt.xticks(rotation=90)
plt.show()

# %%
plt.figure(figsize=(12,8))
sns.countplot(train1['age'],hue=train1['is_promoted'],palette='husl')
plt.show()

# %%
plt.figure(figsize=(12,8))
sns.countplot(train1['KPIs_met >80%'],hue=train1['is_promoted'],palette='Paired')
plt.show()

# %%
plt.figure(figsize=(12,8))
sns.countplot(train1['previous_year_rating'],hue=train1['is_promoted'],palette='Set2')
plt.show()

# %%
plt.figure(figsize=(12,8))
sns.countplot(train1['awards_won?'],hue=train1['is_promoted'],palette='Set1')
plt.show()

# %%
plt.figure(figsize=(12,8))
sns.countplot(train1['avg_training_score'],hue=train1['is_promoted'],palette='RdGy')
plt.xticks(rotation=90)
plt.show()

# %%
sns.pairplot(train1)
plt.show()

# %%
plt.figure(figsize=(12,8))
sns.countplot(train1['age'],hue=train1['department'])
plt.show()

# %%
plt.figure(figsize=(12,8))
sns.countplot(train1['education'],hue=train1['department'])
plt.show()

# %%
plt.figure(figsize=(12,8))
sns.countplot(train1['recruitment_channel'],hue=train1['department'])
plt.show()

# %%
sns.countplot(train1['is_promoted'])
plt.show()

# %%
sns.heatmap(train1.corr())
plt.show()

# %%
train1['work_fraction']=train1['length_of_service']/train1['age']
test1['work_fraction']=test1['length_of_service']/test1['age']

# %%
train1['work_start_year']=train1['age']-train1['length_of_service']
test1['work_start_year']=test1['age']-test1['length_of_service']

# %%
train1['years_remaining_to_retire']=np.abs(train1['age']-60)
test1['years_remaining_to_retire']=np.abs(test1['age']-60)

# %%
ntrain=train1.drop('employee_id',1)
ntest=test1.drop('employee_id',1)

# %%
ntrain.head()

# %%
# importing Logistic Regression from sklean package
from sklearn.linear_model import LogisticRegression

# %%
x=ntrain.drop('is_promoted',1)
y=ntrain.is_promoted

# %%
x=pd.get_dummies(x)


# %%
test_var=pd.get_dummies(ntest)
lg=LogisticRegression()
lg.fit(x,y)
pred=lg.predict(test_var)

# %%
from sklearn.metrics import f1_score,recall_score,accuracy_score,confusion_matrix

# %%
f1_score(y,lg.predict(x))

# %%
recall_score(y,lg.predict(x))

# %%
accuracy_score(y,lg.predict(x))

# %%
confusion_matrix(y,lg.predict(x))

# %%
test_var['is_promoted']=pred
##employee_id of test data and the predicted data

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
import xgboost as xgb

# %%
lr=LogisticRegression()

# %%
dt=DecisionTreeClassifier()

# %%
from sklearn.model_selection import GridSearchCV
param={'max_depth':np.arange(1,40),'criterion':['entropy','gini']}
Gs=GridSearchCV(dt,param,cv=3,scoring='f1_weighted')
Gs.fit(x,y)

# %%
Gs.best_params_


# %%
rf=RandomForestClassifier(n_estimators=31,criterion='entropy')


# %%
nb=GaussianNB()


# %%
#Takes 200+ minutes to compile this grid search
param={'max_depth':[7,8,9],'n_estimators':[900,1000,1100],'objective':['binary:logistic'],'reg_alpha':[0.3,0.4,0.5],'learning_rate':[0.01]}
Gs=GridSearchCV(xgb_model,param,cv=3,scoring='f1_weighted',n_jobs=2,verbose=True)
Gs.fit(x,y)

# %%
Gs.best_params_

# %%
bagdt=BaggingClassifier(base_estimator=dt,n_estimators=150,random_state=0)
baglr=BaggingClassifier(base_estimator=lr,n_estimators=10,random_state=0)
bagnb=BaggingClassifier(base_estimator=nb,n_estimators=10,random_state=0)
boodt=AdaBoostClassifier(base_estimator=dt,n_estimators=80,random_state=0)
boorf=AdaBoostClassifier(base_estimator=rf,n_estimators=50,random_state=0)
boolr=AdaBoostClassifier(base_estimator=lr,n_estimators=50,random_state=0)
boonb=AdaBoostClassifier(base_estimator=nb,n_estimators=50,random_state=0)
gboost=GradientBoostingClassifier(n_estimators=700,random_state=0)
xgb_model=xgb.XGBClassifier(learning_rate=0.01,max_depth=9,n_estimators=1000,objective='binary:logistic',reg_alpha=0.3,num_round=20)

# %%
from sklearn.model_selection import KFold

kf = KFold(n_splits = 2, shuffle=True, random_state=2)
for model, name in zip([lr,dt,rf,nb,bagdt,baglr,bagnb,boodt,boolr,boonb,boorf,gboost,xgb_model],
                       ['Logistic regression','decision tree','random forest','naive bayes',
                        'bagged_decision tree','bagged_logistic regression','bagged_naive bayes',
                        'boosted_decision tree','boosted_logistic regression','boosted_naive bayes','boosted_random forest',
                       'gradient boost','xgboost']):
    k=0
    recall = np.zeros((2,2))
    prec = np.zeros((2,2))
    fscore = np.zeros((2,2))
    for train,test in kf.split(x,y):
        xtrain,xtest = x.iloc[train,:],x.iloc[test,:]
        ytrain,ytest = y[train],y[test]
        model.fit(xtrain,ytrain)
        y_predict= model.predict(xtest)
        cm = confusion_matrix(ytest,y_predict)
        for i in np.arange(0,2):
            recall[i,k] = cm[i,i]/cm[i,:].sum()
        for i in np.arange(0,2):
            prec[i,k] = cm[i,i]/cm[:,i].sum()
        k = k+1
    for row in np.arange(0,2):
        for col in np.arange(0,2):
            fscore[row,col]=2*(recall[row,col]*prec[row,col])/(recall[row,col]+prec[row,col])
    print('f1_score for class1: %0.02f (+/- %0.5f) [%s]' % (np.mean(fscore[0,:]), np.var(fscore[0,:], ddof = 1),name))
    print('f1_score for class2: %0.02f (+/- %0.5f) [%s]' % (np.mean(fscore[1,:]), np.var(fscore[1,:], ddof = 1),name))

# %%
trainingx=ntrain.drop('is_promoted',1)
labels=ntrain.is_promoted
trainingx=pd.get_dummies(trainingx)

# %%
testingx=pd.get_dummies(ntest)
model=gboost.fit(trainingx,labels)
pred=model.predict(testingx)

# %%
#training accuracy
accuracy_score(labels,model.predict(trainingx))


# %%
f1_score(labels,model.predict(trainingx))

