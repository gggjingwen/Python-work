import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

weatherall=pd.read_csv("weather.csv")
weatherall['Weather']=weatherall['Weather'].str.replace("转","~")
weatherall
for i in range(len(weatherall['Weather'])):
    if '~' in weatherall['Weather'][i]:
        weatherall['Weather'][i]=weatherall['Weather'][i][:weatherall['Weather'][i].find('~')]
weatherall
weatherall['Date'] = weatherall['Date'].astype('str')
weatherall['Date']=pd.to_datetime(weatherall['Date'],format='%Y-%m-%d')
weatherall['Year']=weatherall['Date'].dt.year
weatherall

weatherall = weatherall.set_index(weatherall['Date'])
weatherall=weatherall.iloc[:,2:8]

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score
X_Temp=np.array(weatherall[['DayTemp']])
y_Temp=np.array(weatherall['NightTemp'])
X_Temp_train,X_Temp_test,y_Temp_train,y_Temp_test = train_test_split(
    X_Temp,y_Temp,test_size=0.3,random_state=1)
regr = linear_model.LinearRegression()
regr.fit(X_Temp_train,y_Temp_train)
y_Temp_pred=regr.predict(X_Temp_test)
print('Coef_:{} Intercept_:{}\n'.format(regr.coef_,regr.intercept_))
print("Mean squared error: %.2f" % mean_squared_error (y_Temp_test,y_Temp_pred))
print('Variance score : %.2f '% r2_score(y_Temp_test,y_Temp_pred))

plt.scatter(X_Temp_train,y_Temp_train,color='black',alpha=0.7)
plt.scatter(X_Temp_test,y_Temp_test,color='red')
plt.plot(X_Temp_test,y_Temp_pred,color='blue',linewidth=3)
plt.show()

ww=pd.read_table('s.txt',sep=' ')
ww=ww.iloc[:,1:9]
ww
ww.columns=['Day','MeanTemp','MeanMaxTemp','MeanMinTemp','MeanPre','Precipi','Precipii',
            'Windvel']
ww=ww.drop(columns=['Precipii','MeanMaxTemp','MeanMinTemp'])
ww.to_csv('meanweather.csv',encoding='utf_8_sig')

meanweather=pd.read_csv('meanweather.csv')
meanweather=meanweather.iloc[:,1:8]

fig,ax=plt.subplots()
fig.set_size_inches([8,6])
ax.plot(meanweather.Day,meanweather.MeanTemp,color='r',label='MeanTemp',alpha=0.8)
ax.plot(meanweather.Day,meanweather.MeanPre,color='y',label='MeanPre',alpha=0.8)
ax.plot(meanweather.Day,meanweather.Precipi,color='m',label='Precipi',alpha=0.8)
ax.plot(meanweather.Day,meanweather.Windvel,color='c',label='Windvel',alpha=0.8)
plt.legend()
plt.show()

import seaborn as sns
www = list(set(ww.columns) - set(['Day']))
corr_matrix = ww[www].corr()

sns.pairplot(meanweather, x_vars=['MeanPre','Precipi','Windvel'], y_vars='MeanTemp', size=5, aspect=0.8, kind='reg')
sns.heatmap(corr_matrix,cmap='rainbow')

X=np.array(meanweather[['MeanPre','Precipi','Windvel']])
y=np.array(meanweather['MeanTemp'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
y_pred=regr.predict(X_test)
print('Coef_:{} Intercept_:{}\n'.format(regr.coef_,regr.intercept_))
print("Mean squared error: %.2f" % mean_squared_error (y_test,y_pred))
print('Variance score : %.2f '% r2_score(y_test,y_pred))
fig,ax=plt.subplots()
fig.set_size_inches([10,6])
ax.plot(range(len(y_pred)),y_pred,'b',label="predict")
ax.plot(range(len(y_pred)),y_test,'r',label="test")
plt.legend()
plt.show()

kobe=pd.read_csv('kobe.csv',encoding="gbk")
kobe
kobe=kobe.iloc[:,0:7]
kobe.Final=kobe.Final.replace("胜","1").replace("负","0")
X=np.array(kobe[['MIN','FG%','FGA','AST','TO','PTS']])
y=kobe['Final']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

neighbors = np.arange(1,10)
test_accuracy = np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test) 

plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.grid()
plt.show()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train,y_train)
y_knn_pred=knn.predict(X_test)
print('knn score:{}'.format(knn.score(X_test,y_test)))

from sklearn import svm
clf = svm.SVC(gamma='auto')
clf.fit(X_train,y_train)
y_clf_pred=clf.predict(X_test)
print('clf score:{}'.format(clf.score(X_test,y_test)))

max_depth = np.arange(1,10)
test_accuracy = np.empty(len(max_depth))
for i,k in enumerate(neighbors):
    dt = DecisionTreeClassifier(max_depth=k)
    dt.fit(X_train, y_train)
    test_accuracy[i] = dt.score(X_test, y_test) 

plt.title('DT Varying number of max_depth')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.grid()
plt.show()
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=4)
dt.fit(X_train, y_train)
y_dt_pred = dt.predict(X_test)
print('dt score:{}'.format(dt.score(X_test,y_test)))

from sklearn import linear_model
LR = linear_model.LogisticRegression(solver='lbfgs',max_iter=1000,multi_class='auto')
LR.fit(X_train,y_train)
y_LR_pred=LR.predict(X_test)
print('LR score:{}'.format(LR.score(X_test,y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_knn_pred)
print('y_test: ALL: {}, Label0: {}, Label1: {}'.format(len(y_test),len(y_test[y_test == '0']), len(y_test[y_test == '1'])))
print('y_knn_pred: ALL: {}, Label0: {}, Label1: {}'.format(len(y_knn_pred),len(y_knn_pred[y_knn_pred == '0']), 
                                                           len(y_knn_pred[y_knn_pred == '1'])))
print(confusion_matrix)
plt.matshow(confusion_matrix)
plt.title('KNN Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.colorbar()
plt.show()

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_clf_pred)
print('y_test: ALL: {}, Label0: {}, Label1: {}'.format(len(y_test),len(y_test[y_test == '0']), len(y_test[y_test == '1'])))
print('y_clf_pred: ALL: {}, Label0: {}, Label1: {}'.format(len(y_clf_pred),len(y_clf_pred[y_clf_pred == '0']), 
                                                           len(y_clf_pred[y_clf_pred == '1'])))
print(confusion_matrix)
plt.matshow(confusion_matrix)
plt.title('SVM Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.colorbar()
plt.show()

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_dt_pred)
print('y_test: ALL: {}, Label0: {}, Label1: {}'.format(len(y_test),len(y_test[y_test == '0']), len(y_test[y_test == '1'])))
print('y_dt_pred: ALL: {}, Label0: {}, Label1: {}'.format(len(y_dt_pred),len(y_dt_pred[y_dt_pred == '0']), 
                                                          len(y_dt_pred[y_dt_pred == '1'])))
print(confusion_matrix)
plt.matshow(confusion_matrix)
plt.title('Dt Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.colorbar()
plt.show()

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_LR_pred)
print('y_test: ALL: {}, Label0: {}, Label1: {}'.format(len(y_test),len(y_test[y_test == '0']), len(y_test[y_test == '1'])))
print('y_LR_pred: ALL: {}, Label0: {}, Label1: {}'.format(len(y_LR_pred),len(y_LR_pred[y_LR_pred == '0']), 
                                                          len(y_LR_pred[y_LR_pred == '1'])))
print(confusion_matrix)
plt.matshow(confusion_matrix)
plt.title('LR Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.colorbar()
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_knn_pred))
print(classification_report(y_test, y_clf_pred))
print(classification_report(y_test, y_dt_pred))
print(classification_report(y_test, y_LR_pred))
