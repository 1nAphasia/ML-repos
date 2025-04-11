import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from XGVBoost import XGBClassifier
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score


df=pd.read_excel('./data/信用卡交易数据.xlsx')
df.head()

x=df.drop(columns='欺诈标签')
y=df['欺诈标签']

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=100)
clf=XGBClassifier(n_estimators=100,learning_rate=0.05)
clf.fit(X_train,Y_train)

Y_pred=clf.predict(X_test)

# print(Y_pred)


a=pd.DataFrame()
a['预测值']=list(Y_pred)
a['实际值']=list(Y_test)

#score=accuracy_score(Y_pred,Y_test)
#print(score)
#print(clf.score(X_test,Y_test))

Y_pred_prob=clf.predict_proba(X_test)
# print(Y_pred_prob[0:10])

fpr,tpr,thres=roc_curve(Y_test,Y_pred_prob[:,1])
#plt.plot(fpr,tpr)
#plt.show()
score=roc_auc_score(Y_test,Y_pred_prob[:,1])
print(score)

features=x.columns
importances=clf.feature_importances_
importances_df=pd.DataFrame()
importances_df['特征名称']=features
importances_df['特征重要性']=importances
importances_df.sort_values('特征重要性',ascending=False)

parameters={'max_depth':[1,3,5],'n_estimators':[50,100,150],'learning_rate':[0.01,0.05,0.1,0.2]}
clf=XGBClassifier()
grid_search=GridSearchCV(clf,parameters,scoring='roc_auc',cv=5)

grid_search.fit(X_train,Y_train)
print(grid_search.best_params_)

clf=XGBClassifier(max_depth=1,n_estimators=100,learning_rate=0.05)
clf.fit(X_train,Y_train)

Y_pred_prob=clf.predict_proba(X_test)
score=roc_auc_score(Y_test,Y_pred_prob[:,1])
print(score)