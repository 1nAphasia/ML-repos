from xgboost import XGBClassifier
import numpy as np

X=np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])
Y=[0,0,0,1,1]

model=XGBClassifier()
model.fit(X,Y)

print(model.predict(np.array([[5,5]])))