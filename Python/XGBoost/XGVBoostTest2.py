import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

df = pd.read_excel('./data/信用评分卡模型.xlsx')
print(df.head())

X = df.drop(columns='信用评分')
Y = df['信用评分']

model = LinearRegression()
model.fit(X,Y)

print('各系数为:' + str(model.coef_))
print('常数项系数k0为:' + str(model.intercept_))

X2 = sm.add_constant(X)
est = sm.OLS(Y, X2).fit()
print(est.summary())