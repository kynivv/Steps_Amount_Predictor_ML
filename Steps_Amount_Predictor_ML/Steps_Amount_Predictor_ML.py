import pandas as pd
import warnings 
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

# Data Import
df = pd.read_csv('Walk.csv')
df = df.drop('Unnamed: 0', axis=1)

# EDA
#print(df)
#print(df.isnull().sum())
#print(df.dtypes)
#print(df['Day'])


# Data Transformation
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    df[col] = df[col].astype('float')

#print(df.dtypes)
#print(df)


# Train Test Split
features = df.drop('StepCount', axis=1)
target = df['StepCount']

#print(features, target)

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.22, random_state=25)

#print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# Model Training

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import explained_variance_score as evs

models = [SVR(), RandomForestRegressor(), GradientBoostingRegressor(), DecisionTreeRegressor(), LinearRegression(), LogisticRegression()]

for model in models:
    model.fit(X_train, Y_train)

    print(model)
    train_pred = model.predict(X_train)
    print(f'Train Accuracy : {evs(Y_train, train_pred)}')

    test_pred = model.predict(X_test)
    print(f'Test Accuracy : {evs(Y_test, test_pred)}')
    print('|')
    print('|')
    print('|')

