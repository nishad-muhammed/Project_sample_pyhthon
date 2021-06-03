from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
from matplotlib import pyplot as plt
# %matplotlib inline


# read dataset
dataset = pd.read_csv('diabetes.csv')

# find missing values
dataset.isnull().sum(axis=0)
dataset.head(15)

# how many column have zero
(dataset == 0).sum()

# convert 0 to its mean value
mean_bmi = dataset['BMI'].mean(skipna=True)
dataset.loc[dataset.BMI == 0, 'BMI'] = mean_bmi

mean_Glucose = dataset['Glucose'].mean(skipna=True)
dataset.loc[dataset.Glucose == 0, 'Glucose'] = mean_Glucose

mean_BloodPressure = dataset['BloodPressure'].mean(skipna=True)
dataset.loc[dataset.BloodPressure == 0, 'BloodPressure'] = mean_BloodPressure

# how many column have zero
(dataset == 0).sum()

# make input(X) and output(Y)
X = dataset[['Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction']]
Y = dataset.iloc[:,  -1]

mmScaler = MinMaxScaler(feature_range=(0, 1))
rescaleX = mmScaler.fit_transform(X)
rescaleX

X[:].mean()
X[:].std()
sScaler = StandardScaler()
reScaleX = sScaler.fit_transform(X)
reScaleX
reScaleX[:, 0].mean()
reScaleX[:, 0].std()

# normalization
norm = Normalizer()
norm.fit_transform(X)

# binaary
bina = Binarizer(threshold=5).fit(X)
bina.fit_transform(X)

#plot to identify more rlation to the output
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax.bar(dataset.Outcome,dataset.BloodPressure)
# plt.show()

# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax.bar(dataset.Outcome,dataset.Glucose)
# plt.show()

# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax.bar(dataset.Outcome,dataset.DiabetesPedigreeFunction	)
# plt.show()

# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax.bar(dataset.Outcome,dataset.BMI)
# plt.show()

#training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,train_size=0.8)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

model.predict_proba(X_test)

model.score(X_test,y_test)
    
 #pickle
pickle.dump(model, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
