import pandas as pd
import numpy as np

# Load in the csv
df = pd.read_csv('Border_Crossing_Entry_Data.csv')

# Change the Value column so it's 1 if greater than a certain value and 0 if not
a = np.array(df['Value'].values.tolist())
df['Value'] = np.where(a > 5000, 1, a).tolist()
b = np.array(df['Value'].values.tolist())
df['Value'] = np.where((b <= 5000) & (b > 1), 0, b).tolist()

# Define the y variable
y = df['Value'] 

# Drop the label column
df = df.drop('Value', axis=1)

# one-hot encode the data and normalize it up to 20
df = pd.get_dummies(df)
X = df.values
X = (X-X.min())/(X.max() - X.min()) * 20

# Split the data into training and testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Design the model
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(436, input_dim=436, activation='relu'))
model.add(Dense(872, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(436, activation='relu'))
model.add(Dense(218, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(109, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile,fit and evaluate the model
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=10, batch_size=128)
results = model.evaluate(X_test, y_test, batch_size=128)
print('test loss, test acc:', results)

