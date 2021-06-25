# Hyper parameter Tuning

# Import libraries
import pandas as pd
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Reading the dataset
dataset = pd.read_csv('Dataset/Churn_Modelling.csv')

print(dataset.head())

# define X(independent) and y(dependent) features
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

# Create dummy variables
geography = pd.get_dummies(X['Geography'], drop_first=True)
gender = pd.get_dummies(X['Gender'], drop_first=True)

# Concatenate the Data Frames
X = pd.concat([X, geography, gender], axis=1)

# Drop Unnecessary columns
X = X.drop(['Geography', 'Gender'], axis=1)

# Splitting the dataset into train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Perform Hyperparameter Optimization
def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes, input_dim=X_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))

    model.add(Dense(units=1, kernel_initializer='glorot_uniform',
                    activation='sigmoid'))  # Note: no activation beyond this point

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, verbose=0)

layers = [(150,), (160, 100), (140, 100, 500)]
activations = ['sigmoid', 'relu']
param_grid = dict(layers=layers, activation=activations, batch_size=[128, 256], epochs=[30])
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

grid_result = grid.fit(X_train, y_train)

[grid_result.best_score_, grid_result.get_params]

print(grid_result.best_score_, grid_result.get_params)
