from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from tools import find_latest_version
import yaml
import pandas as pd

# Loading config file and subsets
with open('config.yaml') as file:
    config = yaml.safe_load(file)['train']

if config['processed_data_version'] == 'recent':
    processed_data = find_latest_version('./data', 'processed_data', next_version=False)
else:
    processed_data = './data/processed_data_' + config['processed_data_version']


X = pd.read_csv(f'{processed_data}/X.csv', index_col=0)
y = pd.read_csv(f'{processed_data}/y.csv', index_col=0)

# Setting hyperparameters variables
l1 = config['l1']
l2 = config['l2']
dropout = config['dropout']
batchSize = config['bathSize']
epochs = config['epochs']


# Defining sequential model's architecture
model = Sequential()

model.add(Dense(256, input_dim=X.shape[1], activation='relu'))
model.add(Dropout(dropout))

model.add(Dense(128, activation='relu', bias_regularizer=regularizers.L1L2(l1, l2)))
model.add(Dropout(dropout))

model.add(Dense(64, activation='relu', bias_regularizer=regularizers.L1L2(l1, l2)))
model.add(Dropout(dropout))

model.add(Dense(32, activation='relu', bias_regularizer=regularizers.L1L2(l1, l2)))
model.add(Dropout(dropout))

model.add(Dense(3, activation='softmax', bias_regularizer=regularizers.L1L2(l1, l2)))

# Model configuration
model.compile(optimizer='adam',
              loss='CategoricalCrossentropy',
              metrics=['acc'])
print(model.summary())


# Training the model
history = model.fit(X, y,
                    batch_size=batchSize,
                    epochs=epochs,
                    validation_split=0.2)


# Exporting model
saved_model_path = find_latest_version('./saved_models', 'classification_sequential', next_version=True)
model.save(saved_model_path)
