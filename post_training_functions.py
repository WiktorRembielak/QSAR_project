import os
import re
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def find_latest_version(model_name: str, next_version: bool, extention: str = '') -> str:
    if any(f'{model_name}_' in dir_name for dir_name in os.listdir('./saved_models')):
        version = max(map(int, re.findall(rf'{model_name}_(\d+)',
                                          ';'.join(os.listdir('./saved_models')))))
        if next_version:
            version += 1

        return f'./saved_models/{model_name}_{version}{extention}'

    else:
        return f'./saved_models/{model_name}_1{extention}'


# Function for saving and loading trained sequential model's history and hiperparameters
def transfer_history(saved_model_path: str, history_dict: dict = None,
                     save: bool = False, load: bool = False, filename: str = 'history.txt'):
    if save:
        with open(f'{saved_model_path}/{filename}', 'w') as history_file:
            comma = ','
            for key, value in history_dict.items():
                if type(value) == list:
                    value = map(str, value)
                    history_file.write(f'{key}\n{comma.join(value)}\n')
                else:
                    value = str(value)
                    history_file.write(f'{key}\n{value}\n')

            print(f'{filename} saved in {saved_model_path}')
    if load:
        with open(f'{saved_model_path}/{filename}', 'r') as history_file:
            history = [line[:-1] for line in history_file]
            history = dict((history[i], list(map(float, history[i + 1].split(',')))) for i in range(0, len(history), 2))
            return history


def regression_metrics(model, X_train, y_train, y_true, y_predicted):
    mse_train = np.round(mean_squared_error(y_train, model.predict(X_train), squared=False), 2)
    rmse_train = np.round(mean_squared_error(y_train, model.predict(X_train), squared=True), 2)
    mse_test = np.round(mean_squared_error(y_true, y_predicted, squared=False), 2)
    rmse_test = np.round(mean_squared_error(y_true, y_predicted, squared=True), 2)
    r2 = np.round(r2_score(y_true, y_predicted), 2)
    print(f'\nMSE on train set: {mse_train}   RMSE on train set: {rmse_train}')
    print(f'MSE on test set: {mse_test}   RMSE on test set: {rmse_test}\nR2 score: {r2}')
    pass
