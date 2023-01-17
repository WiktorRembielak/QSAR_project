import os
import re
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

'''
This script is a place for functions used in the project other than ETL
'''


def find_latest_version(path: str, name: str, next_version: bool, extention: str = '') -> str:
    if any(f'{name}_' in dir_name for dir_name in os.listdir(path)):
        version = max(map(int, re.findall(rf'{name}_(\d+)',
                                          ';'.join(os.listdir(path)))))
        if next_version:
            version += 1

        return f'{path}/{name}_{version}{extention}'

    else:
        return f'{path}/{name}_1{extention}'
