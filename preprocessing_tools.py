import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

'''
Class Dataset enables creating an object with methods preparing data, either for training and evaluating a model
or using a model for predictions of bioconcentration classes.
At this point the program needs data consisting of 9 molecular descriptors, which can be calculated directly
from a chemical structure in SMILES format with chemical software.
'''


class Dataset:
    def __init__(
            self,
            path: str,
            index_col=None):
        
        self.raw = self.load_dataset(path, index_col)
        # Attribute self.descriptors have defined column names of molecular descriptors
        # that need to be present in a dataset
        self.descriptors = ['nHM', 'piPC09', 'X2Av', 'MLOGP', 'ON1V', 'N-072',
                            'PCD', 'B02[C-N]',
                            'F04[C-O]']
        self.check_descriptors_completeness()
        self.X = None
        self.X_test = None
        self.X_val = None
        self.y = None
        self.y_test = None
        self.y_val = None
        self.subsets = ('X', 'X_test', 'X_val', 'y', 'y_test', 'y_val')


    def check_descriptors_completeness(self):
        if [desc for desc in self.descriptors if desc not in self.raw.columns]:
            print(f'The dataset has to contain all the needed molecular descriptors with column names in format:\
            \n{self.descriptors}')
            raise Exception(f"descriptor(s) missing")


    @staticmethod
    def load_dataset(path, index_col):
        return pd.read_csv(path, index_col=index_col)


    def split_data(
            self,
            class_column: str,
            reg_column: str,
            classification: bool = False,
            regression: bool = False,
            test_size: float = 0.,
            val_size: float = 0.,
            random_state: int = 42):
        
        # Method splitting the dataset either to:
        # - features subset (X) and class labels / dependent variables subset (y)
        # - training, validation and test subsets.

        # Changes values of self.X, self.y
        # Optionaly changes values of self.X_test, self.y_test, self.X_val, self.y_val

        if classification == regression:
            print('Choose only classification or only regression')
            return

        self.X = self.raw[self.descriptors]
        if classification:
            self.y = self.raw[class_column]
            stratify = self.y

        if regression:
            self.y = self.raw[reg_column]
            stratify = None

        if test_size > 0:
            self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                        shuffle=True,
                                                                        random_state=random_state,
                                                                        stratify=stratify)
            
        if val_size > 0:
            self.X, self.X_val, self.y, self.y_val = train_test_split(self.X, self.y, test_size=val_size,
                                                                      shuffle=True,
                                                                      random_state=random_state,
                                                                      stratify=stratify)


    def remove_outliers(self, num_of_std: int = 3):
        # Perform before scaling

        # Method searches outliers in columns with floating point values and removes records containing them.
        # By default the method recognises values higher or lower by 3 standard deviations
        # than the mean value as outliers.

        # Changes values of self.X and self.y (if dependent values are floating points).

        subset = pd.concat([self.X, self.y], axis=1)
        columns = list(col for col in subset.columns if subset[col].dtype == 'float')

        for col in columns:
            up_limit = subset[col].mean() + subset[col].std() * num_of_std
            low_limit = subset[col].mean() - subset[col].std() * num_of_std
            subset = subset[(subset[col] < up_limit) & (subset[col] > low_limit)]
        print(f'Removed outliers in columns: {columns}')

        self.y = subset.pop(self.y.name)
        self.X = subset


    def scale(self, standard: bool = False, minmax: bool = False):
        # If user wants to remove outliers, it should be performed before scaling

        # Standarization or normalization of columns with floating point values
        # Changes values of self.X and, if it's present in the dataset object, self.X_test

        if standard == minmax:
            raise ValueError('Scaling failed. Choose only standarization or only normalization')
        
        if standard:
            scaler = StandardScaler()
            scaling = 'Standarization'

        if minmax:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaling = 'Normalization'

        columns = list(col for col in self.X.columns if self.X[col].dtype == 'float')
        self.X.loc[:, columns] = scaler.fit_transform(self.X.loc[:, columns])
        if self.X_test is not None:
            self.X_test.loc[:, columns] = scaler.fit_transform(self.X_test.loc[:, columns])

        # Check if scaling was performed properly
        if standard:
            failed = [col for col in columns if np.round(self.X.loc[:, col].mean(), 4) != 0
                      or np.round(self.X.loc[:, col].std(), 2) != 1]
        if minmax:
            failed = [col for col in columns if (np.round(self.X.loc[:, col].min(), 2) != 0)
                      or np.round(self.X.loc[:, col].max(), 2) != 1]

        if not failed:
            print(f'{scaling} done properly in columns: {columns}')
        else:
            print(f'{scaling} failed in columns: {failed}')


    def resample(self):
        # Fixing training on imbalanced data with SMOTE resampling
        self.X, self.y = SMOTE().fit_resample(self.X, self.y)


def adjust_shape(subset):
    # Changing shape of class labels column to format required by sequential model
    subset = np.array(subset).reshape(-1, 1)
    enc = OneHotEncoder()
    return pd.DataFrame(enc.fit_transform(subset).toarray())
