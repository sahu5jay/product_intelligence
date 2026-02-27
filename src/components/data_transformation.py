import sys
import os
import numpy as np 
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Pipeline Initiation')

            # Correctly aligned columns
            numerical_cols = ["OverallQual", "GrLivArea", "GarageCars"]
            ordinal_cols = ["ExterQual", "KitchenQual", "BsmtQual", "GarageFinish"]
            
            # Categories mapped to the ordinal_cols order
            qual_categories = ['Po', 'Fa', 'TA', 'Gd', 'Ex'] # For ExterQual and KitchenQual
            bsmt_categories = ['NoBsmt', 'Po', 'Fa', 'TA', 'Gd', 'Ex'] # For BsmtQual
            finish_categories = ['Unf', 'RFn', 'Fin'] # For GarageFinish

            # Numerical Pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Ordinal Pipeline
            # NOTE: The order in the list below must match 'ordinal_cols'
            ord_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder', OrdinalEncoder(categories=[
                    qual_categories, # ExterQual
                    qual_categories, # KitchenQual
                    bsmt_categories, # BsmtQual
                    finish_categories # GarageFinish
                ])),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('ord_pipeline', ord_pipeline, ordinal_cols)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'SalePrice'
            # We only keep the features the user will actually provide
            input_cols = ["OverallQual", "GrLivArea", "GarageCars", "ExterQual", "KitchenQual", "BsmtQual", "GarageFinish"]

            input_feature_train_df = train_df[input_cols]
            target_feature_train_df = np.log1p(train_df[target_column_name])

            input_feature_test_df = test_df[input_cols]
            target_feature_test_df = np.log1p(test_df[target_column_name])

            # Transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)