import sys
import pickle
from src.exception import CustomException

def save_object(file_path, object):
    '''
    Save a python object to a file using pickle.
    
    parameters:
    file_path: str : The path where the object should be saved.
    object: The python object to be saved.
    '''
    try:
        with open(file_path, "wb") as file:
            pickle.dump(object, file)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    '''
    Load a python object from a pickle file.
    
    parameters:
    file_path: str : The path from where the object should be loaded.
    
    returns:
    The loaded python object.
    '''
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e, sys)