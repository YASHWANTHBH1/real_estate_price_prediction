import json
import pickle
import numpy as np

__locations=None
__data_columns=None
__model=None
def get_estimated_price(location,sqft,bath,bhk):
    global __data_columns,__model
    if __data_columns is None:
        print("Artifacts are not loaded. Please load them first using `load_saved_arteffects()`.")
        return None
    try :
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    X = np.zeros(len(__data_columns))
    X[0] = sqft
    X[1] = bath
    X[2] = bhk
    if loc_index >= 0:
        X[loc_index] = 1
    return round(__model.predict([X])[0],2)
def get_location_name():
    global __locations
    return __locations

def load_saved_arteffects():
    print("loading saved arteffects ....")
    global __data_columns
    global __locations

    with open("./arteffects/column.json",'r') as f:
        __data_columns=json.load(f)['data_columns']
        __locations=__data_columns[3:]
    print(f"Loaded locations: {__locations}")
    global __model
    with open("./arteffects/banglore_home-price_ml_model.pkl",'rb') as f:
        __model=pickle.load(f)
    print("loading saved arteffects...done")



if __name__ =='__main__':
    load_saved_arteffects()
    print(get_estimated_price('1st phase jp nagar',1000,2,2))
    print(get_estimated_price('abbigere',1000,2,2))
    print(get_estimated_price('Elipura', 1000, 2, 2))
    print(get_location_name())