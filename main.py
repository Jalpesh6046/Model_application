from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

origins = ["*"]

app.add_middleware(CORSMiddleware, allow_origins= origins,
                   allow_credentials = True,
                   allow_methods= ["*"],
                   allow_headers=["*"],
                   )

class model_input(BaseModel):
    
    Index : int
    experience : int 
    test_score : int 
    
salary_predict = pickle.load(open('model.pkl','rb'))

@app.post('/salary_prediction')
def salary_pred(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    Index = input_dictionary['Index']
    experience = input_dictionary['experience']
    test_score = input_dictionary['test_score']
    
    input_list = [Index, experience, test_score]
    
    prediction = salary_predict.predict([input_list])
    
    return prediction