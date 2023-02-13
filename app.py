
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

class info(BaseModel):
    Mean: float
    STD: float

class Key(BaseModel):
    model: str
    HT: info
    PPT: info
    RRT: info
    RPT: info


clf = None
app = FastAPI(title="Test REST API", description="API keystrokes", version="1.0")

@app.get('/')
def get_root():
    return {'message': 'Welcome to the SMS spam detection API'}



def load_model(name:str):
    clf = load(name + '.joblib')
    return clf


@app.post('/predict', tags=["predictions"])
async def get_prediction(kstroke: Key):
    features = [] 
    
    data = list(dict(kstroke).values())

    model_name = data[0]
    clf = load_model(model_name)

    for i in data[1:]:
        features += list(dict(i).values())


    prediction = clf.predict([features])

    return {"prediction": prediction[0].item()}
