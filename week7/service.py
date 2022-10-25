import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray
import sklearn
from pydantic import BaseModel

# make sure the field match (name and data type) to the fields you 
# expect as input
class CreditApplication(BaseModel):
    seniority: int
    home: str
    time: int
    age: int
    marital: str
    records: str
    job: str
    expenses: int
    income: float
    assets: float
    debt: float
    amount: int
    price: int


#use keyword 'latest' to pull patest model from repo
#model_ref = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")
model_ref = bentoml.sklearn.get('mlzoomcamp_homework:jsi67fslz6txydu5')

#dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service('coolmodel.bentomodel', runners=[model_runner])


@svc.api(input=NumpyNdarray(), output=JSON())
async def classify(vector):
    #application_data = credit_applcation.dict()
 
    #vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)
    print(prediction)

    result = prediction[0]

    if result > 0.5:
        return {'status': 'DECLINED'}

    elif result > 0.23:
        return {'status': 'MAYBE'}

    else:
        return {'status': 'APPROVED'}
