import pandas as pd
from sklearn.linear_model import LinearRegression

import joblib
import uvicorn

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel


def predict(model, total_square=170, rooms=2, floor=27):
    """Функция для получения предсказания"""
    data_inp = pd.DataFrame({
        "total_square": [total_square],
        "rooms": [rooms],
        "floor": [floor],
    }
    )
    return model.predict(data_inp)[0]


# Чтение данных
data = pd.read_csv('realty_data.csv')

# Подготовка данных к обучению
df = data.drop(labels=['product_name', 'period', 'postcode', 'address_name',
                       'lat', 'lon', 'district', 'area',
                       'description', 'source', 'object_type', 'city',
                       'settlement'], axis=1)
df['rooms'] = df['rooms'].fillna(0)

x = df.drop(labels='price', axis=1)
y = df['price']

# Обучение
lr = LinearRegression()
lr.fit(x, y)

app = FastAPI()

model = lr


class ModelRequestData(BaseModel):
    total_square: float
    rooms: int
    floor: int


class Result(BaseModel):
    result: float


@app.get("/health")
def health():
    return JSONResponse(content={"message": "It's alive!"}, status_code=200)


@app.get("/predict_get")
def predict_get():
    return JSONResponse(content={"message": f"The cost of a flat is: {predict(lr):.2f} rubles!"}, status_code=200)


@app.post("/predict_post", response_model=Result)
def predict_post(data: ModelRequestData):
    input_data = data.dict()
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)