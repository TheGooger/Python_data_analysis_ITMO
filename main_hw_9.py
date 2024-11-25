import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st


def predict(model, total_square=170, rooms=2, floor=27):
    """Функция для получения предсказания"""
    data_inp = pd.DataFrame({
        "total_square": [total_square],
        "rooms": [rooms],
        "floor": [floor],
    }
    )
    return model.predict(data_inp)


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

# Настройка интерфейса в streamlit
st.set_page_config(page_title="Realty App")

# Получение параметров из интерфейса
total_square = st.sidebar.number_input("What is the total square? ", 8, 2070, 100)

rooms = st.sidebar.selectbox("How many rooms? ", (0, 1, 2, 3, 4, 5,
                                                  6, 7, 8, 9, 10,
                                                  11, 12, 13, 14, 15))

floor = st.sidebar.number_input("Which floor? ", 1, 66, 1)

# Формирование датафрейма-инпута из полученных параметров
inputDF = pd.DataFrame(
    {
        "total_square": total_square,
        "rooms": rooms,
        "floor": floor,
    },
    index=[0],
)

# Небольшой декор
st.image("imgs/flat.jpg", use_column_width=True)
# Запрос на ввод параметров и нажате кнопки
st.write("Choose the parameters and click 'Calculate' to calculate the price")
# Нажатие кнопки и предсказание
if st.button('Calculate'):
    preds = lr.predict(inputDF)[0]
    st.write(f"Cost of the flat is: {round(preds/10**6, 3)} million rubbles")
