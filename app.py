from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing
st.title('🏠House Price prediction using ML')

st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRvQdqIasHkDTf5733FK14z5mPQ18VPhg_R_Q&s')

df = pd.read_csv('house_data.csv')
X = df.iloc[:,:-3]
y = df.iloc[:,-1]

final_X = X
scaler = StandardScaler()
scaled_X = scaler.fit_transform(final_X)

st.sidebar.title('Select House features: ')
st.sidebar.image('data:image/png;base64,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')
all_value = []
for i in final_X:
  min_value = final_X[i].min()
  max_value = final_X[i].max()
  result = st.sidebar.slider(f'select {i} value',min_value,max_value)
  all_value.append(result)

user_X = scaler.transform([all_value])

@st.cache_data
def ML_model(X,y):
  model = RandomForestRegressor()
  model.fit(X,y)
  return model

model = ML_model(scaled_X,y)
house_price = model.predict(user_X)[0]

final_price = round(house_price * 100000,2)

with st.spinner('Prediction House Price'):
  import time 
  time.sleep(2)

st.success(f'Estimated house price is:$ {final_price}')
st.markdown('''**design and developed by: Kaushiki Trivedi**''')
