import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing
st.title('🏠House Price prediction using ML')

st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRvQdqIasHkDTf5733FK14z5mPQ18VPhg_R_Q&s')

data = fetch_california_housing()

X = pd.DataFrame(data['data'],
            columns = data['feature_names'])

final_X = X.iloc[:,:-2]
scaler = StandardScaler()
scaled_X = scaler.fit_transform(final_X)
