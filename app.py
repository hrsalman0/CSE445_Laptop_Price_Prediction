import streamlit as st
import pickle
import numpy as np
import math
# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))
st.title("CSE445: Laptop Price Prediction Project (Group 8)")
# brand
company = st.selectbox('Brand',df['Company'].unique())

# laptop Type
type = st.selectbox('Type',df['TypeName'].unique())

# Ram(GB)
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# weight of laptop
weight = st.number_input('Weight of the Laptop')

# touchScreen
touch = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# screensize
screen_size = st.number_input('Screen Size')

# resolution

resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900',
                                                '3840x2160', '3200x1800', '2880x1800', '2560x1600',
                                                '2560x1440', '2304x1440'])
# cpu brand
cpu = st.selectbox('CPU',df['Cpu brand'].unique())
# hdd
hdd = st.selectbox('HDD(GB)', [0, 128, 256, 512, 1024, 2048])
# sdd
sdd = st.selectbox('SDD(GB)', [0, 8, 128, 256, 512, 1024])

# Gpu Brand
gpu = st.selectbox('GPU', df['Gpu Brand'].unique())
# OS
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict'):
    # query

    if touch == "Yes":
        touch = 1
    else:
        touch = 0

    if ips == "Yes":
        ips = 1
    else:
        ips = 0

    # split of resolution
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])

    # ppi calculation
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size

    query = np.array([company, type, ram, weight, touch, ips, ppi, cpu, hdd, sdd, gpu, os])

    query = query.reshape(1,12)
    st.title("The Predicted Price: "+str(int(np.exp(pipe.predict(query)[0]))*1.30))

