import streamlit as st
import time
from io import StringIO
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import cv2 as cv


@st.cache(allow_output_mutation=True)
def load_model_axial():
    with st.spinner('Model is being loaded for axial...'):
        # model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'D:\E\DXAssignment\streamlit\braintumor\models\best_1.pt', force_reload=True)
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'D:\E\DXAssignment\YOLOv5Projects\yolov5\runs\train\exp16\weights\best_1.pt', force_reload=True)
    return model

@st.cache(allow_output_mutation=True)
def load_model_coronal():
    with st.spinner('Model is being loaded for coronal...'):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'D:\E\DXAssignment\streamlit\braintumor\models\best_2.pt', force_reload=True)
    return model

@st.cache(allow_output_mutation=True)
def load_model_sagittal():
    with st.spinner('Model is being loaded for sagittal...'):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'D:\E\DXAssignment\streamlit\braintumor\models\best_3.pt', force_reload=True)
    return model

source = ("1. Axial Image", "2. coronal Image", "3. sagittal Image")
source_index = st.sidebar.selectbox("Select Data Source", range(len(source)), format_func=lambda x: source[x])

if __name__ == '__main__':

    st.title('Brain Tumor Detection using Yolov5')

    if source_index == 0:
        model = load_model_axial()
    elif source_index == 1 :
        model = load_model_coronal()
    elif source_index == 2 :
        model = load_model_sagittal()
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload an Image", type=['png', 'jpeg', 'jpg'])
    
    if uploaded_file is not None:
        with st.spinner(text='Loading Image...'):
            st.sidebar.image(uploaded_file)
            picture = Image.open(uploaded_file)
            picture = model(picture.resize((640,640)))
            st.image(np.squeeze(picture.render()))

#         # picture = picture.save(f'data/images/{uploaded_file.name}')
#         # opt.source = f'data/images/{uploaded_file.name}'