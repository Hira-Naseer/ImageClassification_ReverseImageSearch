import streamlit as st
import tensorflow as tf
import numpy as np
from urllib.request import urlopen
# rdn = RDN(weights='psnr-small')
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import pickle
from numpy.linalg import norm
import os
from PIL import Image
from keras.layers import *
import numpy as np
from urllib.request import urlopen
import io
from super_image import HanModel, ImageLoader, EdsrModel
from PIL import Image
import requests



resize = (256,256)
is_loading = False
defualt_img_name = 'image.jpg'
defualt_img_path = f'uploads/{defualt_img_name}'


def pipeline(test_img):
    class_dict = {
        'vehicle': {0:'airplane', 1:'automobile', 2:'ship', 3:'truck'},
        'non_vehicle':{0:'bird',1:'cat',2:'deer',3:'dog',4:'frog',5:'horse'}
    }

    base_path = 'cifar10/train'
    images_path = {}
    for folders in os.listdir(base_path):
        for fol in os.listdir(f'{base_path}/{folders}'):
            list = []
            for file in os.listdir(f'{base_path}/{folders}/{fol}'):

                list.append((f'{base_path}/{folders}/{fol}/{file}'))
            images_path[f'{folders}/{fol}'] = list

    model = ResNet50(weights='imagenet', include_top=False, input_shape=(32,32,3))
    model.trainable = False

    model = keras.Sequential([
        model,
        GlobalMaxPool2D()
    ])

    test_img = Image.open(test_img)
    b_model = keras.models.load_model('models/best_model_weights_Binary.h5')
    cnn_model_veh = keras.models.load_model('models/model_weights_cVeh.h5')
    cnn_model_nonVeh = keras.models.load_model('models/model_weights_cNonVe.h5')

    img = test_img.resize((150,150))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    pred_img = b_model.predict(img)


    predictions = np.array([])
    if pred_img < 0.5  and pred_img >= 0:
        predictions = cnn_model_nonVeh.predict(img)
    elif(pred_img > 0.5):
        predictions = cnn_model_veh.predict(img)

    predicted_class = np.argmax(predictions,axis=-1)
    prediction = ""
    if pred_img < 0.5:
        prediction = 'non_vehicle/'
        prediction = prediction + class_dict['non_vehicle'][predicted_class[0]]
    elif pred_img > 0.5:
        prediction = 'vehicle/'
        prediction = prediction + class_dict['vehicle'][predicted_class[0]]

    t_img = test_img.resize((32,32))
    img_array = image.img_to_array(t_img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    pre_model = model.predict(preprocessed_img).flatten()
    normalized_result = pre_model/norm(pre_model)

    features_list = {}
    filenames = {}

    for files in images_path:
        name = files.replace('/','_')
        features_list[files] = np.array(pd.read_pickle(f'featuresList/features_{name}.pkl'))
        filenames[files] = pd.read_pickle(open(f'filenames/filenames_{name}.pkl','rb'))


    neighbors = NearestNeighbors(n_neighbors = 16,algorithm='brute',metric='euclidean')
    neighbors.fit(features_list[prediction])

    distances, indices = neighbors.kneighbors(normalized_result.reshape(1,-1))
    file_list = filenames[prediction]

    list = []
    for i in indices[0]:
        list.append(file_list[i])

    return prediction.split('/')[0],prediction.split('/')[1],list




def super_resolution(paths):
    image = Image.open(paths)
    model = EdsrModel.from_pretrained('eugenesiow/edsr', scale=2)
    inputs = ImageLoader.load_image(image)
    preds = model(inputs)
    ImageLoader.save_image(preds, f'output/img.png')
    preds = Image.open('output/img.png')

    return preds




def show_col_images():
    test_image = defualt_img_path
    is_loading = True

    with st.spinner('Loading...'):
        binaryClass,Class,paths = pipeline(test_image)
    st.success('Done!')


    display_image = Image.open(test_image)
    with st.expander('display image',expanded=True):
        st.image(display_image)

    st.write(f"The image given is <b>{binaryClass.split('_')[0]}</b> <b>{binaryClass.split('_')[1]}</b> and is classified as <b>{Class}</b>", unsafe_allow_html=True)

    st.write('The similar images are:')

    co1, co2,co3,co4 = st.columns(4)
    for i in range(0, len(paths), 4):
        with co1:
            st.image(super_resolution(paths[i]), use_column_width=True)
        with co2:
            st.image(super_resolution(paths[i+1]), use_column_width=True)
        with co3:
            st.image(super_resolution(paths[i+2]), use_column_width=True)
        with co4:
            st.image(super_resolution(paths[i+3]), use_column_width=True)

st.set_page_config(page_title="Image Classifier and Research", page_icon="🧐", layout="wide", initial_sidebar_state="expanded")
col1, col2 = st.columns(2)
with st.container():
    with col1:
        col1.markdown('<h1 style="right-margin:0px;right-padding:0px;text-align: right;">Learning through image</h1>',
                    unsafe_allow_html=True)

    with col2:
        col2.image("logo.jpg",width=100)

def save_upload_file():
    try:
        with open(defualt_img_path,'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0

uploaded_file = st.sidebar.file_uploader("Choose a file")
url = st.sidebar.text_input("Enter a url")
pic = st.sidebar.camera_input("Take a picture to Find More")
save_button = st.sidebar.button("Save")

if pic is not None:
    if save_button:
        with open(defualt_img_path,'wb') as f:
            f.write(pic.getbuffer())
            st.sidebar.success("Saved")
            show_col_images()

elif uploaded_file is not None:
    if save_upload_file():
        show_col_images()

elif url is not None or url != "":
    if url.startswith('http') or url.startswith('https'):
        image_bytes = urlopen(url).read()
        save_image = Image.open(io.BytesIO(image_bytes))
        save_image.save(defualt_img_path)
        show_col_images()


else:
    st.header("Some error occured in file ulploads")

