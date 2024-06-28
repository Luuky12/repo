import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np 

st.header('Klasifikasi Gambar Kucing Dan Anjing')
model = load_model('Image_classify.keras')

data_cat = ['Anjing', 'Kucing']

img_height = 100
img_width = 100

image =st.text_input('Masukan Nama gambar','cat.jpg')

image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image, width=200)
st.write('Gambar ini adalah : ' + data_cat[np.argmax(score)])
st.write('Dengan Ketepatan ' + str(np.max(score)*100))
