import streamlit as st
from tensorflow.keras.preprocessing.image  import load_img,img_to_array
from keras.models import load_model
import numpy as np


model=load_model("wild.h5")

st.title("Multiclass Image Classification")
st.subheader("Wild Animals ")

file=st.file_uploader("Upload on image",type=["jpg","png","jpeg","jfif"]) 
st.balloons()

if file is not None:
   st.image(file,caption="uploaded image")  
   img=load_img(file,target_size=(224,224))
   img_array=img_to_array(img)/255.0
   img_expand=np.expand_dims(img_array,axis=0)

if st.button("Classify"): 

    predict=model.predict(img_expand)
    predict_class=np.argmax(predict,axis=1)
    names=["Cheetah","Fox","Hyena","Lion","Tiger","Wolf"]

    if predict_class==0:
       st.success(f"This is {names[0]}")
    elif predict_class==1:
       st.success(f"This is {names[1]}")
    elif predict_class==2:
        st.success(f"This is {names[2]}")
    elif predict_class==3:
        st.success(f"This is {names[3]}")
    elif  predict_class==4:
        st.success(f"This is {names[4]}")
    elif predict_class==5:
        st.success(f"This is {names[5]}")



