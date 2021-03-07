import streamlit as st
from streamllit_func import respiratory_pathology_detect
import io
import os
from PIL import Image
st.sidebar.header('Select dashboard')
page = st.sidebar.selectbox("Select a page", ["Results","App"])

#page = st.sidebar.selectbox("Select a page", ["App"])
st.set_option('deprecation.showfileUploaderEncoding', False)


if page == "App":
    
    st.title('Respiratory Pathology Detection App')
    st.markdown('<h1> Select patient audio files to check their respiratory pathology </h1>',unsafe_allow_html=True)
    uploaded_file = None

    option = st.selectbox(
    'Patient numbers',
    ('Patient 1', 'Patient 2', 'Patient 3','Patient 4', 'Patient 5', 'Patient 6','Patient 7'))

    st.write('You selected:', option)
    if option=='Patient 1':
        filename='patient_data/101_1b1_Al_sc_Meditron.wav'
    elif option=='Patient 2':
        filename='patient_data/102_1b1_Ar_sc_Meditron.wav'
    elif option=='Patient 3':
        filename='patient_data/103_2b2_Ar_mc_LittC2SE.wav'
    elif option=='Patient 4':
        filename='patient_data/104_1b1_Ar_sc_Litt3200.wav'
    elif option=='Patient 5':
        filename='patient_data/105_1b1_Tc_sc_Meditron.wav'
    elif option=='Patient 6':
        filename='patient_data/106_2b1_Pl_mc_LittC2SE.wav'
    elif option=='Patient 7':
        filename='patient_data/107_2b3_Ar_mc_AKGC417L.wav'

    st.markdown('<h1> To check pathology: </h1>',unsafe_allow_html=True)
    
    if st.button('Proceed'):
        result=respiratory_pathology_detect(filename)
        st.write('Result: %s' % result)

elif page == "Results":
    st.title('Respiratory Pathology Detection Results')
    st.markdown('<h2> This page is dedicated to demonstrate the finding of our research work </h1>',unsafe_allow_html=True)
    st.markdown('<h3> Data Augmention was done on ICBHI dataset </h3>',unsafe_allow_html=True)
    image = Image.open('results/data_augment.PNG')
    st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

    st.markdown('<h3> Feature transformation </h3>',unsafe_allow_html=True)
    image = Image.open('results/feature.png')
    st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

    st.markdown('<h3> Method flow </h3>',unsafe_allow_html=True)
    image = Image.open('results/respi.jpg')
    st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

    st.markdown('<h3> Model architecture </h3>',unsafe_allow_html=True)
    image = Image.open('results/model_plot.png')
    st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

    st.markdown('<h3> Training and Validation accuracy </h3>',unsafe_allow_html=True)
    image = Image.open('results/kfoldACC_44.png')
    st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

    st.markdown('<h3> Training and Validation loss </h3>',unsafe_allow_html=True)
    image = Image.open('results/kfoldLoss_44.png')
    st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')


