import streamlit as st
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (40,20)

from helpers import *
from datetime import datetime, timedelta

st.markdown(footer,unsafe_allow_html=True)
st.title('Key Features')

property_id = st.sidebar.text_input('Rightmove Property ID', '126302255')

st.sidebar.write("Disclaimer: This non-profit web-app is for illustrative purposes, only. The code may produce errors for some property IDs as it's not been fully tested.")

page_model = page_model_scraper(property_id=property_id)

primaryPrice = page_model['propertyData']['prices']['primaryPrice']

displayAddress = page_model['propertyData']['address']['displayAddress']
postcode = page_model['analyticsInfo']['analyticsProperty']['postcode']

key_features = (['**' + primaryPrice + '**'] + 
                page_model['propertyData']['keyFeatures'] + 
                ['Address: **' + displayAddress + ', '+ postcode + '**'])

st.markdown("The key features of this property are: \n- " + 
            '\n- '.join([x for x in key_features])
           )

Image_URL = floorplan_url_finder(page_model=page_model)

img = open_floorplan_url(floorplan_url=Image_URL)

st.title('Floorplan')
st.image(img, caption='Floorplan of the Property')

thresh = transform_floorplan_image(img=img)

df_img = pytesseract_image(transformed_image=thresh)

st.title('Plot Characters')
fig, ax = plt.subplots()
ax.scatter(df_img['left'].tolist(), df_img['top'].tolist())
for i, txt in enumerate(df_img['char'].tolist()):
    ax.annotate(txt, (df_img['left'].tolist()[i],
                      df_img['top'].tolist()[i]), 
                
                textcoords='data', 
                
                fontsize=28)

st.pyplot(fig)

df_img = cluster_characters(image_dataframe=df_img)

st.title('Cluster Characters')
fig, ax = plt.subplots()
ax.scatter(df_img['left'].tolist(), df_img['top'].tolist(), 
           s=[1500 for n in range(df_img.shape[0])], 
           c=df_img['cluster'].tolist(), cmap='Paired')
st.pyplot(fig)

df_cc = loop_through_clusters(image_dataframe=df_img)

df_cc_grouped = group_cluster_characters(cluster_dataframe=df_cc)
st.title('Cleaned Cluster Results')
st.dataframe(df_cc_grouped)