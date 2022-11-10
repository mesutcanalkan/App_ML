from matplotlib import pyplot as plt
import cv2
import numpy as np
import pandas as pd
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.max_colwidth', 500)
# %matplotlib inline
# plt.rcParams["figure.figsize"] = (40,20)
import pytesseract
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import requests
import imageio
from bs4 import BeautifulSoup
import json
import urllib.request
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

footer="""<style>
#MainMenu {visibility: hidden;}
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by <a style='display: block; text-align: center;' href='https://medium.com/@mesutcanalkan' target="_blank">Mesut Alkan</a></p>
</div>
"""

def page_model_scraper(rightmove_url: str, keyword_to_look: str):

    page = requests.get(rightmove_url)

    soup = BeautifulSoup(page.content, 'html.parser')

    all_scripts = soup.find_all('script')

    page_model = json.loads(all_scripts[
        
        max(
            
                [index for (index, script) in enumerate(all_scripts) if keyword_to_look in str(script)]
                
            )

                            ].get_text(strip=True)[len(keyword_to_look):])

    return page_model


def floorplan_url_finder(page_model: dict):

    floorplans = page_model['propertyData']['floorplans']

    if type(floorplans)==dict:

        floorplan_url = floorplans['url']

    else:
        
        floorplan_url = floorplans[0]['url']

    return floorplan_url     


def open_floorplan_url(floorplan_url: str):

    req = urllib.request.urlopen(floorplan_url)
    img = imageio.v2.imread(req.read())

    return img


def transform_floorplan_image(img: np.ndarray):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    return thresh

def pytesseract_image(transformed_image: np.ndarray):

    df_img = pd.DataFrame([x.split(' ') for x in 
    
        pytesseract.image_to_boxes(transformed_image).split('\n')], 
        
                            columns=['char', 'left', 'top', 'width',        'height', 'other'])
    
    df_img = df_img[ ~ df_img['left'].isnull()]
    # dropping whitespace characters like
    # [',' '.' '/' '~' '"' "'" ':' '°' '-' '|' '=' '%' '”']
    df_img = df_img[ ~ df_img['char'].str.contains(r'[^\w\s]')].reset_index(drop=True)
    df_img[['left', 'top', 'width', 'height']] = df_img[['left', 'top', 'width', 'height']].astype(int)
    
    return df_img

def cluster_characters(image_dataframe: pd.DataFrame):

    X = StandardScaler().fit_transform(image_dataframe[['left', 'top']].values)
    db = DBSCAN(eps=0.19, min_samples=10)
    db.fit(X)
    y_pred = db.fit_predict(X)
    # plt.figure(figsize=(10,6))
    # plt.scatter(X[:,0], X[:,1],c=y_pred, cmap='Paired')
    # plt.title("Clusters determined by DBSCAN")
    image_dataframe['cluster'] = pd.Series(y_pred)
    # df_img.groupby(['cluster'])['char'].apply(lambda x: ' '.join(x)).reset_index()

    return image_dataframe


def loop_through_clusters(image_dataframe: pd.DataFrame):

    df_cc = image_dataframe.copy().reset_index(drop=True)

    for cluster_no in df_cc['cluster'].unique():
        
        index_char_top_list = []
        
        # if the data point is not an outlier
        if cluster_no!=-1:
        
            index_char_top_list = [
            (index, char, top) for index, char, top in 
                
                
                zip(df_cc[(df_cc['cluster']==cluster_no)].index, 
                    df_cc[(df_cc['cluster']==cluster_no)]['char'].values, 
                    df_cc[(df_cc['cluster']==cluster_no)]['top'].values)
                                        if
                                        char.isdigit()
                                        ]
        
        if index_char_top_list:
        
                df_cc = df_cc[
                                ~ ((df_cc['cluster']==cluster_no) & (df_cc['top'] <= ( index_char_top_list[0][2] - 5 )))
                            ]
    return df_cc    


def dimension_splitter(input_text:str):
    
    input_text_len = len(input_text)
    if input_text_len%2==0:
        split_text_by = int(input_text_len/2)
    else:
        split_text_by = int(input_text_len/2+0.5)
    
    dim1 = input_text[:split_text_by]
    dim2 = input_text[split_text_by:]
    
    dim1 = float('{}.{}'.format(dim1[:-2], dim1[-2:]))
    dim2 = float('{}.{}'.format(dim2[:-2], dim2[-2:]))
    return dim1, dim2


def group_cluster_characters(cluster_dataframe: pd.DataFrame):

    df_cc = cluster_dataframe.reset_index(drop=True)
    
    df_cc_grouped = df_cc.groupby(['cluster'])['char'].apply(lambda x: ' '.join(x)).reset_index(name='text')
    df_cc_grouped['text_digits'] = df_cc_grouped.apply(lambda x: ''.join([y for y in x['text'] if y.isdigit()]), axis=1)
    df_cc_grouped['text_digits_len'] = df_cc_grouped.apply(lambda x: len([y for y in x['text'] if y.isdigit()]), axis=1)
    df_cc_grouped = df_cc_grouped[(df_cc_grouped['cluster']!=-1) & 
                                (df_cc_grouped['text_digits_len']>=5)].reset_index(drop=True)
    df_cc_grouped['room'] = df_cc_grouped.apply(
        
        lambda x: x['text'][:[x.isdigit() for x in x['text']].index(True)].strip()
                                                        
                                                        , axis=1)
    df_cc_grouped['length'] = df_cc_grouped.apply(lambda x: dimension_splitter(x['text_digits'])[0]
                                                        
                                                        , axis=1)
    df_cc_grouped['width'] = df_cc_grouped.apply(lambda x: dimension_splitter(x['text_digits'])[1]
                                                                                                            
                                                        , axis=1)
    df_cc_grouped['area_sqm'] = np.round(df_cc_grouped['length'] * df_cc_grouped['width'], 2)
    
    return df_cc_grouped.drop(['text_digits', 'text_digits_len'], axis=1)