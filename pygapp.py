import pygwalker as pyg
import pandas as pd
import streamlit.components.v1 as components
import streamlit as st
import random
import numpy as np
import datetime

# 今回のデータ生成に使う関数
#
# ランダムなデータをnum_rows行を生成する
p = 3
path = "experiment_data/NIPS/param/persona={}/persona_ration.npy".format(p)
data = np.load(path)
arr = data[0]
num_rows = p
num_column = 32
df = pd.DataFrame(arr,index=range(num_column), columns=range(num_rows))
df.columns = df.columns.map(str)
#print(data)
# Streamlitページの幅を調整する
st.set_page_config(layout="wide")

# Pygwalkerを使用してHTMLを生成する
pyg_html = pyg.walk(df, env='Streamlit', return_html=True, dark='light').to_html()
 
# HTMLをStreamlitアプリケーションに埋め込む
components.html(pyg_html, width=1300, height=1000, scrolling=True)
