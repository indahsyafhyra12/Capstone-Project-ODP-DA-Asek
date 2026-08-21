from turtle import title, width
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

#<div style ="text-align: justify">
st.set_page_config(page_title="Technician Rating Dashboard", page_icon=":bar_chart:", layout="wide")
st.header('Latar Belakang')
st.markdown('Kepuasan pelanggan dapat diukur salah satunya dari aspek kepuasan pelayanan terhadap pelanggan. Jika pelayanan yang diterima oleh pelanggan tidak/memenuhi harapan mereka, maka pelanggan akan memberikan feedback sesuai apa yang mereka peroleh. Data rating teknisi diperlukan untuk mengevaluasi teknisi apakah mereka telah menjalankan tugas dengan baik atau tidak.')
st.markdown('''
Alur untuk mengisi survey Data Teknisi:
*Isi FAQ.
*Melaporkan gangguan
*Menjadwalkan kapan untuk perbaikan perangkat
*Repair Request (Riwayat Tracking Teknisi berangkat hingga problem solved)
*Setelah problem solved, pelanggan mengisi survey Data Teknisi (rating, alasan memberikan rating tersebut, regional, witel, id teknisi, data diri pengguna, dan lain-lain)
''')

#uploader file
file = st.file_uploader("Upload file", type=['csv','xlsx'])
st.write(file)

# ---- READ EXCEL ----
data = pd.read_excel('technician rating.xlsx')
data.dropna(inplace=True)
st.title('Data Rating Teknisi')
AgGrid(data)

# ---- SIDEBAR ----
st.sidebar.header("Please Filter Here:")
date = data["responses.date"].unique()
selected_date = st.sidebar.multiselect('Date', date, date[:1])

witel = data["responses.witel"].unique()
selected_witel = st.sidebar.multiselect('Witel', witel, witel[:1])

# use selected values from widgets to filter dataset down to only the rows we need
df_selected = data['responses.witel'].isin(selected_witel)

st.header('Dataset yang telah dipilih')
st.write('Data Dimension: ' + str(data[df_selected].shape[0]) + ' rows and ' + str(data[df_selected].shape[1])+ ' columns.')

# ---- MAINPAGE ----
st.title(":bar_chart: Technician Rating Dashboard")
st.markdown("##")
# simple description
st.write('In this dashboard we will analyze the Technician Rating data from NPS. '
           'These charts are interactive')

# TOP KPI's
total_review = int(data["responses.rate"].count())
average_rating = round(data["responses.rate"].mean(), 1)
star_rating = ":star:" * int(round(average_rating, 0))

left_column, right_column = st.columns(2)
with left_column:
    st.subheader("Total Review:")
    st.subheader(f" {total_review:,}")
with right_column:
    st.subheader("Average Rating:")
    st.subheader(f"{average_rating} {star_rating}")

st.markdown("""---""")

#Bargraph Witel
source = pd.DataFrame({
    'witel' : data['responses.witel'].unique(),
    'count' : data['responses.witel'].value_counts()
})

c = alt.Chart(source).mark_bar(color="blue").encode(
    x=alt.X('witel',sort='-y'),
    y='count'
).properties(width=800,height=500)

st.altair_chart(c, use_container_width=False)

#Bargraph Region
source = pd.DataFrame({
    'region' : data['responses.region'].unique(),
    'count' : data['responses.region'].value_counts()
})

c = alt.Chart(source).mark_bar(color="blue").encode(
    x=alt.X('region',sort='-y'),
    y='count'
).properties(width=800,height=500)

st.altair_chart(c, use_container_width=False)