import streamlit as st
import pandas as pd
from personal_ai import *

st.set_page_config(layout="wide")

personal_ai = personalAI("roscacor.mp4")
personal_ai.run()

placeholder = st.empty()
while True:
    frame, result, ts = personal_ai.image_q.get()

    if ts == "done":
        break

    with placeholder.container():
        st.image(frame)
