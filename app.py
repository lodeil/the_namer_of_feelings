import os
import json
import io

import numpy as np
import streamlit as st
from transformers import pipeline

# Config :
# To launch : streamlit run app.py

classifier = pipeline("zero-shot-classification",model="facebook/bart-large-mnli")

st.set_page_config(
    page_title="The Namer of feelings",
    page_icon="🔵",
)

st.title("🔵 The Namer of feelings ")
st.header("")


# with st.expander("ℹ️ - Information on this app", expanded=True):

#     st.write(
#         """     
# -   The *The Namer of feelings* is an app that allow you to extract the feelings in a text       
# -   It build to be easy to use with nominal performance 
# -   ⚠️ the model wasn't audited , it can and probably will exhibit harmfull behavior for some data cases
# 	    """
#     )

#     st.markdown("")

# st.title("🪐 Write your text ")
# st.header("")

# with st.form(key="text_form"):

#     c29, c30, c31 = st.columns([0.08, 6, 0.18])

#     with c30:

#         text = st.text_area(
#             'Write or past your text', 
#             "And suddenly the memory came to me. This taste was that of the little piece of madeleine that on Sunday mornings at Combray (because that day I didn't go out before mass time), when I went to say hello to her in her room, my Aunt Léonie offered it to me after having dipped it in her infusion of tea or lime blossom. The sight of the little madeleine had reminded me of nothing before I had tasted it; perhaps because, having often seen them since, without eating them, on the shelves of pastry chefs, their image had left those days of Combray to be linked to other more recent ones; perhaps because of those memories so long abandoned [...] - Proust"
#         ,
#         height=300
#         )

#         submit_button = st.form_submit_button(label="💡 Get me the feelings !")


# if not submit_button:
#     st.stop()

# st.title("🌌 Feelings :")
# st.header("")

# with st.spinner('The model is thinking very hard ...'):
    
#     sequence_to_classify = text

#     candidate_labels = [
#         '👏 Admiration',
#         '🥰 Adoration',
#         '🦢 Aesthetic Appreciation',
#         '🎢 Amusement',
#         '😡 Anger',
#         '😟 Anxiety',
#         '😬 Awkwardness',
#         '🥱 Boredom',
#         '🌬️ Calmness',
#         '😵 Confusion',
#         '🤤 Craving',
#         '🤢 Disgust',
#         '😧 Empathetic pain',
#         '✨ Entrancement',
#         '🤩 Excitement',
#         '😨 Fear',
#         '😱 Horror',
#         '🛎️ Interest',
#         '😃 Joy',
#         '📼 Nostalgia',
#         '😌 Relief',
#         '🌹 Romance',
#         '😥 Sadness',
#         '😊 Satisfaction',
#         '😲 Surprise',
#     ]
    
#     answer = classifier(sequence_to_classify, candidate_labels, multi_class=True)

#     label_probability_dict = {}

#     for i in range(len(answer['labels'])):
#         label_probability_dict[answer['labels'][i]] = answer['scores'][i]

#     label_score_to_show = {k: v for k, v in label_probability_dict.items() if v > 0.3 }
#     # label_score_to_show = {k: str(np.floor(10000*float(v))/100.0 ) + '%' for k, v in label_probability_dict.items() if v > 0.3 }

# label_score_to_show


# st.markdown("P.S: The text wanted to tell you 'thank you for understanding me 😊'")