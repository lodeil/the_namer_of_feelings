import os
import json
import io

import numpy as np
import streamlit as st
from transformers import pipeline

# Config :
# To launch : streamlit run app.py


# fail : 
# classifier = pipeline("zero-shot-classification",model="shash2409/bert-finetuned-squad")
# classifier = pipeline("zero-shot-classification",model="t5-small")
# big good 
# classifier = pipeline("zero-shot-classification",model="Narsil/deberta-large-mnli-zero-cls")
# classifier = pipeline("zero-shot-classification",model="facebook/bart-large-mnli")
# bug
# classifier = pipeline("zero-shot-classification",model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
# classifier = pipeline("zero-shot-classification",model="cross-encoder/nli-deberta-v3-xsmall")

classifier = pipeline("zero-shot-classification",model="cross-encoder/nli-distilroberta-base")
### classifier = pipeline("zero-shot-classification",model="tals/albert-xlarge-vitaminc-mnli")
# classifier = pipeline("zero-shot-classification",model="finiteautomata/beto-sentiment-analysis")
# classifier = pipeline("zero-shot-classification",model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")


# classifier = pipeline("zero-shot-classification",model="valhalla/distilbart-mnli-12-1")
# classifier = pipeline("zero-shot-classification",model="Narsil/deberta-large-mnli-zero-cls")

# xlm-roberta-base
# The `multi_class` argument has been deprecated and renamed to `multi_label`. `multi_class` will be removed in a future 
# version of Transformers.


st.set_page_config(
    page_title="The Name of the feelings",
    page_icon="🔵",
)

st.title("🔵 The Name of the feelings")
st.header("")


with st.expander("ℹ️ - Information on this app", expanded=True):

    st.write(
        """     
-   The *The Name of the feelings* is an app that allow you to extract the feelings in a text       
-   It build to be easy to use with nominal performance 
-   ⚠️ the model wasn't audited , it can and probably will exhibit harmfull behavior for some data cases
	    """
    )

    st.markdown("")

st.title("🪐 Write your text ")
st.header("")

with st.form(key="text_form"):

    c29, c30, c31 = st.columns([0.08, 6, 0.18])

    with c30:

        text = st.text_area(
            'Write or past your text', 
            "And suddenly the memory came to me. This taste was that of the little piece of madeleine that on Sunday mornings at Combray (because that day I didn't go out before mass time), when I went to say hello to her in her room, my Aunt Léonie offered it to me after having dipped it in her infusion of tea or lime blossom. The sight of the little madeleine had reminded me of nothing before I had tasted it; perhaps because, having often seen them since, without eating them, on the shelves of pastry chefs, their image had left those days of Combray to be linked to other more recent ones; perhaps because of those memories so long abandoned [...] - Proust"
        ,
        height=300
        )

        submit_button = st.form_submit_button(label="💡 Get me the feelings !")


if not submit_button:
    st.stop()

st.title("🌌 Feelings :")
st.header("")

with st.spinner('The model is thinking very hard ...'):
    
    sequence_to_classify = text

    candidate_labels = [
        '👏 Admiration',
        '🥰 Adoration',
        '🦢 Aesthetic Appreciation',
        '🎢 Amusement',
        '😡 Anger',
        '😟 Anxiety',
        '😬 Awkwardness',
        '🥱 Boredom',
        '🌬️ Calmness',
        '😵 Confusion',
        '🤤 Craving',
        '🤢 Disgust',
        '😧 Empathetic pain',
        '✨ Entrancement',
        '🤩 Excitement',
        '😨 Fear',
        '😱 Horror',
        '🛎️ Interest',
        '😃 Joy',
        '📼 Nostalgia',
        '😌 Relief',
        '🌹 Romance',
        '😥 Sadness',
        '😊 Satisfaction',
        '😲 Surprise',
    ]
    
    answer = classifier(sequence_to_classify, candidate_labels, multi_class=True)

    label_probability_dict = {}

    for i in range(len(answer['labels'])):
        label_probability_dict[answer['labels'][i]] = answer['scores'][i]

    # label_score_to_show = {k: v for k, v in label_probability_dict.items() if v > 0.3 }
    label_score_to_show = {k: str(np.floor(10000*float(v))/100.0 ) + '%' for k, v in label_probability_dict.items() if v > 0.05 }

label_score_to_show


st.markdown("P.S: The text wanted to tell you 'thank you for understanding me 😊'")