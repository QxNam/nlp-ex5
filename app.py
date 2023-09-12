import pickle
import underthesea
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import metrics

import streamlit as st

st.set_page_config(
    page_title="Text Classification App",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': '''Quách Xuân Nam - 20020541 - IUH\n
        https://www.facebook.com/20020541.nam'''
    }
)

class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model = None):
        self.model = model
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def f1_score(self, y_true, y_pred):
        return metrics.f1_score(y_true, y_pred, average='micro')
    
    def accuracy_score(self, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)
    
    def confusion_matrix(self, y_true, y_pred):
        return metrics.confusion_matrix(y_true, y_pred)
    
list_class = {'giao-duc': 'giáo dục', 
              'phap-luat': 'pháp luật', 
              'oto-xe-may': 'xe', 
              'suc-khoe': 'sức khỏe', 
              'khoa-hoc': 'khoa học'
            }
model = None
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
vectorizer = None
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def predict(data):
    if data == '':
        return '?'
    text = underthesea.word_tokenize(data, format="text")
    text = vectorizer.transform([text])
    res = model.predict(text.toarray())[0]
    return list_class[list(list_class.keys())[res]]
    
def input_features():
    text = ''
    st.markdown('<h2 style="text-align: center;">Input text</h1>', unsafe_allow_html=True)
    text = st.text_area(text, height=200)
    st.markdown('---')
    st.markdown('<h2 style="text-align: center;">Upload file</h1>', unsafe_allow_html=True)
    file_loaded = st.file_uploader("", type="txt")

    if file_loaded is not None:
        text = file_loaded.read().decode("utf-8")
    return text

def main():
    st.markdown('<h1 style="text-align: center;">📄 TEXT CLASSIFICATION</h1>', unsafe_allow_html=True)
    st.markdown('---')
    data = ''
    col1, col2 = st.columns(2)
    with col1:
        data = input_features()
    pred = predict(data)
    with col2:
        st.markdown('<h2 style="text-align: center;">Prediction</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="text-align: center; border: 2px solid green; margin-top: 30px; ; border-radius: 10px; height: 100px">{pred.upper()}</h1>', unsafe_allow_html=True)
        
        st.markdown('<h3 style="text-align: left;">Topic prediction of Vietnamese text</h3>', unsafe_allow_html=True)
        st.markdown('<h4 style="text-align: left;">🧑‍⚖️ Pháp luật</h4>', unsafe_allow_html=True)
        st.markdown('<h4 style="text-align: left;">🧑‍⚕️ Sức khỏe</h4>', unsafe_allow_html=True)
        st.markdown('<h4 style="text-align: left;">🧑‍🚀 Khoa học</h4>', unsafe_allow_html=True)
        st.markdown('<h4 style="text-align: left;">🧑‍🦽 Xe</h1>', unsafe_allow_html=True)
        st.markdown('<h4 style="text-align: left;">🧑‍🏫 Giáo dục</h4>', unsafe_allow_html=True)
    st.caption('Modify by :blue[qxnam]', unsafe_allow_html=True)
    
if __name__ == '__main__':
    main()