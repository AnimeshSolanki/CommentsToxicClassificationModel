import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.layers import TextVectorization
import gradio as gr

with open('vectorizer.pkl', 'rb') as f:
    vectorizer_data = pickle.load(f)

vectorizer = TextVectorization.from_config(vectorizer_data['config'])
vectorizer.set_weights(vectorizer_data['weights'])

model = tf.keras.models.load_model('toxicity.h5')

def predict_toxicity(comment):
    # Preprocess the input comment
    vectorized_comment = vectorizer(np.array([comment]))
    # Predict toxicity
    prediction = model.predict(vectorized_comment)
    return {label: float(prediction[0][i]) for i, label in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])}

# Create Gradio interface
interface = gr.Interface(fn=predict_toxicity, inputs="textbox", outputs="label", title="Toxic Comment Detector")

interface.launch()
