from transformers import MarianMTModel, MarianTokenizer
import gradio as gr

model_name = 'Helsinki-NLP/opus-mt-en-fr'

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text):
    tokenized_text = tokenizer(text, return_tensors='pt', padding=True)
    
    translation = model.generate(**tokenized_text)
    
    translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
    
    return translated_text

iface = gr.Interface(
    fn=translate, 
    inputs=gr.Textbox(label="Enter text to translate"), 
    outputs=gr.Textbox(label="Translated text"),
    title="English to French Translator",
    description="Enter English text and get the French translation."
)

iface.launch()
