import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
from PIL import Image, ImageTk
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random

# Load the model and necessary data
model = load_model('model_new.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        result = "Response not found"
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

def send_message(event=None):
    user_input = input_box.get("1.0",'end-1c').strip()
    if user_input:
        chat_history.config(state=tk.NORMAL)
        chat_history.insert(tk.END, "You: " + user_input + '\n\n')
        chat_history.config(state=tk.DISABLED)
        chat_history.see(tk.END)
        input_box.delete("1.0", tk.END)

        response = chatbot_response(user_input)
        chat_history.config(state=tk.NORMAL)
        chat_history.insert(tk.END, "Bot: " + response + '\n\n')
        chat_history.config(state=tk.DISABLED)
        chat_history.see(tk.END)

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()

root = tk.Tk()
root.title("Chatbot")

root.geometry("600x450")
root.resizable(width=False, height=False)

style = ttk.Style()
style.theme_use('clam')
style.configure("TButton", font=("Arial", 10), padding=5)
style.configure("TFrame", background="#f0f0f0")
style.configure("TLabel", font=("Arial", 12), background="#f0f0f0")
style.configure("TScrollbar", gripcount=0, background="#f0f0f0", troughcolor="#cccccc")

main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

chat_history = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=50, height=15)
chat_history.config(state=tk.DISABLED)
chat_history.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

input_box = tk.Text(main_frame, height=3, width=50)
input_box.grid(row=1, column=0, padx=10, pady=(0,10), sticky="ew")
input_box.bind("<Return>", send_message)



send_button = ttk.Button(main_frame, text="Send", command=send_message)
send_button.grid(row=1, column=1, padx=10, pady=(0,10), sticky="ew")


root.bind('<Return>', send_message)
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
