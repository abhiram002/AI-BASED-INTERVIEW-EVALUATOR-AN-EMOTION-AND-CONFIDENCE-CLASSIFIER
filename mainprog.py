from flask import Flask, render_template, request, session, flash, redirect, url_for, send_file

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import cv2
from keras.models import load_model
import numpy as np
import time
import tkinter as tk
import threading
import sounddevice as sd
import soundfile as sf
import os
import speech_recognition as sr
import csv
from difflib import SequenceMatcher
from PIL import Image, ImageTk
from subprocess import Popen
import nltk
nltk.download('punkt')
import nltk
nltk.download('wordnet')
from deepface import DeepFace
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pymongo import MongoClient
import uuid
import datetime
from flask_session import Session
import re
import random


app = Flask(__name__)
app.config["SESSION_PERMANENT"]=False
app.config["SESSION_TYPE"]="filesystem"
Session(app)






class QuizApp:
    def __init__(self, root, session_data):
        self.key=True
        self.root = root
        self.max_emotion_label = None
        self.max_emotion_count = None
        self.y=None
        self.similarity=[]
        self.time_axis=None
        self.root.title("Real time Emotion and voice frequency Detection")
        width=root.winfo_screenwidth()
        height=root.winfo_screenheight()
        self.root.geometry("%dx%d" % (width, height))
        
        self.session_data=session_data
        
        # Load the pre-trained face detection classifier
        # self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Load the pre-trained emotion recognition model
        # self.emotion_model = (r"C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT\\my_model.zip")
        # self.emotion_labels = ['Angry','Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.emotion_counts = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
        
        self.start_button = tk.Button(root, text="Start", command=self.start_quiz, bg="#aaffaa")
        self.start_button.pack(pady=20)
        self.question_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.question_label.pack()
        self.countdown_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.countdown_label.pack()
        self.speak_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.speak_label.pack()
        self.canvas = tk.Canvas(root, width=600, height=400)
        self.canvas.pack()
        self.cap = None
        self.emotion_thread = None
        self.t3=None
        self.t4=None
       
        self.recording=True
        self.audio_data= np.array([])
        self.clf=None
        self.emotion_start_times = {label: None for label in list(self.emotion_counts.keys())}

    def start_quiz(self):
        self.start_button.config(state=tk.DISABLED)
        self.question_label.config(text='PLEASE WAIT UNTIL CAMERA TURN ON..')
        
        # Start the emotion detection thread
        self.emotion_thread = threading.Thread(target=self.detect_emotions)
        self.emotion_thread.start()
        self.start_countdown_beforecam(21)
    # Schedule the function to continue after 25 seconds
        self.root.after(21000, self.continue_quiz)

    def continue_quiz(self):
    # Clear similarity list, set questions, and start asking questions
        self.similarity.clear()
        self.questions = ["Define Python ?", "What is pandas in python ?", "What is numpy in python ?"]
        self.current_question_index = 0
        self.ask_question_after_delay()


    def start_countdown_beforecam(self, seconds):
        self.countdown_label.config(text=f"Time remaining: {seconds} seconds")
        if seconds > 0:
            self.root.after(1000, self.start_countdown_beforecam, seconds - 1)
       

    def ask_question_after_delay(self):
        if self.current_question_index < len(self.questions):
            question = self.questions[self.current_question_index]
            self.question_label.config(text=question)
            self.start_countdown(5)
            self.current_question_index += 1
        else:
            self.key=False
            self.show_completion_message()

    def start_countdown(self, seconds):
        self.countdown_label.config(text=f"Time remaining: {seconds} seconds")
        if seconds > 0:
            self.root.after(1000, self.start_countdown, seconds - 1)
        else:
            self.countdown_label.config(text="")
            self.record_audio()

    def record_audio(self):
        duration = 8
        self.speak_label.config(text="Speak now")
        threading.Thread(target=self._record_audio, args=(duration,)).start()
        # self.t3 = threading.Thread(target=self.mfcc).start()
        # self.key=True


    def _record_audio(self, duration):
        fs = 44100
        self.audio_data = sd.rec(int(fs * duration), samplerate=fs, channels=2, dtype='int16')
        sd.wait()
        save_directory = r"C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT\\static\\audiofolder"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        filename_wav = os.path.join(save_directory, f"question_{self.current_question_index}.wav")
        sf.write(filename_wav, self.audio_data, fs)
        text = self.convert_audio_to_text(filename_wav)
        filename_text = os.path.join(save_directory, f"question_{self.current_question_index}.txt")
        with open(filename_text, 'w') as text_file:
            text_file.write(text)
        self.speak_label.config(text="")
        
        self.ask_question_after_delay()

    def convert_audio_to_text(self, audio_file):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            self.audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(self.audio_data)
            filename_answer = os.path.join(r'C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT\\Answer\\Pythonanswer',
                                           f"Ans{self.current_question_index}.txt")
            with open(filename_answer, 'r') as answer_file:
                answer_text = answer_file.read()
                similarity = SequenceMatcher(None, text.lower(), answer_text.lower()).ratio()
            
            result_text = f"Question {self.current_question_index} - Similarity with Answer: {similarity * 100:.2f}%"
            
            self.similarity.append(result_text)
            self.save_result_to_csv(result_text)
        except sr.UnknownValueError:
            result_text = f"Question {self.current_question_index} - Couldn't Recognize the Answer"
            self.similarity.append(result_text)
            self.save_result_to_csv(result_text)
            return result_text
        except sr.RequestError as e:
            return f"Could not request results from Speech Recognition service; {e}"
        return text

    def save_result_to_csv(self, result_text):
        csv_filename = os.path.join(r"C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT", "results.csv")
        if not os.path.isfile(csv_filename):
            with open(csv_filename, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["Question Number", "Similarity Result"])
        with open(csv_filename, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([self.current_question_index, result_text])

    def clear_question(self):
        self.question_label.config(text="")
        self.speak_label.config(text="")
        self.ask_question_after_delay()

    def show_completion_message(self):
        self.question_label.config(text="Congratulations! You have completed the quiz.")
        self.start_button.config(state=tk.NORMAL)
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            
        if self.emotion_thread is not None and self.emotion_thread.is_alive():
            self.emotion_thread.join()
            threading.join()
            
        max_emotion_label = max(self.emotion_counts, key=self.emotion_counts.get)
        max_emotion_count = self.emotion_counts[max_emotion_label]
        self.max_emotion_label = max_emotion_label
        self.max_emotion_count = max_emotion_count
        

        try:
            result_document = {
                    'uuid':self.session_data['uuid'],
                    'email': self.session_data['email'],
                    'max_emotion_label': self.max_emotion_label,
                    'max_emotion_count': self.max_emotion_count,
                    'similarity of Q1' : self.similarity[0],
                    'similarity of Q2' : self.similarity[1],
                    'similarity of Q3' : self.similarity[2],
                    'quiz_time' : time.strftime('%H:%M:%S', time.gmtime()),
                    'timestamp' : datetime.datetime.now().isoformat(),
                    'confidence level from Q1' : " ",
                    'confidence level from Q2' : " ",
                    'confidence level from Q3' : " "
                    
                }
            print('setttt')
            # Insert the document into the "result" collection
            db['resultdb'].insert_one(result_document)
            
            print("Emotions stored successfully!")
        except Exception as e:
            print(e)








    def detect_emotions(self):
        # Dictionary to store the emotion counters

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
        self.cap = cv2.VideoCapture(0)

# Variables to store emotion counts
        

        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Convert grayscale frame to RGB format
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Extract the face ROI (Region of Interest)
                face_roi = rgb_frame[y:y + h, x:x + w]

                # Perform emotion analysis on the face ROI
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                # Determine the dominant emotion
                emotion = result[0]['dominant_emotion']
                
                # Increment emotion count
                self.emotion_counts[emotion] += 1

                # Draw rectangle around face and label with predicted emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
    # Display the resulting frame
            # self.cap.imshow(frame)
            self.display_frame(frame)


    def display_frame(self, frame):
        # Convert the frame to RGB format for displaying in tkinter
          
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to ImageTk format
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the canvas from the main thread
        self.root.after(0, self.update_canvas, img_tk)

    # Update the canvas with the new frame    
    def update_canvas(self, img_tk):
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk  # Keep a reference to avoid garbage collection


#-------------------------------ML--------------------------------------------------------


class Mlapp:
    def __init__(self, root, session_data):
        self.key=True
        self.root = root
        self.max_emotion_label = None
        self.max_emotion_count = None
        self.y=None
        self.similarity=[]
        self.time_axis=None
        self.root.title("Real time Emotion and voice frequency Detection")
        width=root.winfo_screenwidth()
        height=root.winfo_screenheight()
        self.root.geometry("%dx%d" % (width, height))
        
        self.session_data=session_data
        
        # Load the pre-trained face detection classifier
        # self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Load the pre-trained emotion recognition model
        # self.emotion_model = (r"C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT\\my_model.zip")
        # self.emotion_labels = ['Angry','Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.emotion_counts = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
        
        self.start_button = tk.Button(root, text="Start", command=self.start_quiz, bg="#aaffaa")
        self.start_button.pack(pady=20)
        self.question_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.question_label.pack()
        self.countdown_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.countdown_label.pack()
        self.speak_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.speak_label.pack()
        self.canvas = tk.Canvas(root, width=600, height=400)
        self.canvas.pack()
        self.cap = None
        self.emotion_thread = None
        self.t3=None
        self.t4=None
       
        self.recording=True
        self.audio_data= np.array([])
        self.clf=None
        self.emotion_start_times = {label: None for label in list(self.emotion_counts.keys())}

    def start_quiz(self):
        self.start_button.config(state=tk.DISABLED)
        self.question_label.config(text='PLEASE WAIT UNTIL CAMERA TURN ON..')
        
        # Start the emotion detection thread
        self.emotion_thread = threading.Thread(target=self.detect_emotions)
        self.emotion_thread.start()
        self.start_countdown_beforecam(21)
    # Schedule the function to continue after 25 seconds
        self.root.after(21000, self.continue_quiz)

    def continue_quiz(self):
    # Clear similarity list, set questions, and start asking questions
        self.similarity.clear()
        self.questions = ["Define ML ?", "What is Neural network ?", "What is deeplearning ?"]
        self.current_question_index = 0
        self.ask_question_after_delay()


    def start_countdown_beforecam(self, seconds):
        self.countdown_label.config(text=f"Time remaining: {seconds} seconds")
        if seconds > 0:
            self.root.after(1000, self.start_countdown_beforecam, seconds - 1)
       

    def ask_question_after_delay(self):
        if self.current_question_index < len(self.questions):
            question = self.questions[self.current_question_index]
            self.question_label.config(text=question)
            self.start_countdown(5)
            self.current_question_index += 1
        else:
            self.key=False
            self.show_completion_message()

    def start_countdown(self, seconds):
        self.countdown_label.config(text=f"Time remaining: {seconds} seconds")
        if seconds > 0:
            self.root.after(1000, self.start_countdown, seconds - 1)
        else:
            self.countdown_label.config(text="")
            self.record_audio()

    def record_audio(self):
        duration = 8
        self.speak_label.config(text="Speak now")
        threading.Thread(target=self._record_audio, args=(duration,)).start()
        # self.t3 = threading.Thread(target=self.mfcc).start()
        # self.key=True


    def _record_audio(self, duration):
        fs = 44100
        self.audio_data = sd.rec(int(fs * duration), samplerate=fs, channels=2, dtype='int16')
        sd.wait()
        save_directory = r"C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT\\static\\audiofolder"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        filename_wav = os.path.join(save_directory, f"question_{self.current_question_index}.wav")
        sf.write(filename_wav, self.audio_data, fs)
        text = self.convert_audio_to_text(filename_wav)
        filename_text = os.path.join(save_directory, f"question_{self.current_question_index}.txt")
        with open(filename_text, 'w') as text_file:
            text_file.write(text)
        self.speak_label.config(text="")
        
        self.ask_question_after_delay()

    def convert_audio_to_text(self, audio_file):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            self.audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(self.audio_data)
            filename_answer = os.path.join(r'C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT\\Answer\\Mlanswer',
                                           f"Ans{self.current_question_index}.txt")
            with open(filename_answer, 'r') as answer_file:
                answer_text = answer_file.read()
                similarity = SequenceMatcher(None, text.lower(), answer_text.lower()).ratio()
            
            result_text = f"Question {self.current_question_index} - Similarity with Answer: {similarity * 100:.2f}%"
            
            self.similarity.append(result_text)
            self.save_result_to_csv(result_text)
        except sr.UnknownValueError:
            result_text = f"Question {self.current_question_index} - Couldn't Recognize the Answer"
            self.similarity.append(result_text)
            self.save_result_to_csv(result_text)
            return result_text
        except sr.RequestError as e:
            return f"Could not request results from Speech Recognition service; {e}"
        return text

    def save_result_to_csv(self, result_text):
        csv_filename = os.path.join(r"C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT", "results.csv")
        if not os.path.isfile(csv_filename):
            with open(csv_filename, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["Question Number", "Similarity Result"])
        with open(csv_filename, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([self.current_question_index, result_text])

    def clear_question(self):
        self.question_label.config(text="")
        self.speak_label.config(text="")
        self.ask_question_after_delay()

    def show_completion_message(self):
        self.question_label.config(text="Congratulations! You have completed the quiz.")
        self.start_button.config(state=tk.NORMAL)
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            
        if self.emotion_thread is not None and self.emotion_thread.is_alive():
            self.emotion_thread.join()
            threading.join()
            
        max_emotion_label = max(self.emotion_counts, key=self.emotion_counts.get)
        max_emotion_count = self.emotion_counts[max_emotion_label]
        self.max_emotion_label = max_emotion_label
        self.max_emotion_count = max_emotion_count
        

        try:
            result_document = {
                    'uuid':self.session_data['uuid'],
                    'email': self.session_data['email'],

                    'max_emotion_label': self.max_emotion_label,
                    'max_emotion_count': self.max_emotion_count,
                    'similarity of Q1' : self.similarity[0],
                    'similarity of Q2' : self.similarity[1],
                    'similarity of Q3' : self.similarity[2],
                    'quiz_time' : time.strftime('%H:%M:%S', time.gmtime()),
                    'timestamp' : datetime.datetime.now().isoformat(),
                    'confidence level from Q1' : " ",
                    'confidence level from Q2' : " ",
                    'confidence level from Q3' : " "
                    
                }
            print('setttt')
            # Insert the document into the "result" collection
            db['resultdb'].insert_one(result_document)
            
            print("Emotions stored successfully!")
        except Exception as e:
            print(e)








    def detect_emotions(self):
        # Dictionary to store the emotion counters

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
        self.cap = cv2.VideoCapture(0)

# Variables to store emotion counts
        

        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Convert grayscale frame to RGB format
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Extract the face ROI (Region of Interest)
                face_roi = rgb_frame[y:y + h, x:x + w]

                # Perform emotion analysis on the face ROI
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                # Determine the dominant emotion
                emotion = result[0]['dominant_emotion']
                
                # Increment emotion count
                self.emotion_counts[emotion] += 1

                # Draw rectangle around face and label with predicted emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
    # Display the resulting frame
            # self.cap.imshow(frame)
            self.display_frame(frame)


    def display_frame(self, frame):
        # Convert the frame to RGB format for displaying in tkinter
          
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to ImageTk format
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the canvas from the main thread
        self.root.after(0, self.update_canvas, img_tk)

    # Update the canvas with the new frame    
    def update_canvas(self, img_tk):
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk  # Keep a reference to avoid garbage collection




#-------------------------------MLEND-----------------------------------------------------


#-------------------------------AI-----------------------------------------------------------

class Aiapp:
    def __init__(self, root, session_data):
        self.key=True
        self.root = root
        self.max_emotion_label = None
        self.max_emotion_count = None
        self.y=None
        self.similarity=[]
        self.time_axis=None
        self.root.title("Real time Emotion and voice frequency Detection")
        width=root.winfo_screenwidth()
        height=root.winfo_screenheight()
        self.root.geometry("%dx%d" % (width, height))
        
        self.session_data=session_data
        
        # Load the pre-trained face detection classifier
        # self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Load the pre-trained emotion recognition model
        # self.emotion_model = (r"C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT\\my_model.zip")
        # self.emotion_labels = ['Angry','Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.emotion_counts = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
        
        self.start_button = tk.Button(root, text="Start", command=self.start_quiz, bg="#aaffaa")
        self.start_button.pack(pady=20)
        self.question_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.question_label.pack()
        self.countdown_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.countdown_label.pack()
        self.speak_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.speak_label.pack()
        self.canvas = tk.Canvas(root, width=600, height=400)
        self.canvas.pack()
        self.cap = None
        self.emotion_thread = None
        self.t3=None
        self.t4=None
       
        self.recording=True
        self.audio_data= np.array([])
        self.clf=None
        self.emotion_start_times = {label: None for label in list(self.emotion_counts.keys())}

    def start_quiz(self):
        self.start_button.config(state=tk.DISABLED)
        self.question_label.config(text='PLEASE WAIT UNTIL CAMERA TURN ON..')
        
        # Start the emotion detection thread
        self.emotion_thread = threading.Thread(target=self.detect_emotions)
        self.emotion_thread.start()
        self.start_countdown_beforecam(21)
    # Schedule the function to continue after 25 seconds
        self.root.after(21000, self.continue_quiz)

    def continue_quiz(self):
    # Clear similarity list, set questions, and start asking questions
        self.similarity.clear()
        self.questions = ["What is artificial intelligence ?", "How is machine learning related to artificial intelligence ?", "Explain the concept of reinforcement learning in AI.?"]
        self.current_question_index = 0
        self.ask_question_after_delay()


    def start_countdown_beforecam(self, seconds):
        self.countdown_label.config(text=f"Time remaining: {seconds} seconds")
        if seconds > 0:
            self.root.after(1000, self.start_countdown_beforecam, seconds - 1)
       

    def ask_question_after_delay(self):
        if self.current_question_index < len(self.questions):
            question = self.questions[self.current_question_index]
            self.question_label.config(text=question)
            self.start_countdown(5)
            self.current_question_index += 1
        else:
            self.key=False
            self.show_completion_message()

    def start_countdown(self, seconds):
        self.countdown_label.config(text=f"Time remaining: {seconds} seconds")
        if seconds > 0:
            self.root.after(1000, self.start_countdown, seconds - 1)
        else:
            self.countdown_label.config(text="")
            self.record_audio()

    def record_audio(self):
        duration = 8
        self.speak_label.config(text="Speak now")
        threading.Thread(target=self._record_audio, args=(duration,)).start()
        # self.t3 = threading.Thread(target=self.mfcc).start()
        # self.key=True


    def _record_audio(self, duration):
        fs = 44100
        self.audio_data = sd.rec(int(fs * duration), samplerate=fs, channels=2, dtype='int16')
        sd.wait()
        save_directory = r"C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT\\static\\audiofolder"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        filename_wav = os.path.join(save_directory, f"question_{self.current_question_index}.wav")
        sf.write(filename_wav, self.audio_data, fs)
        text = self.convert_audio_to_text(filename_wav)
        filename_text = os.path.join(save_directory, f"question_{self.current_question_index}.txt")
        with open(filename_text, 'w') as text_file:
            text_file.write(text)
        self.speak_label.config(text="")
        
        self.ask_question_after_delay()

    def convert_audio_to_text(self, audio_file):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            self.audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(self.audio_data)
            filename_answer = os.path.join(r'C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT\\Answer\\Aianswer',
                                           f"Ans{self.current_question_index}.txt")
            with open(filename_answer, 'r') as answer_file:
                answer_text = answer_file.read()
                similarity = SequenceMatcher(None, text.lower(), answer_text.lower()).ratio()
            
            result_text = f"Question {self.current_question_index} - Similarity with Answer: {similarity * 100:.2f}%"
            
            self.similarity.append(result_text)
            self.save_result_to_csv(result_text)
        except sr.UnknownValueError:
            result_text = f"Question {self.current_question_index} - Couldn't Recognize the Answer"
            self.similarity.append(result_text)
            self.save_result_to_csv(result_text)
            return result_text
        except sr.RequestError as e:
            return f"Could not request results from Speech Recognition service; {e}"
        return text

    def save_result_to_csv(self, result_text):
        csv_filename = os.path.join(r"C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT", "results.csv")
        if not os.path.isfile(csv_filename):
            with open(csv_filename, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["Question Number", "Similarity Result"])
        with open(csv_filename, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([self.current_question_index, result_text])

    def clear_question(self):
        self.question_label.config(text="")
        self.speak_label.config(text="")
        self.ask_question_after_delay()

    def show_completion_message(self):
        self.question_label.config(text="Congratulations! You have completed the quiz.")
        self.start_button.config(state=tk.NORMAL)
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            
        if self.emotion_thread is not None and self.emotion_thread.is_alive():
            self.emotion_thread.join()
            threading.join()
            
        max_emotion_label = max(self.emotion_counts, key=self.emotion_counts.get)
        max_emotion_count = self.emotion_counts[max_emotion_label]
        self.max_emotion_label = max_emotion_label
        self.max_emotion_count = max_emotion_count
        

        try:
            result_document = {
                    'uuid':self.session_data['uuid'],
                    'email': self.session_data['email'],

                    'max_emotion_label': self.max_emotion_label,
                    'max_emotion_count': self.max_emotion_count,
                    'similarity of Q1' : self.similarity[0],
                    'similarity of Q2' : self.similarity[1],
                    'similarity of Q3' : self.similarity[2],
                    'quiz_time' : time.strftime('%H:%M:%S', time.gmtime()),
                    'timestamp' : datetime.datetime.now().isoformat(),
                    'confidence level from Q1' : " ",
                    'confidence level from Q2' : " ",
                    'confidence level from Q3' : " "
                    
                }
            print('setttt')
            # Insert the document into the "result" collection
            db['resultdb'].insert_one(result_document)
            
            print("Emotions stored successfully!")
        except Exception as e:
            print(e)

    def detect_emotions(self):
        # Dictionary to store the emotion counters

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
        self.cap = cv2.VideoCapture(0)

# Variables to store emotion counts
        

        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Convert grayscale frame to RGB format
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Extract the face ROI (Region of Interest)
                face_roi = rgb_frame[y:y + h, x:x + w]

                # Perform emotion analysis on the face ROI
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                # Determine the dominant emotion
                emotion = result[0]['dominant_emotion']
                
                # Increment emotion count
                self.emotion_counts[emotion] += 1

                # Draw rectangle around face and label with predicted emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
    # Display the resulting frame
            # self.cap.imshow(frame)
            self.display_frame(frame)


    def display_frame(self, frame):
        # Convert the frame to RGB format for displaying in tkinter
          
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to ImageTk format
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the canvas from the main thread
        self.root.after(0, self.update_canvas, img_tk)

    # Update the canvas with the new frame    
    def update_canvas(self, img_tk):
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk  # Keep a reference to avoid garbage collection


#-------------------------------AI end----------------------------------------------------

#------------------------------FrontendWEbdev---------------------------------------

class Frontendapp:
    def __init__(self, root, session_data):
        self.key=True
        self.root = root
        self.max_emotion_label = None
        self.max_emotion_count = None
        self.y=None
        self.similarity=[]
        self.time_axis=None
        self.root.title("Real time Emotion and voice frequency Detection")
        width=root.winfo_screenwidth()
        height=root.winfo_screenheight()
        self.root.geometry("%dx%d" % (width, height))
        
        self.session_data=session_data
        
        # Load the pre-trained face detection classifier
        # self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Load the pre-trained emotion recognition model
        # self.emotion_model = (r"C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT\\my_model.zip")
        # self.emotion_labels = ['Angry','Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.emotion_counts = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
        
        self.start_button = tk.Button(root, text="Start", command=self.start_quiz, bg="#aaffaa")
        self.start_button.pack(pady=20)
        self.question_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.question_label.pack()
        self.countdown_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.countdown_label.pack()
        self.speak_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.speak_label.pack()
        self.canvas = tk.Canvas(root, width=600, height=400)
        self.canvas.pack()
        self.cap = None
        self.emotion_thread = None
        self.t3=None
        self.t4=None
       
        self.recording=True
        self.audio_data= np.array([])
        self.clf=None
        self.emotion_start_times = {label: None for label in list(self.emotion_counts.keys())}

    def start_quiz(self):
        self.start_button.config(state=tk.DISABLED)
        self.question_label.config(text='PLEASE WAIT UNTIL CAMERA TURN ON..')
        
        # Start the emotion detection thread
        self.emotion_thread = threading.Thread(target=self.detect_emotions)
        self.emotion_thread.start()
        self.start_countdown_beforecam(21)
    # Schedule the function to continue after 25 seconds
        self.root.after(21000, self.continue_quiz)

    def continue_quiz(self):
    # Clear similarity list, set questions, and start asking questions
        self.similarity.clear()
        self.questions = ["What is the difference between == and === in JavaScript? ?", "Explain the difference between client-side scripting and server-side scripting. ?", "Why is the @media query used in CSS?.?"]
        self.current_question_index = 0
        self.ask_question_after_delay()


    def start_countdown_beforecam(self, seconds):
        self.countdown_label.config(text=f"Time remaining: {seconds} seconds")
        if seconds > 0:
            self.root.after(1000, self.start_countdown_beforecam, seconds - 1)
       

    def ask_question_after_delay(self):
        if self.current_question_index < len(self.questions):
            question = self.questions[self.current_question_index]
            self.question_label.config(text=question)
            self.start_countdown(5)
            self.current_question_index += 1
        else:
            self.key=False
            self.show_completion_message()

    def start_countdown(self, seconds):
        self.countdown_label.config(text=f"Time remaining: {seconds} seconds")
        if seconds > 0:
            self.root.after(1000, self.start_countdown, seconds - 1)
        else:
            self.countdown_label.config(text="")
            self.record_audio()

    def record_audio(self):
        duration = 8
        self.speak_label.config(text="Speak now")
        threading.Thread(target=self._record_audio, args=(duration,)).start()
        # self.t3 = threading.Thread(target=self.mfcc).start()
        # self.key=True


    def _record_audio(self, duration):
        fs = 44100
        self.audio_data = sd.rec(int(fs * duration), samplerate=fs, channels=2, dtype='int16')
        sd.wait()
        save_directory = r"C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT\\static\\audiofolder"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        filename_wav = os.path.join(save_directory, f"question_{self.current_question_index}.wav")
        sf.write(filename_wav, self.audio_data, fs)
        text = self.convert_audio_to_text(filename_wav)
        filename_text = os.path.join(save_directory, f"question_{self.current_question_index}.txt")
        with open(filename_text, 'w') as text_file:
            text_file.write(text)
        self.speak_label.config(text="")
        
        self.ask_question_after_delay()

    def convert_audio_to_text(self, audio_file):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            self.audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(self.audio_data)
            filename_answer = os.path.join(r'C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT\\Answer\\Frontend',
                                           f"Ans{self.current_question_index}.txt")
            with open(filename_answer, 'r') as answer_file:
                answer_text = answer_file.read()
                similarity = SequenceMatcher(None, text.lower(), answer_text.lower()).ratio()
            
            result_text = f"Question {self.current_question_index} - Similarity with Answer: {similarity * 100:.2f}%"
            
            self.similarity.append(result_text)
            self.save_result_to_csv(result_text)
        except sr.UnknownValueError:
            result_text = f"Question {self.current_question_index} - Couldn't Recognize the Answer"
            self.similarity.append(result_text)
            self.save_result_to_csv(result_text)
            return result_text
        except sr.RequestError as e:
            return f"Could not request results from Speech Recognition service; {e}"
        return text

    def save_result_to_csv(self, result_text):
        csv_filename = os.path.join(r"C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT", "results.csv")
        if not os.path.isfile(csv_filename):
            with open(csv_filename, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["Question Number", "Similarity Result"])
        with open(csv_filename, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([self.current_question_index, result_text])

    def clear_question(self):
        self.question_label.config(text="")
        self.speak_label.config(text="")
        self.ask_question_after_delay()

    def show_completion_message(self):
        self.question_label.config(text="Congratulations! You have completed the quiz.")
        self.start_button.config(state=tk.NORMAL)
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            
        if self.emotion_thread is not None and self.emotion_thread.is_alive():
            self.emotion_thread.join()
            threading.join()
            
        max_emotion_label = max(self.emotion_counts, key=self.emotion_counts.get)
        max_emotion_count = self.emotion_counts[max_emotion_label]
        self.max_emotion_label = max_emotion_label
        self.max_emotion_count = max_emotion_count
        

        try:
            result_document = {
                    'uuid':self.session_data['uuid'],
                    'email': self.session_data['email'],

                    'max_emotion_label': self.max_emotion_label,
                    'max_emotion_count': self.max_emotion_count,
                    'similarity of Q1' : self.similarity[0],
                    'similarity of Q2' : self.similarity[1],
                    'similarity of Q3' : self.similarity[2],
                    'quiz_time' : time.strftime('%H:%M:%S', time.gmtime()),
                    'timestamp' : datetime.datetime.now().isoformat(),
                    'confidence level from Q1' : " ",
                    'confidence level from Q2' : " ",
                    'confidence level from Q3' : " "
                    
                }
            print('setttt')
            # Insert the document into the "result" collection
            db['resultdb'].insert_one(result_document)
            
            print("Emotions stored successfully!")
        except Exception as e:
            print(e)

    def detect_emotions(self):
        # Dictionary to store the emotion counters

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
        self.cap = cv2.VideoCapture(0)

# Variables to store emotion counts
        

        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Convert grayscale frame to RGB format
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Extract the face ROI (Region of Interest)
                face_roi = rgb_frame[y:y + h, x:x + w]

                # Perform emotion analysis on the face ROI
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                # Determine the dominant emotion
                emotion = result[0]['dominant_emotion']
                
                # Increment emotion count
                self.emotion_counts[emotion] += 1

                # Draw rectangle around face and label with predicted emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
    # Display the resulting frame
            # self.cap.imshow(frame)
            self.display_frame(frame)


    def display_frame(self, frame):
        # Convert the frame to RGB format for displaying in tkinter
          
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to ImageTk format
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the canvas from the main thread
        self.root.after(0, self.update_canvas, img_tk)

    # Update the canvas with the new frame    
    def update_canvas(self, img_tk):
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk 



#-------------------------------frontend webdev end-----------------------------------

#-------------------------------Backend developer-------------------------------------

class Backendapp:
    def __init__(self, root, session_data):
        self.key=True
        self.root = root
        self.max_emotion_label = None
        self.max_emotion_count = None
        self.y=None
        self.similarity=[]
        self.time_axis=None
        self.root.title("Real time Emotion and voice frequency Detection")
        width=root.winfo_screenwidth()
        height=root.winfo_screenheight()
        self.root.geometry("%dx%d" % (width, height))
        
        self.session_data=session_data
        
        # Load the pre-trained face detection classifier
        # self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Load the pre-trained emotion recognition model
        # self.emotion_model = (r"C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT\\my_model.zip")
        # self.emotion_labels = ['Angry','Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.emotion_counts = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
        
        self.start_button = tk.Button(root, text="Start", command=self.start_quiz, bg="#aaffaa")
        self.start_button.pack(pady=20)
        self.question_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.question_label.pack()
        self.countdown_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.countdown_label.pack()
        self.speak_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.speak_label.pack()
        self.canvas = tk.Canvas(root, width=600, height=400)
        self.canvas.pack()
        self.cap = None
        self.emotion_thread = None
        self.t3=None
        self.t4=None
       
        self.recording=True
        self.audio_data= np.array([])
        self.clf=None
        self.emotion_start_times = {label: None for label in list(self.emotion_counts.keys())}

    def start_quiz(self):
        self.start_button.config(state=tk.DISABLED)
        self.question_label.config(text='PLEASE WAIT UNTIL CAMERA TURN ON..')
        
        # Start the emotion detection thread
        self.emotion_thread = threading.Thread(target=self.detect_emotions)
        self.emotion_thread.start()
        self.start_countdown_beforecam(21)
    # Schedule the function to continue after 25 seconds
        self.root.after(21000, self.continue_quiz)

    def continue_quiz(self):
    # Clear similarity list, set questions, and start asking questions
        self.similarity.clear()
        self.questions = ["What is the difference between synchronous and asynchronous programming, and when would you use each in backend development ?", "What is the role of a reverse proxy server in backend development ?", "Explain the concept of middleware in the context of web development.?"]
        self.current_question_index = 0
        self.ask_question_after_delay()


    def start_countdown_beforecam(self, seconds):
        self.countdown_label.config(text=f"Time remaining: {seconds} seconds")
        if seconds > 0:
            self.root.after(1000, self.start_countdown_beforecam, seconds - 1)
       

    def ask_question_after_delay(self):
        if self.current_question_index < len(self.questions):
            question = self.questions[self.current_question_index]
            self.question_label.config(text=question)
            self.start_countdown(5)
            self.current_question_index += 1
        else:
            self.key=False
            self.show_completion_message()

    def start_countdown(self, seconds):
        self.countdown_label.config(text=f"Time remaining: {seconds} seconds")
        if seconds > 0:
            self.root.after(1000, self.start_countdown, seconds - 1)
        else:
            self.countdown_label.config(text="")
            self.record_audio()

    def record_audio(self):
        duration = 8
        self.speak_label.config(text="Speak now")
        threading.Thread(target=self._record_audio, args=(duration,)).start()
        # self.t3 = threading.Thread(target=self.mfcc).start()
        # self.key=True


    def _record_audio(self, duration):
        fs = 44100
        self.audio_data = sd.rec(int(fs * duration), samplerate=fs, channels=2, dtype='int16')
        sd.wait()
        save_directory = r"C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT\\static\\audiofolder"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        filename_wav = os.path.join(save_directory, f"question_{self.current_question_index}.wav")
        sf.write(filename_wav, self.audio_data, fs)
        text = self.convert_audio_to_text(filename_wav)
        filename_text = os.path.join(save_directory, f"question_{self.current_question_index}.txt")
        with open(filename_text, 'w') as text_file:
            text_file.write(text)
        self.speak_label.config(text="")
        
        self.ask_question_after_delay()

    def convert_audio_to_text(self, audio_file):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            self.audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(self.audio_data)
            filename_answer = os.path.join(r'C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT\\Answer\\Backendanswer',
                                           f"Ans{self.current_question_index}.txt")
            with open(filename_answer, 'r') as answer_file:
                answer_text = answer_file.read()
                similarity = SequenceMatcher(None, text.lower(), answer_text.lower()).ratio()
            
            result_text = f"Question {self.current_question_index} - Similarity with Answer: {similarity * 100:.2f}%"
            
            self.similarity.append(result_text)
            self.save_result_to_csv(result_text)
        except sr.UnknownValueError:
            result_text = f"Question {self.current_question_index} - Couldn't Recognize the Answer"
            self.similarity.append(result_text)
            self.save_result_to_csv(result_text)
            return result_text
        except sr.RequestError as e:
            return f"Could not request results from Speech Recognition service; {e}"
        return text

    def save_result_to_csv(self, result_text):
        csv_filename = os.path.join(r"C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT", "results.csv")
        if not os.path.isfile(csv_filename):
            with open(csv_filename, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["Question Number", "Similarity Result"])
        with open(csv_filename, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([self.current_question_index, result_text])

    def clear_question(self):
        self.question_label.config(text="")
        self.speak_label.config(text="")
        self.ask_question_after_delay()

    def show_completion_message(self):
        self.question_label.config(text="Congratulations! You have completed the quiz.")
        self.start_button.config(state=tk.NORMAL)
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            
        if self.emotion_thread is not None and self.emotion_thread.is_alive():
            self.emotion_thread.join()
            threading.join()
            
        max_emotion_label = max(self.emotion_counts, key=self.emotion_counts.get)
        max_emotion_count = self.emotion_counts[max_emotion_label]
        self.max_emotion_label = max_emotion_label
        self.max_emotion_count = max_emotion_count
        

        try:
            result_document = {
                    'uuid':self.session_data['uuid'],
                    'email': self.session_data['email'],

                    'max_emotion_label': self.max_emotion_label,
                    'max_emotion_count': self.max_emotion_count,
                    'similarity of Q1' : self.similarity[0],
                    'similarity of Q2' : self.similarity[1],
                    'similarity of Q3' : self.similarity[2],
                    'quiz_time' : time.strftime('%H:%M:%S', time.gmtime()),
                    'timestamp' : datetime.datetime.now().isoformat(),
                    'confidence level from Q1' : " ",
                    'confidence level from Q2' : " ",
                    'confidence level from Q3' : " "
                    
                }
            print('setttt')
            # Insert the document into the "result" collection
            db['resultdb'].insert_one(result_document)
            
            print("Emotions stored successfully!")
        except Exception as e:
            print(e)

    def detect_emotions(self):
        # Dictionary to store the emotion counters

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
        self.cap = cv2.VideoCapture(0)

# Variables to store emotion counts
        

        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Convert grayscale frame to RGB format
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Extract the face ROI (Region of Interest)
                face_roi = rgb_frame[y:y + h, x:x + w]

                # Perform emotion analysis on the face ROI
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                # Determine the dominant emotion
                emotion = result[0]['dominant_emotion']
                
                # Increment emotion count
                self.emotion_counts[emotion] += 1

                # Draw rectangle around face and label with predicted emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
    # Display the resulting frame
            # self.cap.imshow(frame)
            self.display_frame(frame)


    def display_frame(self, frame):
        # Convert the frame to RGB format for displaying in tkinter
          
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to ImageTk format
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the canvas from the main thread
        self.root.after(0, self.update_canvas, img_tk)

    # Update the canvas with the new frame    
    def update_canvas(self, img_tk):
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk 

#------------------------------Backend developer  end--------------------------------

#-------------------------------android app-------------------------------

class Androidapp:
    def __init__(self, root, session_data):
        self.key=True
        self.root = root
        self.max_emotion_label = None
        self.max_emotion_count = None
        self.y=None
        self.similarity=[]
        self.time_axis=None
        self.root.title("Real time Emotion and voice frequency Detection")
        width=root.winfo_screenwidth()
        height=root.winfo_screenheight()
        self.root.geometry("%dx%d" % (width, height))
        
        self.session_data=session_data
        
        # Load the pre-trained face detection classifier
        # self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Load the pre-trained emotion recognition model
        # self.emotion_model = (r"C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT\\my_model.zip")
        # self.emotion_labels = ['Angry','Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.emotion_counts = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
        
        self.start_button = tk.Button(root, text="Start", command=self.start_quiz, bg="#aaffaa")
        self.start_button.pack(pady=20)
        self.question_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.question_label.pack()
        self.countdown_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.countdown_label.pack()
        self.speak_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.speak_label.pack()
        self.canvas = tk.Canvas(root, width=600, height=400)
        self.canvas.pack()
        self.cap = None
        self.emotion_thread = None
        self.t3=None
        self.t4=None
       
        self.recording=True
        self.audio_data= np.array([])
        self.clf=None
        self.emotion_start_times = {label: None for label in list(self.emotion_counts.keys())}

    def start_quiz(self):
        self.start_button.config(state=tk.DISABLED)
        self.question_label.config(text='PLEASE WAIT UNTIL CAMERA TURN ON..')
        
        # Start the emotion detection thread
        self.emotion_thread = threading.Thread(target=self.detect_emotions)
        self.emotion_thread.start()
        self.start_countdown_beforecam(21)
    # Schedule the function to continue after 25 seconds
        self.root.after(21000, self.continue_quiz)

    def continue_quiz(self):
    # Clear similarity list, set questions, and start asking questions
        self.similarity.clear()
        self.questions = ["What is an Activity in Android development, and how does it relate to the user interface of an app ?", "Explain the purpose of the AndroidManifest.xml file and some key elements it may contain. ?", "What is the purpose of an AsyncTask in Android development.?"]
        self.current_question_index = 0
        self.ask_question_after_delay()


    def start_countdown_beforecam(self, seconds):
        self.countdown_label.config(text=f"Time remaining: {seconds} seconds")
        if seconds > 0:
            self.root.after(1000, self.start_countdown_beforecam, seconds - 1)
       

    def ask_question_after_delay(self):
        if self.current_question_index < len(self.questions):
            question = self.questions[self.current_question_index]
            self.question_label.config(text=question)
            self.start_countdown(5)
            self.current_question_index += 1
        else:
            self.key=False
            self.show_completion_message()

    def start_countdown(self, seconds):
        self.countdown_label.config(text=f"Time remaining: {seconds} seconds")
        if seconds > 0:
            self.root.after(1000, self.start_countdown, seconds - 1)
        else:
            self.countdown_label.config(text="")
            self.record_audio()

    def record_audio(self):
        duration = 8
        self.speak_label.config(text="Speak now")
        threading.Thread(target=self._record_audio, args=(duration,)).start()
        # self.t3 = threading.Thread(target=self.mfcc).start()
        # self.key=True


    def _record_audio(self, duration):
        fs = 44100
        self.audio_data = sd.rec(int(fs * duration), samplerate=fs, channels=2, dtype='int16')
        sd.wait()
        save_directory = r"C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT\\static\\audiofolder"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        filename_wav = os.path.join(save_directory, f"question_{self.current_question_index}.wav")
        sf.write(filename_wav, self.audio_data, fs)
        text = self.convert_audio_to_text(filename_wav)
        filename_text = os.path.join(save_directory, f"question_{self.current_question_index}.txt")
        with open(filename_text, 'w') as text_file:
            text_file.write(text)
        self.speak_label.config(text="")
        
        self.ask_question_after_delay()

    def convert_audio_to_text(self, audio_file):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            self.audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(self.audio_data)
            filename_answer = os.path.join(r'C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT\\Answer\\Androidanswer',
                                           f"Ans{self.current_question_index}.txt")
            with open(filename_answer, 'r') as answer_file:
                answer_text = answer_file.read()
                similarity = SequenceMatcher(None, text.lower(), answer_text.lower()).ratio()
            
            result_text = f"Question {self.current_question_index} - Similarity with Answer: {similarity * 100:.2f}%"
            
            self.similarity.append(result_text)
            self.save_result_to_csv(result_text)
        except sr.UnknownValueError:
            result_text = f"Question {self.current_question_index} - Couldn't Recognize the Answer"
            self.similarity.append(result_text)
            self.save_result_to_csv(result_text)
            return result_text
        except sr.RequestError as e:
            return f"Could not request results from Speech Recognition service; {e}"
        return text

    def save_result_to_csv(self, result_text):
        csv_filename = os.path.join(r"C:\\Users\\hp\\Desktop\\INTERVIEW_&_CHATBOT", "results.csv")
        if not os.path.isfile(csv_filename):
            with open(csv_filename, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["Question Number", "Similarity Result"])
        with open(csv_filename, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([self.current_question_index, result_text])

    def clear_question(self):
        self.question_label.config(text="")
        self.speak_label.config(text="")
        self.ask_question_after_delay()

    def show_completion_message(self):
        self.question_label.config(text="Congratulations! You have completed the quiz.")
        self.start_button.config(state=tk.NORMAL)
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            
        if self.emotion_thread is not None and self.emotion_thread.is_alive():
            self.emotion_thread.join()
            threading.join()
            
        max_emotion_label = max(self.emotion_counts, key=self.emotion_counts.get)
        max_emotion_count = self.emotion_counts[max_emotion_label]
        self.max_emotion_label = max_emotion_label
        self.max_emotion_count = max_emotion_count
        

        try:
            result_document = {
                    'uuid':self.session_data['uuid'],
                    'email': self.session_data['email'],

                    'max_emotion_label': self.max_emotion_label,
                    'max_emotion_count': self.max_emotion_count,
                    'similarity of Q1' : self.similarity[0],
                    'similarity of Q2' : self.similarity[1],
                    'similarity of Q3' : self.similarity[2],
                    'quiz_time' : time.strftime('%H:%M:%S', time.gmtime()),
                    'timestamp' : datetime.datetime.now().isoformat(),
                    'confidence level from Q1' : " ",
                    'confidence level from Q2' : " ",
                    'confidence level from Q3' : " "
                    
                }
            print('setttt')
            # Insert the document into the "result" collection
            db['resultdb'].insert_one(result_document)
            
            print("Emotions stored successfully!")
        except Exception as e:
            print(e)

    def detect_emotions(self):
        # Dictionary to store the emotion counters

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
        self.cap = cv2.VideoCapture(0)

# Variables to store emotion counts
        

        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Convert grayscale frame to RGB format
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Extract the face ROI (Region of Interest)
                face_roi = rgb_frame[y:y + h, x:x + w]

                # Perform emotion analysis on the face ROI
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                # Determine the dominant emotion
                emotion = result[0]['dominant_emotion']
                
                # Increment emotion count
                self.emotion_counts[emotion] += 1

                # Draw rectangle around face and label with predicted emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
    # Display the resulting frame
            # self.cap.imshow(frame)
            self.display_frame(frame)


    def display_frame(self, frame):
        # Convert the frame to RGB format for displaying in tkinter
          
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to ImageTk format
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the canvas from the main thread
        self.root.after(0, self.update_canvas, img_tk)

    # Update the canvas with the new frame    
    def update_canvas(self, img_tk):
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk 





#------------------------------android app-----------------------------------
@app.route('/')
def home():
    return render_template('main.html')

@app.route('/index')
def index():
    return render_template('index.html')


from flask import Flask, render_template, request, jsonify

@app.route('/content.html')
def content():
    return render_template('content.html')

# Route to handle AJAX request for loading content
@app.route('/load_content', methods=['GET'])
def load_content():
    

    import sounddevice as sd
    import soundfile as sf
    import librosa
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    import os
    import threading
    import matplotlib.pyplot as plt
    import json

    def extract_features(file_path):
        y, sr = librosa.load(file_path, duration=10, sr=None)  # Load audio file
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCC features
        mfccs_processed = np.mean(mfccs.T, axis=0)  # Take the mean of MFCCs over time
        return mfccs_processed
    # Assuming you have the paths to your "confident" and "non confident" audio folders
    confident_folder = r"Voicedata\\confident"
    non_confident_folder = r"Voicedata\\Non-confident"
    confident_features = np.array([extract_features(f"{confident_folder}/{file}") for file in os.listdir(confident_folder)])
    non_confident_features = np.array([extract_features(f"{non_confident_folder}/{file}") for file in os.listdir(non_confident_folder)])




    confident_labels = np.ones(confident_features.shape[0])
    non_confident_labels = np.zeros(non_confident_features.shape[0])

    X = np.vstack((confident_features, non_confident_features))
    y = np.hstack((confident_labels, non_confident_labels))

    

    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    print(f"Mean cross-validation accuracy: {np.mean(accuracies):.3f}")

    def predict_confidence_level(features):
        confidence_level = clf.predict_proba([features])[0][1] * 10  # Scale the output to 0-10
        return confidence_level
    
    all_confidence=[]
    # Process existing audio file for confidence level
    for i in range(3):
        

        folder_path = 'static/confidence_graph'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = f"static\\audiofolder\\question_{i+1}.wav"
        features = extract_features(file_path)
        confidence_level = predict_confidence_level(features)
        all_confidence.append(confidence_level)
        print("Confidence level (out of 10):", confidence_level)
        y, sr = librosa.load(file_path, sr=None)
        plt.figure(figsize=(10, 4))
        time_axis = np.linspace(0, len(y) / sr, num=len(y))
        plt.plot(time_axis, y, color='b')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Audio Waveform While Answering Q{i+1}')
        plt.savefig(os.path.join(folder_path, f'audio_waveform_{i+1}.png'))
        # Plot the confidence level graph
        

    

# Create the folder if it doesn't exist
        

        # Save the graph in the folder
        plt.figure(figsize=(6, 4))
        plt.plot([0, confidence_level], color='r', marker='o')
        plt.ylim(0, 10)  # Set the y-axis limits to 0 and 10
        plt.yticks(range(11))
        plt.xticks([])
        plt.xlabel('Duration (10s)')
        plt.ylabel('Confidence Level')
        plt.title(f'Confidence Level While Answering Q{i+1}')
        plt.savefig(os.path.join(folder_path, f'confidence_{i+1}.png'))
    
    print(all_confidence)
    # Load content from content.html
    with open('templates/content.html', 'r') as file:
            content = file.read()
    # new()
    # def new(self):
    #     self.write(f'<p id="all_confi">{all_confidence}</p>')
    try:
        # Create the update query to update the confidence levels for Q1, Q2, and Q3
        update_query = {
            '$set': {
                'confidence level from Q1': all_confidence[0],
                'confidence level from Q2': all_confidence[1],
                'confidence level from Q3': all_confidence[2]
            }
        }

        # Update the document in the "result" collection
        db['resultdb'].update_one({'uuid': session['uuid']}, update_query)

        print("Confidence levels stored successfully!")
    except Exception as e:
        print(e)
    
    # Simulate loading delay (replace with actual content loading if needed)
    data = {
        'content': content,
        'all_confidence': json.dumps(all_confidence)
    }

    # Return the data as JSON
    return jsonify(data)



@app.route('/gohome')
def homepage():
    return render_template('index.html')


@app.route('/service')
def servicepage():
    return render_template('services.html')


@app.route('/mainpage')
def coconutpage():
    return render_template('home1.html')


# @app.route('/cocoa')
# def cocoapage():
#     return render_template('cocoa.html')


# @app.route('/arecanut')
# def arecanutpage():
#     return render_template('arecanut.html')


# @app.route('/paddy')
# def paddypage():
#     return render_template('paddy.html')


@app.route('/about')
def aboutpage():
    return render_template('about.html')


@app.route('/enternew')
def new_user():
    return render_template('signup.html')


#----------------------------------------------------------------------------------------------------
#                                                  Suggestion page
#-----------------------------------------------------------------------------------------------------------------------------

# Generate suggestions based on similarity



def generate_emotion_suggestion(emotion):
    # Convert emotion to lowercase to match the keys in the suggestions dictionary
    emotion = emotion.lower()
    suggestions = {
        "angry": [
            "Try to remain calm and composed while answering questions.",
            "Focus on expressing your thoughts clearly and avoid getting frustrated.",
            "Identify the reasons for your anger and try to address them positively."
        ],
        "disgust": [              
            "Focus on finding a positive aspect of the questions to improve your engagement.",
            "Try to approach the questions with an open mind to reduce feelings of disgust.",
            "Identify the specific aspects of the questions that trigger feelings of disgust and try to reframe them."
        ],
        "fear": [
            "Identify the reasons for your fear and try to address them positively.",
            "Practice relaxation techniques to help reduce feelings of fear.",
            "Focus on building your confidence to overcome feelings of fear."
        ],
        "happy": [
            "Maintain a positive attitude towards the questions and express your answers confidently.",
            "Use your feelings of happiness to enhance your engagement with the questions.",
            "Share your positive feelings with others to create a more engaging interview experience."
        ],
        "sad": [
            "Try to find motivation and positivity in the questions to improve your mood.",
            "Focus on the positive aspects of the questions to help uplift your mood.",
            "Practice self-care and relaxation techniques to help improve your mood."
        ],
        "surprise": [
            "Embrace the unexpected aspects of the questions and use them to your advantage.",
            "Try to stay open-minded and adapt to the unexpected nature of the questions.",
            "Use your feelings of surprise to enhance your creativity in answering the questions."
        ],
        "neutral": [
            "Try to engage more with the questions and express your thoughts clearly.",
            "Focus on maintaining a balanced approach in your answers.",
            "Try to find aspects of the questions that interest you to improve your engagement."
        ]
    }
    return random.choice(suggestions.get(emotion, ["No specific suggestion for this emotion."]))

 #Generate suggestions based on similarity
def generate_similarity_suggestion(similarity):
    if similarity == 0:
        suggestions=[
            "It Seems like your voice was low or you have not answered the question,In this time dont panic, take a deep breath and try to answer the question confidently whatever you know about the given topic",
            "It appears you haven't addressed the question directly, you should speak louder and try to answer the question"
        ]

    elif similarity < 20:
        suggestions = [
            "Your answers show very low similarity with the expected answers. Consider revisiting the questions and reviewing the material.",
            "Focus on understanding the core concepts of the questions to improve the similarity of your answers with the expected ones,since your Your answers show very low similarity with the expected answers."
        ]
    elif similarity < 40:
        suggestions = [
            "Your answers show average similarity with the expected answers. Try to analyze the questions more deeply and provide more detailed answers.",
            "Improving the similarity of your answers with the expected ones can be achieved by practicing more and gaining a better understanding of the material,since Your answers show average similarity with the expected answers."
        ]
    elif similarity < 60:
        suggestions = [
            "Your answers show a moderate level of similarity with the expected answers. Try to provide more detailed explanations to improve further.",
            "Consider discussing the questions with others to gain different perspectives and improve the similarity of your answers."
        ]
    elif similarity < 80:
        suggestions = [
            "Your answers show a high similarity with the expected answers. You seem to understand the questions well.",
            "Maintain a consistent approach to answering questions to improve the similarity of your answers with the expected ones."
        ]
    else:
        suggestions = [
            "Your answers show a very high similarity with the expected answers. Great job!",
            "You have a deep understanding of the questions. Keep up the good work!"
        ]
    return random.choice(suggestions)

# Generate suggestions based on confidence
def generate_confidence_suggestion(confidence):
    if confidence ==0:
        suggestions = [
            "Your confidence level is not recorded for this question"
        ]
        return random.choice(suggestions)

    elif confidence <= 4:
        suggestions = [
            "Your confidence level seems very low. Consider seeking feedback from others to identify areas for improvement.",
            "To boost your confidence, try practicing answering questions in a simulated interview setting.",
            "Improving your confidence level can be achieved by gaining more knowledge and practicing regularly."
        ]
        return random.choice(suggestions)
    elif confidence > 4 and confidence <= 5:
        suggestions = [
            "Your confidence level seems low. Try practicing more and gaining more knowledge to boost your confidence.",
            "Consider reviewing the questions and your answers thoroughly to improve your confidence level.",
            "To boost your confidence, try discussing the questions with others to gain different perspectives."
        ]
        return random.choice(suggestions)
    elif confidence > 5 and confidence < 6.5:
        suggestions = [
            "Your confidence level is average. Focus on understanding the questions better to increase your confidence.",
            "Improving your confidence level can be achieved by practicing more and gaining a deeper understanding of the questions.",
            "Consider seeking feedback from others to improve your confidence in answering questions."
        ]
        return random.choice(suggestions)
    elif confidence >= 6.5 and confidence < 7:
        suggestions = [
            "Your confidence level is above average. Keep up the good work and continue to practice.",
            "Maintain a positive attitude towards the questions to boost your confidence in answering them.",
            "To further improve your confidence, try challenging yourself with more difficult questions."
        ]
        return random.choice(suggestions)
    else:
        suggestions = [
            "Your confidence level is high. Keep up the good work!",
            "Maintain a positive attitude towards the questions to boost your confidence in answering them.",
            "Your confidence level is impressive. Continue to approach the questions with confidence."
        ]
        return random.choice(suggestions)

def extract_similarity(similarity_str):
    if isinstance(similarity_str, str):
        match = re.search(r'\d+\.\d+', similarity_str)
        if match:
            return float(match.group())
    return 0.0


def generate_suggestion():
    # Retrieve the data from the database based on the session UUID
    session_data = db['resultdb'].find_one({'uuid': session['uuid']})

    if session_data:
        suggestions = []
        for i in range(1, 4):  # Assuming there are three questions
            similarity_key = f"similarity of Q{i}"
            confidence_key = f"confidence level from Q{i}"
            emotion_key = f"max_emotion_label"
            
            similarity_suggestion = generate_similarity_suggestion(extract_similarity(session_data.get(similarity_key, 0)))
            confidence_suggestion = generate_confidence_suggestion(session_data.get(confidence_key, 0))
            emotion_suggestion = generate_emotion_suggestion(session_data.get(emotion_key, ""))
            
            suggestion = f"For Question {i}: {similarity_suggestion}. {confidence_suggestion}. {emotion_suggestion}"
            suggestions.append(suggestion)

        suggestion_paragraph = "\n".join(suggestions)
        return suggestion_paragraph
    else:
        return "No session data found for the given session...please take the interview again"

@app.route('/suggestion')
def suggestion():
    suggestion_text = generate_suggestion()
    print(suggestion_text)
    return render_template('suggestion.html', suggestion=suggestion_text)


#----------------------------------------------------------------------------------------------------------------------------
@app.route('/userlogin')
def user_login():
    return render_template("login.html")

#----------------------------------------------------------------------------------------------------

@app.route('/history')
def history():
    # Retrieve all documents from the "result" collection for the current user's email
    results = db['resultdb'].find({'email': session['email']})
    return render_template('history.html', results=results)




@app.route('/info')
def predictin():
    return render_template('info.html')


# @app.route('/info1')
# def predictin1():
#     return render_template('/info1.html')

@app.route('/info', methods=['POST', 'GET'])
def predcrop():
    global app
    if request.method == 'POST':
        if request.form.get("selectedValue") == '1':

            root = tk.Tk()
            root.attributes('-topmost', True) 
            current_session_data = {
                'email': session.get('email', None),
                'uuid' : session.get('uuid', None)  # Get the 'email' value from the session, or None if it doesn't exist
                # Add any other session data you need to pass
            }
            app = QuizApp(root, session_data=current_session_data)
            root.mainloop()

        elif  request.form.get("selectedValue") == '2':
            
            root = tk.Tk()
            root.attributes('-topmost', True) 
            current_session_data = {
                'email': session.get('email', None),
                'uuid' : session.get('uuid', None)  # Get the 'email' value from the session, or None if it doesn't exist
                # Add any other session data you need to pass
            }
            app = Mlapp(root, session_data=current_session_data)
            root.mainloop()

        elif request.form.get("selectedValue") == '3':
            root = tk.Tk()
            root.attributes('-topmost', True) 
            current_session_data = {
                'email': session.get('email', None),
                'uuid' : session.get('uuid', None)  # Get the 'email' value from the session, or None if it doesn't exist
                # Add any other session data you need to pass
            }
            app = Aiapp(root, session_data=current_session_data)
            root.mainloop()

        elif request.form.get("selectedValue") == '4':
            root = tk.Tk()
            root.attributes('-topmost', True) 
            current_session_data = {
                'email': session.get('email', None),
                'uuid' : session.get('uuid', None)  # Get the 'email' value from the session, or None if it doesn't exist
                # Add any other session data you need to pass
            }
            app = Frontendapp(root, session_data=current_session_data)
            root.mainloop()

        elif request.form.get("selectedValue") == '5':
            root = tk.Tk()
            root.attributes('-topmost', True) 
            current_session_data = {
                'email': session.get('email', None),
                'uuid' : session.get('uuid', None)  # Get the 'email' value from the session, or None if it doesn't exist
                # Add any other session data you need to pass
            }
            app = Backendapp(root, session_data=current_session_data)
            root.mainloop()

        elif request.form.get("selectedValue") == '6':
            root = tk.Tk()
            root.attributes('-topmost', True) 
            current_session_data = {
                'email': session.get('email', None),
                'uuid' : session.get('uuid', None)  # Get the 'email' value from the session, or None if it doesn't exist
                # Add any other session data you need to pass
            }
            app = Androidapp(root, session_data=current_session_data)
            root.mainloop()












            
    return render_template('result123.html', max_emotion_label=app.max_emotion_label,
                               max_emotion_count=app.max_emotion_count,emotions=list(app.emotion_counts.keys()), counts=list(app.emotion_counts.values()),similarity=app.similarity)
         

import nltk
nltk.download('popular')  # Ensure NLTK data is downloaded
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import pickle
import numpy as np
import json
import random
# Load the trained model and data
lemmatizer = WordNetLemmatizer()
model = load_model('modelchat.h5')
words = pickle.load(open('textschat.pkl','rb'))
classes = pickle.load(open('labelschat.pkl','rb'))
intents = json.loads(open('intents.json').read())

def clean_up_sentence(sentence):
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
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    if ints:
        res = getResponse(ints, intents)
        return res
    else:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"

from flask import Flask, render_template, request


app.static_folder = 'static'

@app.route("/info1")
def homee():
    return render_template("chatbot.html")

@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    return chatbot_response(user_text)

@app.route("/logout")
def logout():
    session['logged_in'] = False
    return render_template('login.html')




@app.route("/formpage")
def formpage():
    return render_template('form.html')


@app.route("/learning")
def learning():
    return render_template('learning.html')




try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["interviewdb"]
    users = db["users"]
    resultdb=db["resultdb"]
    print("Connected to MongoDB successfully!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")


def is_valid_password(password):
    if len(password) < 7:
        print("Password length check failed")
        return False
    if not re.search(r'[A-Z]', password):
        print("Password uppercase check failed")
        return False
    if not re.search(r'[a-z]', password):
        print("Password lowercase check failed")
        return False
    if not re.search(r'[0-9]', password):
        print("Password numeric check failed")
        return False
    return True





@app.route('/addrec', methods=['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            name = request.form['Name']
            mobile_number = request.form['MobileNumber']
            email = request.form['email']
            username = request.form['Username']
            password = request.form['password']
            
             # Validate password
            
            # Check if the email is already in use
            if users.find_one({'email': email}):
                return render_template('form.html', msg='Email already in use. Please use a different email.')
            if not is_valid_password(password):
                return render_template('form.html', msg='Password must be 7 characters long and contain at least one uppercase letter, one lowercase letter, and one numeric character.')

            # Insert new user
            users.insert_one({
                'Name': name,
                'mobile_number': mobile_number,
                'email': email,
                'username': username,
                'password': password
            })
            return render_template('form.html' , msg='Signed UP succesfully..!')
        
        except Exception as e:
            print(f"Error: {e}")
            return "An error occurred. Please try again later!"

@app.route('/logindetails', methods=['POST', 'GET'])
def logindetails():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Check if email and password match in MongoDB
        user = users.find_one({'email': email, 'password': password})
        
        if user:
            
            session['uuid'] = str(uuid.uuid4())
            session['email'] = email
            print(session['email'])
            
            return redirect(url_for('homeform'))
        else:
            return render_template('form.html', msg='Invalid credentials/SignUP Please.')



@app.route('/homeform')
def homeform():
    if 'email' in session:
        return render_template('home1.html')
    else:
        return redirect(url_for('formpage'))



if __name__ == '__main__':
    app.secret_key = os.urandom(12) 
    app.run(debug=True)
