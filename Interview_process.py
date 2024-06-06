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



class QuizApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quiz App with Emotion Detection")
        self.root.geometry("800x600")

        # Load the pre-trained face detection classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Load the pre-trained emotion recognition model
        self.emotion_model = load_model(r"C:\Users\hp\Desktop\INTERVIEW_&_CHATBOT\my_model.zip")
        self.emotionnnnnx_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        self.start_button = tk.Button(root, text="Start", command=self.start_quiz, bg="#aaffaa")
        self.start_button.pack(pady=20)

        self.question_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.question_label.pack()

        self.countdown_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.countdown_label.pack()

        self.speak_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.speak_label.pack()

        # Create a canvas to display the camera feed
        self.canvas = tk.Canvas(root, width=600, height=400)
        self.canvas.pack()

        # Initialize cap as None
        self.cap = None
        self.emotion_thread = None  # Thread for emotion detection
        self.emotion_start_times = {label: None for label in self.emotion_labels}  # Initialize emotion start times

    def start_quiz(self):
        self.start_button.config(state=tk.DISABLED)  # Disable start button during quiz
        self.questions = ["Define Python ?", "What is pandas in python ?", "What is numpy in python ?"]
        self.current_question_index = 0

        # Start emotion detection thread
        self.emotion_thread = threading.Thread(target=self.detect_emotions)
        self.emotion_thread.start()

        self.ask_question_after_delay()

    def ask_question_after_delay(self):
        if self.current_question_index < len(self.questions):
            question = self.questions[self.current_question_index]
            self.question_label.config(text=question)
            self.start_countdown(5)  # 5 seconds for each question
            self.current_question_index += 1
        else:
            self.show_completion_message()

    def start_countdown(self, seconds):
        self.countdown_label.config(text=f"Time remaining: {seconds} seconds")
        if seconds > 0:
            self.root.after(1000, self.start_countdown, seconds - 1)
        else:
            self.countdown_label.config(text="")
            self.record_audio()

    def record_audio(self):
        duration = 5  # 5 seconds for each question
        self.speak_label.config(text="Speak now")
        threading.Thread(target=self._record_audio, args=(duration,)).start()

    def _record_audio(self, duration):
        fs = 44100  # Sample rate
        audio_data = sd.rec(int(fs * duration), samplerate=fs, channels=2, dtype='int16')
        sd.wait()

        # Specify the directory where you want to save the audio files
        save_directory = r"C:\Users\hp\Desktop\INTERVIEW_&_CHATBOT"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the recorded audio to a WAV file using soundfile
        filename_wav = os.path.join(save_directory, f"question_{self.current_question_index}.wav")
        sf.write(filename_wav, audio_data, fs)

        # Convert the audio to text using SpeechRecognition
        text = self.convert_audio_to_text(filename_wav)

        # Save the text to a text file
        filename_text = os.path.join(save_directory, f"question_{self.current_question_index}.txt")
        with open(filename_text, 'w') as text_file:
            text_file.write(text)

        self.speak_label.config(text="")  # Clear the label after recording
        self.ask_question_after_delay()

    def convert_audio_to_text(self, audio_file):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_data)

            # Check for similarity with the specific answer file
            filename_answer = os.path.join(r'C:\Users\hp\Desktop\INTERVIEW_&_CHATBOT',
                                           f"Ans{self.current_question_index}.txt")
            with open(filename_answer, 'r') as answer_file:
                answer_text = answer_file.read()
                similarity = SequenceMatcher(None, text.lower(), answer_text.lower()).ratio()

            result_text = f"Question {self.current_question_index} - Similarity with Answer: {similarity * 100:.2f}%"

            # Save the result to a CSV file
            self.save_result_to_csv(result_text)

        except sr.UnknownValueError:
            # Handle recognition failure
            return ""
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

        return text  # Return text if recognition is successful

    def save_result_to_csv(self, result_text):
        csv_filename = os.path.join(r"C:\ Users\ hp\Desktop\ INTERVIEW_&_CHATBOT", "results.csv")

        # Check if the CSV file exists, and create headers if it doesn't
        if not os.path.isfile(csv_filename):
            with open(csv_filename, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["Question Number", "Similarity Result"])

        # Append the result to the CSV file
        with open(csv_filename, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([self.current_question_index, result_text])

    def clear_question(self):
        self.question_label.config(text="")
        self.speak_label.config(text="")
        self.ask_question_after_delay()

    def show_completion_message(self):
        self.question_label.config(text="Congratulations! You have completed the quiz.")
        self.start_button.config(state=tk.NORMAL)  # Enable start button for the next round

        # Release the camera and close the window if cap is not None
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()

        # Stop emotion detection thread
        if self.emotion_thread is not None and self.emotion_thread.is_alive():
            self.emotion_thread.join()

        # Find the emotion with the highest count
        max_emotion_label = max(self.emotion_counters, key=self.emotion_counters.get)
        max_emotion_count = self.emotion_counters[max_emotion_label]

        # Print the emotion with the highest count
        print(f"The emotion with the highest count is {max_emotion_label} with count {max_emotion_count}.")

    def detect_emotions(self):
        # Dictionary to store the emotion counters
        self.emotion_counters = {label: 0 for label in self.emotion_labels}

        # Open the system camera (0 represents the default camera, you can change it if you have multiple cameras)
        self.cap = cv2.VideoCapture(0)

        while True:
            # Read a frame from the camera
            ret, frame = self.cap.read()

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Perform face detection
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            # Iterate through detected faces
            for (x, y, w, h) in faces:
                # Extract the face region
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                # Normalize the pixel values
                roi_gray = roi_gray / 255.0

                # Reshape for the model input
                roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

                # Perform emotion prediction
                emotion_prediction = self.emotion_model.predict(roi_gray)

                # Get the dominant emotion
                dominant_emotion = self.emotion_labels[np.argmax(emotion_prediction)]

                # Display the emotion label on the frame
                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                            cv2.LINE_AA)

                # Draw rectangles around the detected faces
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Check if the emotion is the same as the previous frame
                if self.emotion_start_times[dominant_emotion] is None:
                    self.emotion_start_times[dominant_emotion] = time.time()
                elif time.time() - self.emotion_start_times[dominant_emotion] > 2:
                    # If the emotion has been detected for more than 2 seconds, increment the counter
                    self.emotion_counters[dominant_emotion] += 1
                    print(f"{dominant_emotion} count: {self.emotion_counters[dominant_emotion]}")

            # Display the resulting frame
            self.display_frame(frame)

            # Break the loop if 'Esc' key is pressed
            if cv2.waitKey(1) == 27:  # 27 is the ASCII code for 'Esc'
                break

        # Release the camera and close the window
        self.cap.release()
        cv2.destroyAllWindows()

    def display_frame(self, frame):
        # Convert the frame to RGB format for displaying in tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to ImageTk format
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the canvas with the new frame
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk  # Keep a reference to avoid garbage collection

if __name__ == "__main__":
    root = tk.Tk()
    app = QuizApp(root)
    root.mainloop()
