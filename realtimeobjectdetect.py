import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from deep_translator import GoogleTranslator
from database import connect_database, insert_data, retrieve_user_input

net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection App")

        self.username_language_frame = tk.Frame(root)
        self.username_language_frame.pack()

        self.username_label = ttk.Label(self.username_language_frame, text="Username:")
        self.username_label.pack()

        self.username_entry = ttk.Entry(self.username_language_frame)
        self.username_entry.pack()

        self.language_label = ttk.Label(self.username_language_frame, text="Choose Language:")
        self.language_label.pack()

        self.language_var = tk.StringVar()
        self.language_var.set("en")
        self.language_menu = ttk.OptionMenu(self.username_language_frame, self.language_var, "en", "en", "tr", "de")
        self.language_menu.pack()

        self.continue_button = ttk.Button(self.username_language_frame, text="Continue", command=self.change_page_with_username_language)
        self.continue_button.pack()

        self.selected_username = None
        self.selected_language = None
        self.objects = []
        self.current_object = None
        self.translations = {}
        self.cap = cv2.VideoCapture(0)
        self.detection_paused = False
        self.cropped_object = None
        self.captured_image = None
        self.capture_button = ttk.Button(self.root, text="Capture", command=self.capture_image)
        self.save_button = ttk.Button(self.root, text="Save", command=self.save_translation)
        self.total_score = 0
        self.user_input_from_db = None
        self.total_operations = 0

        self.language_frame = None
        self.score_window = None

        self.save_button.configure(state=tk.DISABLED)

    def change_page_with_username_language(self):
        self.selected_username = self.username_entry.get()
        self.selected_language = self.language_var.get()
        if self.selected_username:
            self.username_language_frame.destroy()
            self.create_detection_page()

    def create_detection_page(self):
        input_frame = tk.Frame(self.root)
        input_frame.pack()

        self.translation_entry = ttk.Entry(input_frame)
        self.translation_entry.pack()

        self.capture_button.pack()

        self.save_button.pack()

        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()

        translations_frame = tk.Frame(self.root)
        translations_frame.pack()

        self.translations_text = tk.Text(translations_frame, width=40, height=10)
        self.translations_text.pack()

        self.update()

    def detect_objects(self):
        ret, frame = self.cap.read()

        if frame is None:
            return frame

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

        net.setInput(blob)
        layer_names = net.getUnconnectedOutLayersNames()
        detections = net.forward(layer_names)

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(obj[0] * frame.shape[1])
                    center_y = int(obj[1] * frame.shape[0])
                    width = int(obj[2] * frame.shape[1])
                    height = int(obj[3] * frame.shape[0])
                    x = center_x - width // 2
                    y = center_y - height // 2
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

                    self.current_object = classes[class_id]

                    self.cropped_object = frame[y:y+height, x:x+width]

        return frame

    def update(self):
        if not self.detection_paused:
            frame = self.detect_objects()
            if self.captured_image is not None:
                frame[0:self.captured_image.shape[0], 0:self.captured_image.shape[1]] = self.captured_image

            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                self.canvas.photo = self.photo

        self.root.after(10, self.update)

    def save_translation(self):
        translation = self.translation_entry.get()
        if translation and self.cropped_object is not None:
            conn, cursor = connect_database()

            user_input = translation

            translated_object_selected_lang = GoogleTranslator(source='en', target=self.selected_language).translate(self.current_object)

            other_language_translation = ""
            for lang_code in ["tr", "de"]:
                if lang_code != self.selected_language:
                    translation = GoogleTranslator(source='en', target=lang_code).translate(self.current_object)
                    other_language_translation += f"{lang_code}: {translation}\n"

            if self.selected_language != "en":
                translation = GoogleTranslator(source='en', target='en').translate(self.current_object)
                other_language_translation += f"en: {translation}\n"

            if user_input.lower() == translated_object_selected_lang.lower():
                self.total_score += 10

            insert_data(conn, cursor, self.selected_username, user_input, translated_object_selected_lang)

            self.translation_entry.delete(0, tk.END)

            self.user_input_from_db = retrieve_user_input(conn, cursor)

            self.translations_text.delete(1.0, tk.END)  
            self.translations_text.insert(tk.END, f"User ({self.selected_username}): {user_input}\n")
            self.translations_text.insert(tk.END, f"Object:\n{self.selected_language}: {translated_object_selected_lang}\n")
            self.translations_text.insert(tk.END, f"Object:\n{other_language_translation}\n")

            self.captured_image = None

            self.detection_paused = False

            self.capture_button.configure(state=tk.NORMAL)

            self.total_operations += 1

            if self.total_operations == 2:
                self.show_score()

    def capture_image(self):
        if self.cropped_object is not None:
            self.cap.release()

            self.captured_image = cv2.resize(self.cropped_object, (640, 480))

            self.cap = cv2.VideoCapture(0)
            self.detection_paused = True

            self.save_button.configure(state=tk.NORMAL)

    def show_score(self):
        if self.score_window:
            self.score_window.destroy()

        self.cap.release()

        self.score_window = tk.Toplevel(self.root)
        self.score_window.title("Total Score")

        score_label = tk.Label(self.score_window, text=f"Total Score: {self.total_score}")
        score_label.pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
