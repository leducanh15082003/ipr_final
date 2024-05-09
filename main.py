import tkinter as tk
from tkinter import filedialog
from ObjectDetectionImage import detect_objects
from ObjectDetectionCamera import detect_objects_camera
from ObjectDetectionVideo import detect_object_video

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        detect_objects(file_path)

def select_camera():
    detect_objects_camera()

def select_video():
    file_path = filedialog.askopenfilename()
    if file_path:
        detect_object_video(file_path)

def main():
    root = tk.Tk()
    root.title("Object Detection")
    root.geometry("300x200")

    btn_select_image = tk.Button(root, text="Image", command=select_image)
    btn_select_image.pack(pady=20)

    btn_select_camera = tk.Button(root, text="Camera", command=select_camera)
    btn_select_camera.pack(pady=20)

    btn_select_camera = tk.Button(root, text="Video", command=select_video)
    btn_select_camera.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
