from tkinter import *
from tkinter import messagebox
import face_recognition
import cv2
import numpy as np
import winsound
import csv


root = Tk()
root.title("IMPERSONATOR DETECTOR")
root.geometry('500x200')
label = Label(root, text="IMPERSONATOR DETECTOR", font="Verdana 15 underline")
label.place(x=100, y=50)
global img, name


def close_win(top_):
    top_.destroy()


def open_camera():
    global img, name
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (10, 20)
    fontScale = 0.5
    color = (255, 0, 0)
    thickness = 1
    capture = False
    check = testDevice(0)
    if check:
        while cap.isOpened():
            _, img = cap.read()
            face_locations = face_recognition.face_locations(img)
            if len(face_locations) == 1:
                for (top, right, bottom, left) in face_locations:
                    # cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
                    k = cv2.waitKey(1)
                    if k == ord("s") and capture == False:
                        if capture:
                            cv2.putText(img, 'already captured image of ' + str(name) + ',please exit the window', org,
                                        font,
                                        fontScale, (0, 0, 255), thickness, cv2.LINE_AA)
                        else:
                            cv2.imwrite("dataset/" + str(name) + ".jpg", img)
                            capture = True
                    if capture:
                        cv2.putText(img, 'Successfully captured image of ' + str(name) + 'Press Q to exit', org, font,
                                    fontScale, color, thickness, cv2.LINE_AA)
                cv2.putText(img, 'Press S to capture the image', (0, 440), font,
                            fontScale, (0, 255, 0), thickness, cv2.LINE_AA)
                cv2.putText(img, 'Press Q to exit', (0, 460), font,
                            fontScale, (0, 255, 0), thickness, cv2.LINE_AA)
            else:
                cv2.putText(img, 'No face detected, cannot capture image', org, font,
                            fontScale, (0, 0, 255), thickness, cv2.LINE_AA)

            cv2.imshow("frame", img)
            k = cv2.waitKey(1)
            if k == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
    else:
        messagebox.showwarning("Warning",
                               "No camera found\n please insert a camera and Try again.")


def get_name(name_, top__):
    global name
    name = name_.get("1.0", "end-1c")
    open_camera()
    close_win(top__)


def get_details():
    top_ = Toplevel(root)
    top_.geometry("750x250")
    label = Label(top_, text="please enter the name of the student")
    label.pack(pady=20)
    textBox = Text(top_, height=2, width=10)
    textBox.pack()
    Button(top_, text="Insert", command=lambda: get_name(textBox, top_)).pack(pady=5, side=TOP)


def check_pass(textBox_, top, label_):
    if textBox_.get() == "hello":
        close_win(top)
        get_details()
    elif textBox_.get() == "":
        label_.config(text="Enter a Valid Password!")
    else:
        label_.config(text="Password incorrect, please try again!")


def retrieve_input():
    top = Toplevel(root)
    top.geometry("250x100")
    label = Label(top, text="please enter the password")
    label.pack(pady=20)
    password = StringVar()  # Password variable
    passEntry = Entry(top, textvariable=password, show='*')
    passEntry.pack()
    ok = Button(top, height=1, width=10, text="ok",
                command=lambda: check_pass(password, top, label))
    # command=lambda: retrieve_input() >>> just means do this when i press the button
    ok.pack()


def testDevice(source):
    cap = cv2.VideoCapture(source)
    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', source)
        return False
    else:
        return True


def get_details_name(name):
    if name == "Unknown":
        return name
    else:
        file = open("dataset\\data2.csv")
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
            rows.append(row)
        ar = np.array(rows)
        a = np.where(ar == name)
        if len(a[0])>0:
            row = a[0]
            col = a[1]
            return ar[row][col][0]
        else:
            return [name, "Not In databse", "Not In databse", "Not In databse"]


def detection():
    video_capture = cv2.VideoCapture(0)
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('data.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             2, size)
    known_face_encodings = []
    known_face_names = []
    import os
    directory = "dataset\\"
    jpg_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]

    if len(jpg_files) > 0:
        image = [0] * len(jpg_files)
        image_face_encoding = [0] * len(jpg_files)
        for index, file in enumerate(jpg_files):
            image[index] = face_recognition.load_image_file(directory + file)
            image_face_encoding[index] = face_recognition.face_encodings(image[index])[0]
            known_face_encodings.append(image_face_encoding[index])
            known_face_names.append(str(file[:-4]))

        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
        check = testDevice(0)
        if check:
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)

                face_names = []
                faces = []
                names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)
                show_img = np.zeros((512, 512, 3), dtype="uint8")
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    crop_img = frame[top:bottom, left:right]
                    crop_img = cv2.resize(crop_img, (200,200))
                    #show_img[0:200,0:200,:]  = crop_img[0:200, 0:200, :]
                    faces.append(crop_img)
                    names.append(name)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                    if name == "Unknown":
                        winsound.Beep(2500, 2000)
                        result.write(frame)

                if len(show_img) > 0:
                    for n, image in enumerate(faces):
                        if n == 0:
                            show_img[0:200, 0:200] = image[0:200, 0:200]
                            font = cv2.FONT_HERSHEY_DUPLEX
                            if names[0] != "Unknown":
                                data = get_details_name(names[0])
                                cv2.putText(show_img, "Name:" + data[0], (220, 20), font, 0.7, (255, 255, 255), 1)
                                cv2.putText(show_img, "Roll No:" + data[1], (220, 50), font, 0.7, (255, 255, 255), 1)
                                cv2.putText(show_img, "Department:" + data[2], (220, 80), font, 0.7, (255, 255, 255), 1)
                                cv2.putText(show_img, "College:" + data[3], (220, 110), font, 0.7, (255, 255, 255), 1)
                            else:
                                cv2.putText(show_img, "Unknown", (240, 70), font, 1.2, (0, 0, 255), 1)
                        if n == 1:
                            show_img[210:410, 0:200] = image[0:200, 0:200]
                            font = cv2.FONT_HERSHEY_DUPLEX
                            if names[1] != "Unknown":
                                data = get_details_name(names[1])
                                cv2.putText(show_img, "Name:" + data[0], (220, 220), font, 0.7, (255, 255, 255), 1)
                                cv2.putText(show_img, "Roll No:" + data[1], (220, 250), font, 0.7, (255, 255, 255), 1)
                                cv2.putText(show_img, "Department:" + data[2], (220, 280), font, 0.7, (255, 255, 255), 1)
                                cv2.putText(show_img, "College:" + data[3], (220, 310), font, 0.7, (255, 255, 255), 1)
                            else:
                                cv2.putText(show_img, "Unknown", (240, 270), font, 1.2, (0, 0, 255), 1)
                    cv2.imshow("Detected persons", show_img)
                cv2.imshow('detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    video_capture.release()
                    result.release()
                    cv2.destroyAllWindows()
                    break
        else:
            messagebox.showwarning("Warning",
                                   "No camera found\n please insert a camera and Try again.")
    else:
        video_capture.release()
        messagebox.showwarning("Warning",
                               "No files in dataset, cannot run detection\n please add person and then Try again.")


ap = PhotoImage(file='add_person.png')
buttonCommit = Button(root, text="Add Person",
                      command=lambda: retrieve_input(), bg='#567', fg='White', image=ap, borderwidth=0)
# command=lambda: retrieve_input() >>> just means do this when i press the button
buttonCommit.place(x=100, y=100)
detec = PhotoImage(file='detection.png')
buttonrecog = Button(root, text="detection",
                     command=lambda: detection(), bg='#567', fg='White', image=detec, borderwidth=0)
# command=lambda: retrieve_input() >>> just means do this when i press the button
buttonrecog.place(x=300, y=100)

mainloop()
