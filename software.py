# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from imutils.video import VideoStream
from imutils.video import FPS
import time
from tkinter import Label, Frame, Button, Entry, Tk, X
from tkinter import messagebox
import pandas as pd


ARIAL = ("arial", 10, "bold")


class FacialUi:
    def __init__(self, root):
        self.root = root
        self.header = Label(self.root,
                            text="Facial Recognition",
                            bg="#0019fc",
                            fg="white",
                            font=("arial", 20, "bold"))
        self.header.pack(fill=X)
        self.frame = Frame(self.root, bg="#0019fc", width=800, height=500)
        root.geometry("800x500")
        self.button1 = Button(self.frame,
                              text=("Facial Recognition."
                                    "Click to begin transactions"),
                              bg="#50A8B0",
                              fg="white",
                              font=ARIAL,
                              command=self.begin_page)
        self.q = Button(self.frame,
                        text="Quit",
                        bg="#50A8B0",
                        fg="white",
                        font=ARIAL,
                        command=self.root.destroy)
        self.q.place(x=275, y=340, width=200, height=40)
        self.button1.place(x=155, y=230, width=500, height=30)
        self.frame.pack()
        self.countter = 2

    def begin_page(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#0019fc", width=800, height=500)
        root.geometry("800x500")
        self.enroll = Button(self.frame,
                             text="Enroll",
                             bg="#50A8B0",
                             fg="white",
                             font=ARIAL,
                             command=self.enroll_user)
        self.verify = Button(self.frame,
                             text="Verify Page",
                             bg="#50A8B0",
                             fg="white",
                             font=ARIAL,
                             command=self.verify_page)
        self.q = Button(self.frame,
                        text="Quit",
                        bg="#50A8B0",
                        fg="white",
                        font=ARIAL,
                        command=self.root.destroy)
        self.enroll.place(x=0, y=315, width=200, height=50)
        self.verify.place(x=600, y=315, width=200, height=50)
        self.q.place(x=340, y=340, width=120, height=20)
        self.frame.pack()

    def verify_page(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#0019fc", width=800, height=500)
        # Login Page Form Components
        self.label1 = Label(self.frame,
                            text="Note:",
                            bg="#0019fc",
                            fg="white",
                            font=ARIAL)
        self.label2 = Label(self.frame,
                            text=("1.By clicking on the 'Verify Face Id' "
                                  " button, we proceed to perform facial "
                                  "recognition."),
                            bg="#0019fc",
                            fg="white",
                            font=ARIAL)
        self.label3 = Label(self.frame,
                            text=("2.Each capture will take 15 seconds, "
                                  "you are required to move your face in "
                                  "different directions while being "
                                  "captured."),
                            bg="#0019fc",
                            fg="white",
                            font=ARIAL)
        self.label4 = Label(self.frame,
                            text=("6.To begin, click the"
                                  "'Verify Face Id' button below"),
                            bg="#0019fc",
                            fg="white",
                            font=ARIAL)
        self.button = Button(self.frame,
                             text="Verify Face Id",
                             bg="#50A8B0",
                             fg="white",
                             font=ARIAL,
                             command=self.video_check)
        self.q = Button(self.frame,
                        text="Quit",
                        bg="#50A8B0",
                        fg="white",
                        font=ARIAL,
                        command=self.root.destroy)
        self.b = Button(self.frame,
                        text="Back",
                        bg="#50A8B0",
                        fg="white",
                        font=ARIAL,
                        command=self.begin_page)
        self.label1.place(x=100, y=100, width=600, height=20)
        self.label2.place(x=100, y=120, width=600, height=20)
        self.label3.place(x=100, y=140, width=600, height=20)
        self.label4.place(x=100, y=160, width=600, height=20)
        self.button.place(x=100, y=250, width=600, height=30)
        self.q.place(x=480, y=360, width=120, height=20)
        self.b.place(x=280, y=360, width=120, height=20)
        self.frame.pack()

    def enroll_user(self):
        self.frame.destroy()
        self.frame = Frame(self.root,
                           bg="#0019fc",
                           width=800,
                           height=500)
        # Login Page Form Components
        self.userlabel = Label(self.frame,
                               text="Full Name",
                               bg="#0019fc",
                               fg="white",
                               font=ARIAL)
        self.uentry = Entry(self.frame,
                            bg="honeydew",
                            highlightcolor="#50A8B0",
                            highlightthickness=2,
                            highlightbackground="white")
        self.button1 = Button(self.frame,
                              text="Click to enroll",
                              bg="#50A8B0",
                              fg="white",
                              font=ARIAL,
                              command=self.write_to_csv)
        self.button2 = Button(self.frame,
                              text=("Click to go to video capture"
                                    "after enrolling"),
                              bg="#50A8B0",
                              fg="white",
                              font=ARIAL,
                              command=self.video_capture_page)
        self.q = Button(self.frame,
                        text="Quit",
                        bg="#50A8B0",
                        fg="white",
                        font=ARIAL,
                        command=self.root.destroy)
        self.b = Button(self.frame,
                        text="Back",
                        bg="#50A8B0",
                        fg="white",
                        font=ARIAL,
                        command=self.begin_page)
        self.userlabel.place(x=125, y=100, width=120, height=20)
        self.uentry.place(x=153, y=130, width=200, height=20)
        self.button1.place(x=155, y=230, width=180, height=30)
        self.button2.place(x=355, y=230, width=350, height=30)
        self.q.place(x=480, y=360, width=120, height=20)
        self.b.place(x=280, y=360, width=120, height=20)
        self.frame.pack()

    def write_to_csv(self):
        import csv
        from random import randint
        n = 10; range_start = 10**(n-1); range_end = (10**n)-1
        n = 5; range_start = 10**(n-1); range_end = (10**n)-1
        unique_id = randint(range_start, range_end)
        name = self.uentry.get()
        with open(r'user_details.csv', 'a', newline='\n') as f:
            writer = csv.writer(f)
            writer.writerow([unique_id, name])
        messagebox._show("Enrollment Info!", "Successfully Enrolled!")

    def video_capture_page(self):
        self.frame.destroy()
        self.frame = Frame(self.root,
                           bg="#0019fc",
                           width=800,
                           height=500)
        # Login Page Form Components
        self.label1 = Label(self.frame,
                            text="Note:",
                            bg="#0019fc",
                            fg="white",
                            font=ARIAL)
        self.label2 = Label(self.frame,
                            text=("1.By clicking on the 'Capture' "
                                  "button below, your image gets captured"),
                            bg="#0019fc",
                            fg="white",
                            font=ARIAL)
        self.label3 = Label(self.frame,
                            text=("2.You will be required to capture "
                                  "5 images for full registration"),
                            bg="#0019fc",
                            fg="white",
                            font=ARIAL)
        self.label4 = Label(self.frame,
                            text=("3.To capture each image click the "
                                  "space bar on your keyboard when the "
                                  "camera turn on:"),
                            bg="#0019fc",
                            fg="white",
                            font=ARIAL)
        self.label5 = Label(self.frame,
                            text=("4. Please wait till you are notified that "
                                  "your capture was successful before "
                                  "leaving the page"),
                            bg="#0019fc",
                            fg="white",
                            font=ARIAL)
        self.label6 = Label(self.frame,
                            text=("5.To begin, click the 'Capture' button "
                                  "below and click the space bar to capture "
                                  "a new image"),
                            bg="#0019fc",
                            fg="white",
                            font=ARIAL)
        self.button = Button(self.frame,
                             text="Capture",
                             bg="#50A8B0",
                             fg="white",
                             font=ARIAL,
                             command=self.captureuser)
        self.q = Button(self.frame,
                        text="Quit",
                        bg="#50A8B0",
                        fg="white",
                        font=ARIAL,
                        command=self.root.destroy)
        self.b = Button(self.frame,
                        text="Back",
                        bg="#50A8B0",
                        fg="white",
                        font=ARIAL,
                        command=self.enroll_user)
        self.label1.place(x=100, y=100, width=600, height=20)
        self.label2.place(x=100, y=120, width=600, height=20)
        self.label3.place(x=100, y=140, width=600, height=20)
        self.label4.place(x=100, y=160, width=600, height=20)
        self.label5.place(x=100, y=180, width=600, height=20)
        self.label6.place(x=100, y=200, width=600, height=20)
        self.button.place(x=100, y=230, width=600, height=30)
        self.q.place(x=480, y=360, width=120, height=20)
        self.b.place(x=280, y=360, width=120, height=20)
        self.frame.pack()

    # Hit space bar to capture
    def captureuser(self):
        data = pd.read_csv('user_details.csv')
        name = data.loc[:, 'unique_id'].values[-1]
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("capture")

        img_counter = 0

        dirname = f'dataset/{name}'
        os.mkdir(dirname)

        while True:
            ret, frame = cam.read()
            cv2.imshow("capture", frame)

            if img_counter == 5:
                cv2.destroyWindow("capture")
                break
            if not ret:
                break
            k = cv2.waitKey(1)

            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                path = f'dataset/{name}'
                img_name = "{}.jpg".format(img_counter)
                cv2.imwrite(os.path.join(path, img_name), frame)
                print("{} written!".format(img_name))
                img_counter += 1

        cam.release()

        cv2.destroyAllWindows()

        self.get_embeddings()
        self.train_model()
        messagebox._show("Registration Info!",
                         "Face Id Successfully Registered!")

    def get_embeddings(self):
        # summary:
        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--dataset",
                        required=True,
                        help="path to input directory of faces + images")
        ap.add_argument("-e", "--embeddings", required=True,
                        help=("path to output serialized db of facial "
                              "embeddings"))
        ap.add_argument("-d", "--detector", required=True,
                        help="path to OpenCV's deep learning face detector")
        ap.add_argument("-m", "--embedding-model", required=True,
                        help=("path to OpenCV's deep learning "
                              "face embedding model"))
        ap.add_argument("-c", "--confidence", type=float, default=0.5,
                        help="minimum probability to filter weak detections")
        # args = vars(ap.parse_args())

        # load our serialized face detector from disk
        print("[INFO] loading face detector...")

        detector = (cv2
                    .dnn
                    .readNetFromCaffe('face_detection_model/deploy.prototxt',
                                      ('face_detection_model/res10_300x300_ss'
                                       'd_iter_140000.caffemodel')))
        # load our serialized face embedding model from disk
        embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

        # grab the paths to the input images in our dataset
        print("[INFO] quantifying faces...")
        imagePaths = list(paths.list_images('dataset'))
        # initialize our lists of extracted facial embeddings and
        # corresponding people names
        knownEmbeddings = []
        knownNames = []
        # initialize the total number of faces processed
        total = 0
        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(i + 1,
                                                         len(imagePaths)))
            name = imagePath.split(os.path.sep)[-2]

            # load the image, resize it to have a width of 600 pixels (while
            # maintaining the aspect ratio), and then grab the image
            # dimensions
            image = cv2.imread(imagePath)
            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]
            # construct a blob from the image
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            # apply OpenCV's deep learning-based face detector to localize
            # faces in the input image
            detector.setInput(imageBlob)
            detections = detector.forward()

            # ensure at least one face was found
            if len(detections) > 0:
                # we're making the assumption that each image has only ONE
                # face, so find the bounding box with the largest probability
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]

                # ensure that the detection with the largest probability also
                # means our minimum probability test (thus helping filter out
                # weak detections)
                if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the face ROI and grab the ROI dimensions
                    face = image[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    # ensure the face width and height are sufficiently large
                    if fW < 20 or fH < 20:
                        continue

                    # construct a blob for the face ROI, then pass the blob
                    # through our face embedding model to obtain the 128-d
                    # quantification of the face
                    faceBlob = cv2.dnn.blobFromImage(face,
                                                     1.0 / 255,
                                                     (96, 96),
                                                     (0, 0, 0),
                                                     swapRB=True,
                                                     crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()

                    # add the name of the person + corresponding face
                    # embedding to their respective lists
                    knownNames.append(name)
                    knownEmbeddings.append(vec.flatten())
                    total += 1
        # dump the facial embeddings + names to disk
        print("[INFO] serializing {} encodings...".format(total))
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open('output/embeddings.pickle', "wb")
        f.write(pickle.dumps(data))
        f.close()

    def train_model(self):
        # summary
        print("[INFO] loading face embeddings...")
        data = pickle.loads(open('output/embeddings.pickle', "rb").read())
        le = LabelEncoder()
        labels = le.fit_transform(data["names"])
        # train the model used to accept the 128-d embeddings of the face and
        # then produce the actual face recognition
        print("[INFO] training model...")
        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(data["embeddings"], labels)
        # write the actual face recognition model to disk
        f = open('output/recognizer.pickle', "wb")
        f.write(pickle.dumps(recognizer))
        f.close()

        # write the label encoder to disk
        f = open('output/le.pickle', "wb")
        f.write(pickle.dumps(le))
        f.close()
        print("[INFO] done training model.")

    def video_check(self):
        detector = cv2.dnn.readNetFromCaffe('face_detection_model/'
                                            'deploy.prototxt',
                                            ('face_detection_model/res10_300x3'
                                             '00_ssd_iter_140000.caffemodel'))
        # summary
        # load our serialized face embedding model from disk
        print("[INFO] loading face recognizer...")
        embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

        # load the actual face recognition model along with the label encoder
        recognizer = (pickle.loads(open('output/recognizer.pickle', "rb")
                                   .read()))
        le = pickle.loads(open('output/le.pickle', "rb").read())

        # initialize the video stream, then allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)

        # run check for only 15seconds and then stop
        timeout = time.time() + 15

        # start the FPS throughput estimator
        fps = FPS().start()

        # loop over frames from the video file stream
        real_user_list = []
        while True:
            # run check for only 15seconds and then stop
            if time.time() > timeout:
                cv2.destroyWindow("Frame")
                break

            # grab the frame from the threaded video stream
            frame = vs.read()

            # resize the frame to have a width of 600 pixels (while
            # maintaining the aspect ratio), and then grab the image
            # dimensions
            frame = imutils.resize(frame, width=600)
            (h, w) = frame.shape[:2]

            # construct a blob from the image
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            # apply OpenCV's deep learning-based face detector to localize
            # faces in the input image
            detector.setInput(imageBlob)
            detections = detector.forward()

            # TODO: if 2 faces are detected alert the user of a warning
            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the face ROI
                    face = frame[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    # ensure the face width and height are sufficiently large
                    if fW < 20 or fH < 20:
                        continue

                    # construct a blob for the face ROI, then pass the blob
                    # through our face embedding model to obtain the 128-d
                    # quantification of the face
                    faceBlob = cv2.dnn.blobFromImage(face,
                                                     1.0 / 255,
                                                     (96, 96),
                                                     (0, 0, 0),
                                                     swapRB=True,
                                                     crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()

                    # perform classification to recognize the face
                    preds = recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = le.classes_[j]

                    # draw the bounding box of the face along with the
                    # associated probability
                    text = "{}: {:.2f}%".format(name, proba * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 0, 255), 2)
                    cv2.putText(frame, text,
                                (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    # TODO: Handle if 2 faces are given.
                    # Decision boundary
                    if (name == 'unknown') or (proba * 100) < 20:
                        real_user_list.append('0000')
                        print("New Member")
                    else:
                        real_user_list.append(name)
                        cv2.destroyWindow("Frame")
                        break

            # update the FPS counter
            fps.update()

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # do a bit of cleanup
        vs.stop()
        cv2.destroyAllWindows()
        try:
            Counter(real_user_list).most_common(1)[0][0] == 'unknown'
        except IndexError:
            if self.countter != 0:
                messagebox._show("Verification Info!",
                                 "Face Id match failed! You "
                                 f"have {self.countter} trials left"
                                 )
                self.countter = self.countter - 1
                self.video_check()
            else:
                messagebox._show("Verification Info!", "Could Not View User")
                self.begin_page()
                self.countter = 2

        else:
            if Counter(real_user_list).most_common(1)[0][0] == '0000':
                if self.countter != 0:
                    messagebox._show("Verification Info!",
                                     "Face Id match failed! You "
                                     f"have {self.countter} trials left")
                    self.countter = self.countter - 1
                    self.video_check()
                else:
                    messagebox._show("Verification Info!", "New Member")
                    self.begin_page()
                    self.countter = 2

            else:
                self.real_user = int(Counter(real_user_list)
                                     .most_common(1)[0][0])
                data = pd.read_csv('user_details.csv')
                self.username = (data[data.loc[:, 'unique_id']
                                 ==
                                 self.real_user].loc[:, 'name'].values[0])
                messagebox._show("Verification Info!",
                                 f"Hello {self.realuser}")


root = Tk()
root.title("Facial Recognition")
root.geometry("600x420")
root.configure(bg="red")
obj = FacialUi(root)
root.mainloop()
