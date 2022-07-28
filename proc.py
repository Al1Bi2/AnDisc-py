import cv2
import numpy as np
import pyvirtualcam as vcam
from pyo import *
from pyvirtualcam import PixelFormat


def null(x):
    pass


def open_Cyrillic(path):
    """Open image (safe for non-ASCII paths)"""
    maskFile = path
    m_stream = open(maskFile, 'rb')
    m_bytes = bytearray(m_stream.read())
    array = np.asarray(m_bytes, dtype=np.uint8)
    mask = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    if len(mask.shape) > 2 and mask.shape[2] == 4:
        # convert the image from RGBA2RGB
        mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2BGR)
    return mask


def audio():
    """
    Translate mic pitch down and send to Virtual Audia Cable (code need to be found manually)
    """
    s = Server()
    s.setOutputDevice(13) #num of Virtual Audio Cable (search manually)
    s.boot()
    mic = Input().play()
    s.start()
    a = FreqShift(mic, shift=-60).out()


def virtual_cam(queue):
    """
    Ask frame from queue and send it to OBS or Unity Vcam
    :param queue: queue with frame
    :return:
    """
    cam_on=False

    blank_image=np.zeros((480,854,3),np.uint8)
    with vcam.Camera(854,
                     480,
                     30,
                     fmt=PixelFormat.BGR) as cam:
        while True:

            if cam_on:
                frame = queue.get()
                blank_image[:, 108:748] = frame
                cam.send(blank_image)
                cam.sleep_until_next_frame()
            else:
                if not queue.empty():
                    cam_on = True



def processing(q_in, q_out, q_2proc):
    """
    Capture image from cam and process it with OpenCV DNN module on Res10 model.
    After finding first and only face blur or mask it, then send it to virtual_cam function and to main.
    :param q_in: Queue with frames to GUI in main
    :param q_out: Queue with frames to virtual_cam
    :param q_2proc: Queue with input from GUI in main
    """

    freeze_state = True
    mask_state = False

    message = None

    maskFile = "mask.png"
    mask = cv2.imread(maskFile)

    modelFile = "model/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "model/deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    cap = cv2.VideoCapture(0)
    ######################################################################
    while True:

        #timer = time.perf_counter()

        if not q_2proc.empty():  # look for checboxes changes
            message = q_2proc.get(False)

        if message != None and message[0] != 0:
            if message[0] == 1:
                freeze_state = message[1]
            else:
                mask_state = message[1]
        ###################################################3
        ret, frame = cap.read()  # try read frame frm camera
        if ret:  # if frame exist
            h, w = frame.shape[:2]  # frame sizes
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (150, 150)), 1.0,
                                         (75, 75), (104.0, 117.0, 123.0))  # NN input configure [(150,150),1.0,(75,75)] for low systems;
                                                                           #(300, 30), 1.0, (150, 150)] for medium systems;
                                                                           # [(300,300),1.0,(300,300)] - best

            net.setInput(blob)  # NN search

            faces = net.forward()  # set of faces found by NN

            confidence = faces[0, 0, 0, 2]
            if confidence > 0.2:
                box = faces[0, 0, 0, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")

                y = 0 if y < 0 else y  # if boxes are out frame borders
                y1 = h - 1 if y1 > h - 1 else y1
                x = 0 if x < 0 else x
                x1 = w - 1 if x1 > w - 1 else x1

                if mask_state:  # Apply blur or mask on image
                    frame[y:y1, x:x1] = cv2.resize(mask, (x1 - x, y1 - y), interpolation=cv2.INTER_NEAREST)
                else:
                    frame[y:y1, x:x1] = cv2.GaussianBlur(frame[y:y1, x:x1], (51, 101), 0)

        if not freeze_state or confidence > 0.2:
            q_out.put(frame)  # send frame to virtual camera

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            q_in.put(frame)  # send frame to GUI

            #print(1 / (time.perf_counter() - timer))
        if message != None and message[0] == 0:  # Change of mask pic
            mask = open_Cyrillic(message[1])
        message = None
    cap.release()
