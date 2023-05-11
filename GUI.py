import cv2
import threading

#from keras.backend import set_session

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from PIL import ImageTk
import os


matplotlib.use('Agg')

# 表情标签
emotion_labels = ['angry', 'disguist', 'fear', 'happy', 'sad', 'surprise', 'normal']

# 人脸检测器导入
cascPath = 'core/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

num = 0

path = "result"
isExists = os.path.exists(path)
print(isExists)
if not isExists:
    os.makedirs(path)


tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()


tf.compat.v1.keras.backend.set_session(sess)
# 加载训练好的模型
model = load_model('core/my_model.h5')

# 表情检测
def predict_emotion(face_image_gray):
    face_image_gray = face_image_gray * (1. / 255)
    resized_img = cv2.resize(face_image_gray, (48, 48))
    rsz_img = []
    rsh_img = []
    results = []
    rsz_img.append(resized_img[:, :])  # resized_img[1:46,1:46]
    rsz_img.append(resized_img[2:45, :])
    rsz_img.append(cv2.flip(rsz_img[0], 1))
    i = 0
    for rsz_image in rsz_img:
        rsz_img[i] = cv2.resize(rsz_image, (48, 48))
        i += 1
    for rsz_image in rsz_img:
        rsh_img.append(rsz_image.reshape(1, 48, 48, 1))
    i = 0
    for rsh_image in rsh_img:
        global sess
        global graph
        with graph.as_default():
            tf.compat.v1.keras.backend.set_session(sess)
            list_of_list = model.predict(rsh_image, batch_size=32, verbose=1)  # predict
        result = [prob for lst in list_of_list for prob in lst]
        results.append(result)
    angry=0
    disguist=0
    fear=0
    happy=0
    sad=0
    surprise=0
    normal = 0
    for result in results:
        angry = angry + result[0]
        disguist = disguist + result[1]
        fear = fear + result[2]
        happy = happy + result[3]
        sad = sad + result[4]
        surprise = surprise + result[5]
        normal = normal + result[6]
    angry = angry / len(results)
    disguist = disguist / len(results)
    fear = fear / len(results)
    happy = happy / len(results)
    sad = sad / len(results)
    surprise = surprise / len(results)
    normal = normal / len(results)
    return angry, disguist, fear, happy, sad, surprise, normal


def get_photo():

    path = entry.get()
    print(path)
    img = cv2.imread(path)
    # 灰度处理
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    pic = ""
    for (x, y, w, h) in faces:
        face_image_gray = img_gray[y:y + h, x:x + w]
        # face_image_gray = cv2.resize(face_image_gray, (48, 48))
        # face_image_gray = face_image_gray * (1. / 255)  # 归一化

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        angry, disguist, fear, happy, sad, surprise, normal = predict_emotion(face_image_gray)
        num_list = [angry, disguist, fear, happy, sad, surprise, normal]
        name_list = ['angry', 'disguist', 'fear', 'happy', 'sad', 'surprise', 'normal']
        plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list)

        t_size = 2
        www = int((w + 10) * t_size / 100)
        www_s = int((w + 20) * t_size / 100) * 2 / 5

        cv2.putText(img, name_list[num_list.index(max(num_list))], (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    www_s, (255, 0, 255), thickness=www, lineType=1)
        global num
        pic = "result/"+str(num)+".png"
        print(pic)
        num = num+1
        plt.savefig(pic)
        plt.close('all')
    data_img = ImageTk.PhotoImage(file=pic)
    panel.config(image=data_img)

    cv2.imshow('img', img)
    cv2.waitKey(0)


def open_camera():
    global video_capture
    video_capture = cv2.VideoCapture(0)

    video_capture.set(cv2.CAP_PROP_FPS, 60)
    global flag
    flag = False
    while True:
        if flag:
            break

        ret, frame = video_capture.read()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            img_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(120, 120),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        print(ret)
        print(faces)
        if len(faces):
            print(faces)
            pic = ""
            for (x, y, w, h) in faces:
                print("================================")
                face_image_gray = img_gray[y:y + h, x:x + w]
                # face_image_gray = cv2.resize(face_image_gray, (48, 48))
                # face_image_gray = face_image_gray * (1. / 255)  # 归一化
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                angry, disguist, fear, happy, sad, surprise, normal = predict_emotion(face_image_gray)
                #print(angry, disguist, fear, happy, sad, surprise, normal)
                num_list = [angry, disguist, fear, happy, sad, surprise, normal]
                name_list = ['angry', 'disguist', 'fear', 'happy', 'sad', 'surprise', 'normal']
                font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
                cv2.putText(frame, name_list[num_list.index((max(num_list)))], (x+30, y+30), font, 1, (255,0,255), 4)

                    #fig = plt.figure()  # 新图 0

                plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list)
                global num
                pic = "result/" + str(num) + ".png"
                plt.savefig(pic)
                plt.close('all')

            img = ImageTk.PhotoImage(file=pic)
            panel.config(image=img)
        cv2.imshow('Video', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            flag = True
            cv2.destroyAllWindows()
            video_capture.release()
            break
#        video_capture.release()
#        cv2.destroyAllWindows()

def thread1():
    t1 = threading.Thread(target=open_camera)
    t1.daemon = True
    t1.start()


def thread2():
    t1 = threading.Thread(target=get_photo)
    t1.daemon = True
    t1.start()


def close_camera():
    global video_capture
    global flag
    flag = True
    cv2.destroyAllWindows()
    video_capture.release()


def choose_file():  # 选择文件
    selectFileName = filedialog.askopenfilename(title='选择文件')
    entry.delete(0, 'end')
    entry.insert(0, selectFileName)


root = tk.Tk()
root.title("人脸表情识别")

# 居中显示GUI
width = 680
height = 600
screenwidth = root.winfo_screenwidth()
screenheight = root.winfo_screenheight()
alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth-width)/2, (screenheight-height)/2)
root.geometry(alignstr)


OB = tk.Button(root, text="开启摄像头", command=thread1)
# OB.pack()
OB.place(x=250, y=20)
CB = tk.Button(root, text="关闭摄像头", command=close_camera)
# CB.pack()
CB.place(x=320, y=20)
entry = tk.Entry(root, width=50)
# entry.pack()
entry.place(x=140, y=65)
Button1 = tk.Button(root, text='选择图片', command=choose_file)
# Button1.pack()
Button1.place(x=250, y=100)
Button2 = tk.Button(root, text='开始分析', command=thread2)
# Button2.pack()
Button2.place(x=320, y=100)
label = tk.Label(root, text='分析结果', font=("微软雅黑", 15, "bold"))
# label.pack()
label.place(x=275, y=130)

panel = tk.Label(root)
# panel.pack()
panel.place(x=10, y=160, width=650, height=430)

# 进入消息循环
root.mainloop()