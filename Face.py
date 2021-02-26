import tkinter as tk
window = tk.Tk()  
window.title("Face_Recogniser") 
window.configure(background ='white') 
window.grid_rowconfigure(0, weight = 1) 
window.grid_columnconfigure(0, weight = 1) 
message = tk.Label( 
    window, text ="Face-Recognition-System",  
    bg ="blue", fg = "white", width = 50,  
    height = 3, font = ('times', 30, 'bold'))  
      
message.place(x = 80, y = 20) 
  
lbl2 = tk.Label(window, text ="Name",  
width = 20, fg ="red", bg ="white",  
height = 2, font =('times', 15, ' bold '))  
lbl2.place(x = 300, y = 300) 
  
txt2 = tk.Entry(window, width = 50,  
bg ="white", fg ="green",  
font = ('times', 15, ' bold ')  ) 
txt2.place(x = 600, y = 315) 

lbl3 = tk.Label(window, text ="Email of sender",  
width = 20, fg ="red", bg ="white",  
height = 2, font =('times', 15, ' bold '))  
lbl3.place(x = 300, y = 350) 
  
txt3 = tk.Entry(window, width = 50,  
bg ="white", fg ="green",  
font = ('times', 15, ' bold ')  ) 
txt3.place(x = 600, y = 365) 

lbl4 = tk.Label(window, text ="Email of Receiver",  
width = 20, fg ="red", bg ="white",  
height = 2, font =('times', 15, ' bold '))  
lbl4.place(x = 300, y = 400) 
  
txt4 = tk.Entry(window, width = 50,  
bg ="white", fg ="green",  
font = ('times', 15, ' bold ')  ) 
txt4.place(x = 600, y = 415)

# step 1 collecting automated samples

import cv2
import os
from os import listdir
from os.path import isfile,join
import numpy as np


#data_path='C:/Users/KIIT/Desktop/oepncv/face_data/'
#onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]
#count=len(onlyfiles)+count

def face_detector(img):
    face_classifier=cv2.CascadeClassifier('C:/Users/KIIT/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    gray=cv2.cvtColor(np.array(img),cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.5,5)
    if faces is ():
        return None
    else:
        for (x,y,w,h) in faces:
            cropped_faces=img[y:y+h,x:x+w]
        return cropped_faces

def new_user():
    count2=0
    name=(txt2.get())
    print(name)
    while True:
        cap=cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        count=-1
        while True:
            ret,frame=cap.read()
            if face_detector(frame) is not None:
                count+=1
                face=cv2.resize(face_detector(frame),(400,400))
                face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                print("Detecting")
            #file_save_path=('C:/Users/KIIT/Desktop/oepncv/face_data/'+str(count)+'.jpg')
            #cv2.imwrite(file_save_path,face)
                if count==0:
                    face_data= np.expand_dims(np.array(face), axis= 0)
                #face_data=face
                else:
                    face1= np.expand_dims(np.array(face), axis= 0)
                    face_data= np.append(face_data, face1, axis= 0)
            
            #print(face_data_list[count].shape)
                cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.imshow('face detecting',face)
            else:
                #message.configure(text="Face not found")
                pass
            if (cv2.waitKey(1)==13 or count==9):
                break
    #np.save('face_data_1',face_data)
        cap.release()
        cv2.destroyAllWindows()
        #message.configure(text="collecting samples complete")
        with open('names.txt','a') as file:
            #file.write(name)
            file.write(name)
            file.write(",")
        filesize=os.path.getsize('face_data_1.npy')
        if filesize<=2:
            np.save('face_data_1',face_data)
        else:
            c=np.load('face_data_1.npy',allow_pickle=True)
            #c1= np.expand_dims(np.array(c), axis= 0)
            face_data_final= np.append(c, face_data, axis= 0)
            np.save('face_data_1',face_data_final)
        count2=count2+1
        if count2==1:
            break
    
def label_form():
    c=np.load('face_data_1.npy',allow_pickle=True)
    a=range(len(c))
    label_data=np.expand_dims(np.array(a),axis=0)
    np.save('lables',label_data)




from os import listdir
import os
import cv2
import numpy as np
from os.path import isfile,join
import pandas as pd
from datetime import date 
#data_path='C:/Users/KIIT/Desktop/oepncv/face_data/'
#onlyfiles_index=[f for f in listdir(data_path)]
#onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]
#trainingdata,labels=[],[]
'''for i,files in enumerate(onlyfiles):
    image_path=data_path+onlyfiles[i]
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    trainingdata.append(np.asarray(images,dtype=np.uint8))
    label_index=onlyfiles_index[i].split('.')
    labels.append(label_index[0])'''


def label_decoder(image_id):
    return int(image_id/10)

def face_detector_recognise(img,size=0.5):
    face_classifier=cv2.CascadeClassifier('C:/Users/KIIT/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.5,5)
    if faces is ():
        return img,[]
    else:
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
            roi=img[y:y+h,x:x+w]
            roi=cv2.resize(roi,(400,400))
        return img,roi
    
    

def face_recognizer():
    data=[]
    df=pd.DataFrame(data,columns=['Name','Time'])
    dict1={}
    names=[]
    with open('names.txt','r') as file:
        x=file.readline()
        z=x.split(",")
    for i,j in enumerate(z[:-1]):
        dict1[i]=j
    #dict1={0:'Arjun Biswas',1:'Bireswar Biswas',2:'Suchandra Biswas'}
    training_data=np.load('face_data_1.npy')
    training_labels=np.load('lables.npy')
    training_labels=np.transpose(training_labels)
    #labels=np.asarray(labels,dtype=np.int32)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(training_data,training_labels)
    print('Training complete')
    cap=cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    while True:
        ret,frame=cap.read()
        image,face=face_detector_recognise(frame)
        try:
        
            face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            result=model.predict(face)
        
            if result[1]<500:
                confidence= int(100*(1-(result[1])/300))
            #print(confidence)
        
            if confidence>=85:
                
            #print(result[0])
                label_img=label_decoder(result[0])
            #print(result[0])
                cv2.putText(image,dict1[label_img],(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.imshow('Face cropper',image)
                today = date.today() 
                df.loc[len(df.index)]=[dict1[label_img],today]
                filesize=os.path.getsize('Attendance.xlsx')
                #print(filesize)
                if filesize<=7543:
                    df.to_excel('Attendance.xlsx')
                    print("Attendance updated")
                else:
                    df2=pd.read_excel('Attendance.xlsx')
                    df3=pd.concat([df2,df],axis=0)
                    df3.to_excel('Attendance.xlsx')
                    print("Attendance updated")
                cap.release()
                cv2.destroyAllWindows()
                break
            else:
                cv2.putText(image,"unknown user",(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.imshow('Face cropper',image)
        
        
        except:
            pass
        if (cv2.waitKey(1)==13):
            break
    cap.release()
    cv2.destroyAllWindows()




def send_email():
    import smtplib 
    from email.mime.multipart import MIMEMultipart 
    from email.mime.text import MIMEText 
    from email.mime.base import MIMEBase 
    from email import encoders 

    fromaddr = txt3.get()
    toaddr = txt4.get()

     
    msg = MIMEMultipart() 

     
    msg['From'] = fromaddr 

    
    msg['To'] = toaddr 

    # storing the subject 
    msg['Subject'] = "Attendance For Today"+" "+str(date.today())

    # string to store the body of the mail 
    body = "Dear User find below the attendance for "+str(date.today())

    # attach the body with the msg instance 
    msg.attach(MIMEText(body, 'plain')) 

    # open the file to be sent 
    filename = "Attendance.xlsx"
    attachment = open("C:/Users/KIIT/Desktop/oepncv/Attendance.xlsx", "rb") 

    # instance of MIMEBase and named as p 
    p = MIMEBase('application', 'octet-stream') 

    # To change the payload into encoded form 
    p.set_payload((attachment).read()) 

    # encode into base64 
    encoders.encode_base64(p) 

    p.add_header('Content-Disposition', "attachment; filename= %s" % filename) 

    # attach the instance 'p' to instance 'msg' 
    msg.attach(p) 

    # creates SMTP session 
    s = smtplib.SMTP('smtp.gmail.com', 587) 

    # start TLS for security 
    s.starttls() 

    # Authentication 
    s.login(fromaddr, "oqytjtiewudaqpqh") 

    # Converts the Multipart msg into a string 
    text = msg.as_string() 

    # sending the mail 
    s.sendmail(fromaddr, toaddr, text) 

    # terminating the session 
    s.quit() 



new_users = tk.Button(window, text ="Register user",  
command = new_user, fg ="white", bg ="green",  
width = 20, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
new_users.place(x = 10, y = 500) 

label_forms = tk.Button(window, text ="Create Labels",  
command = label_form, fg ="white", bg ="green",  
width = 20, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
label_forms.place(x = 390, y = 500) 

face_recognizers= tk.Button(window, text ="Take Attendance",  
command= face_recognizer, fg ="white", bg ="green",  
width = 20, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
face_recognizers.place(x = 700, y = 500) 

emails = tk.Button(window, text ="Send Attendance",  
command = send_email, fg ="white", bg ="green",  
width = 20, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
emails.place(x = 500, y = 600) 

quitWindow = tk.Button(window, text ="Quit",  
command = window.destroy, fg ="white", bg ="green",  
width = 20, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
quitWindow.place(x =1000, y = 500) 
