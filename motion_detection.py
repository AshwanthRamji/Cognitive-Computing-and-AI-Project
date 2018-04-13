import cv2, time, pandas
from datetime import datetime
import face_recognition as fr
from twilio.rest import Client
from VoiceIt import *

myVoiceIt = VoiceIt("")
# Your Account SID from twilio.com/console
account_sid = ""
# Your Auth Token from twilio.com/console
auth_token  = ""
client = Client(account_sid, auth_token)
first_frame =None
status_list=[None,None]
times=[]
df= pandas.DataFrame(columns=["Start","End"])
video = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
image_Ashwanth1 = fr.load_image_file("Ashwanth1.jpg")
image_Ashwanth2 = fr.load_image_file("Ashwanth2.jpg")
image_Ashwanth3 = fr.load_image_file("Ashwanth3.jpg")
#image_Ashwanth4 = fr.load_image_file("Ashwanth4.jpg")
#image_Ashwanth5 = fr.load_image_file("Ashwanth5.jpg")
image_Ashwanth6 = fr.load_image_file("Ashwanth6.jpg")
image_Srujani1 = fr.load_image_file("Srujani1.jpg")
image_Srujani2 = fr.load_image_file("Srujani2.jpg")
#image_Srujani3 = fr.load_image_file("Srujani3.jpg")
#image_Srujani4 = fr.load_image_file("Srujani4.jpg")
image_Srujani5 = fr.load_image_file("Srujani5.jpg")

face_encoding_Ashwanth1 = fr.face_encodings(image_Ashwanth1)[0]
face_encoding_Ashwanth2 = fr.face_encodings(image_Ashwanth2)[0]
face_encoding_Ashwanth3 = fr.face_encodings(image_Ashwanth3)[0]
#face_encoding_Ashwanth4 = fr.face_encodings(image_Ashwanth4)[0]
#face_encoding_Ashwanth5 = fr.face_encodings(image_Ashwanth5)[0]
face_encoding_Ashwanth6 = fr.face_encodings(image_Ashwanth6)[0]
face_encoding_Srujani1 = fr.face_encodings(image_Srujani1)[0]
face_encoding_Srujani2 = fr.face_encodings(image_Srujani2)[0]
#face_encoding_Srujani3 = fr.face_encodings(image_Srujani3)[0]
#face_encoding_Srujani4 = fr.face_encodings(image_Srujani4)[0]
face_encoding_Srujani5 = fr.face_encodings(image_Srujani5)[0]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

ash_known_faces = [
    face_encoding_Ashwanth1,
    face_encoding_Ashwanth2,
    face_encoding_Ashwanth3,
#    face_encoding_Ashwanth4,
#    face_encoding_Ashwanth5,
    face_encoding_Ashwanth6
    ]
sru_known_faces = [
    face_encoding_Srujani1,
    face_encoding_Srujani2,
#    face_encoding_Srujani3,
#    face_encoding_Srujani4,
    face_encoding_Srujani5
    ]
# results is an array of True/False telling if the unknown face matched anyone in the known_faces array

while True:
    check, frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0) # 21 is a good number and 0 is the standard deviation
    if(first_frame is None):
        first_frame=gray
        continue

    delta_frame = cv2.absdiff(first_frame,gray)
    thresh_frame=cv2.threshold(delta_frame, 30, 255,cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame,None,iterations=2)
    (_,cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue

        status = 1
        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Only process every other frame of video to save time
        if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
            face_locations = fr.face_locations(small_frame)
            face_encodings = fr.face_encodings(small_frame, face_locations)
            face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matchA = fr.compare_faces(ash_known_faces, face_encoding)
            matchS = fr.compare_faces(sru_known_faces, face_encoding)
            name = "Unknown"
            if matchA[0]:
                name = "Ashwanth"
                #response = myVoiceIt.createEnrollment("Ashwanth", "ashwanth", "Enrollments/Ashwanth-voicePrint.wav", "en-US")
            if matchS[0]:
                name = "Srujani"
            if name == "Unknown":
                message = client.api.account.messages.create(to="+16177174048",
                                            from_="+13393375579",
                                            body="Alert! An unknown person has entered your house")
                                            #media_url=['https://3.bp.blogspot.com/-0YAAfhbWbK8/V85bbvTEv7I/AAAAAAAAAhY/kUK836tzsSw6dXdfUtRgngFWJUB3CqExACEw/s1600/Revitlink%2BDamien%2BWArnings.png'])
            #print(message.sid)
            face_names.append(name)

    process_this_frame = not process_this_frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    status_list.append(status)
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())
    cv2.imshow("Gray Frame",gray)
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("Threshold Frame",thresh_frame)
    cv2.imshow("Color Frame",frame)

    key = cv2.waitKey(1)
    #print(gray)
    #print(delta_frame)
    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break
print(status_list)
print(times)

for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

df.to_csv("times.csv")
video.release()
cv2.destroyAllWindows()
