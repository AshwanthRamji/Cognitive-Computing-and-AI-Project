#Import Files
from __future__ import print_function
from apiclient.discovery import build
from apiclient.http import MediaFileUpload
from httplib2 import Http
from oauth2client import file, client, tools
import cv2, time, pandas
from datetime import datetime
import face_recognition as fr
from twilio.rest import Client
from VoiceIt import *
import pyaudio
import tkinter as Tk
import pyttsx
import wave

##VoiceIt developer ID
myVoiceIt = VoiceIt("")
# Your Account SID from twilio.com/console
account_sid = ""
# Your Auth Token from twilio.com/console
auth_token  = ""
clients = Client(account_sid, auth_token)
first_frame =None
status_list=[None,None]
times=[]

##Dataframe recording from start to the end of video
df= pandas.DataFrame(columns=["Start","End"])
video = cv2.VideoCapture(0)
engine = pyttsx.init()

# Load a sample picture and learn how to recognize it.
#Person 1
image_Ashwanth1 = fr.load_image_file("Ashwanth1.jpg")
image_Ashwanth2 = fr.load_image_file("Ashwanth2.jpg")
image_Ashwanth3 = fr.load_image_file("Ashwanth3.jpg")
image_Ashwanth4 = fr.load_image_file("Ashwanth3-1.jpg")
image_Ashwanth5 = fr.load_image_file("Ashwanth5.jpg")
image_Ashwanth6 = fr.load_image_file("Ashwanth6.jpg")


##Encoding face from sample image - Ashwanth
face_encoding_Ashwanth1 = fr.face_encodings(image_Ashwanth1)[0]
face_encoding_Ashwanth2 = fr.face_encodings(image_Ashwanth2)[0]
face_encoding_Ashwanth3 = fr.face_encodings(image_Ashwanth3)[0]
face_encoding_Ashwanth4 = fr.face_encodings(image_Ashwanth4)[0]
face_encoding_Ashwanth5 = fr.face_encodings(image_Ashwanth5)[0]
face_encoding_Ashwanth6 = fr.face_encodings(image_Ashwanth6)[0]

# Initialize some variables - face recognition
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

## Results is an array of True/False telling if the unknown face matched anyone in the known_faces array
ash_known_faces = [
    face_encoding_Ashwanth1,
    face_encoding_Ashwanth2,
    face_encoding_Ashwanth3,
    face_encoding_Ashwanth4,
    face_encoding_Ashwanth5,
    face_encoding_Ashwanth6
    ]

## Recorder to record voice and converting to file
class Recorder(object):
    '''A recorder class for recording audio to a WAV file.
    Records in mono by default.
    '''

    def __init__(self, channels=1, rate=44100, frames_per_buffer=1024):
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer

    def open(self, fname, mode='wb'):
        return RecordingFile(fname, mode, self.channels, self.rate,
                            self.frames_per_buffer)

class RecordingFile(object):
    def __init__(self, fname, mode, channels,
                rate, frames_per_buffer):
        self.fname = fname
        self.mode = mode
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self._pa = pyaudio.PyAudio()
        self.wavefile = self._prepare_file(self.fname, self.mode)
        self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, exception, value, traceback):
        self.close()

    def record(self, duration):
        # Use a stream with no callback function in blocking mode
        self._stream = self._pa.open(format=pyaudio.paInt16,
                                        channels=self.channels,
                                        rate=self.rate,
                                        input=True,
                                        frames_per_buffer=self.frames_per_buffer)
        for _ in range(int(self.rate / self.frames_per_buffer * duration)):
            audio = self._stream.read(self.frames_per_buffer)
            self.wavefile.writeframes(audio)
        return None

    def start_recording(self):
        # Use a stream with a callback in non-blocking mode
        self._stream = self._pa.open(format=pyaudio.paInt16,
                                        channels=self.channels,
                                        rate=self.rate,
                                        input=True,
                                        frames_per_buffer=self.frames_per_buffer,
                                        stream_callback=self.get_callback())
        self._stream.start_stream()
        return self

    def stop_recording(self):
        self._stream.stop_stream()
        return self

    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            self.wavefile.writeframes(in_data)
            return in_data, pyaudio.paContinue
        return callback


    def close(self):
        self._stream.close()
        self._pa.terminate()
        self.wavefile.close()

    def _prepare_file(self, fname, mode='wb'):
        wavefile = wave.open(fname, mode)
        wavefile.setnchannels(self.channels)
        wavefile.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
        wavefile.setframerate(self.rate)
        return wavefile

## Text-to-speech engine function
def voicephrase():
    engine.say('Say the Catch phrase NEVER FORGET TOMORROW IS A NEW DAY in ')
    engine.say('3')
    engine.say('2')
    engine.say('1')
    engine.runAndWait()
    ## Save catch phrase in .wav file
    print ('Say the catch phrase')
    rec = Recorder(channels=2)
    with rec.open('Ashwanth-voicePrint.wav', 'wb') as recfile:
        recfile.record(duration=5.0)

## Text to speech engine function
engine.say('Please show your face infront of the camera in')
engine.say('3')
engine.say('2')
engine.say('1')
engine.runAndWait()


run_once = 0
show_once = 0
count = 0
#Motion detection runs in the background
while True:
    # storing video feed in variables
    check, frame = video.read()
    status = 0

    #Frame properties
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
                name = "Unknown"
                if matchA[0] and name == "Unknown":
                    if show_once == 0:
                        engine.say('Please show only one face infront of the camera')
                        engine.runAndWait()
                        show_once = 1
            # If face matches known face - Ashwanth
                if matchA[0]:
                    count +=1
                    print(count)
                    #prints name around face as Ashwanth
                    name = "Ashwanth"
                    if count > 20:
                        # function call to convert voice recording to wav file
                        voicephrase()

                        # Response sent to VoiceIt api with credentials to authenticate voice
                        response = myVoiceIt.authentication("Ashwanth", "ashwanth", "Ashwanth-voicePrint.wav", "en-US")
                        print(response)

                        #If response fails once
                        if 'failed' in response:

                            # Setting number of attempts to 2
                            number_of_guesses = 0
                            while number_of_guesses < 2:
                                #Repeats voice authentication
                                engine.say('Authentication Failed. Voice Not Detected. Please say the catch phrase again')
                                engine.runAndWait()
                                voicephrase()
                                response = myVoiceIt.authentication("Ashwanth", "ashwanth", "Ashwanth-voicePrint.wav", "en-US")
                                print(response)
                                #Increase attempt by 1
                                number_of_guesses = number_of_guesses + 1

                            # Success after a failure attempt
                                if 'success' in response:
                                    engine.say('Authentication Successful. Voice Detected. You can enter the house.')
                                    engine.runAndWait()

                                    #Repeats process
                                    response = myVoiceIt.authentication("Ashwanth", "ashwanth", "Ashwanth-voicePrint.wav", "en-US")
                                    print(response)
                                    print('Authentication Successful. Voice Detected. You can enter the house.')
                                    exit()

                        # If authentication fails three times
                            engine.say('Authentication Failed three times. You cannot enter the house.')
                            engine.runAndWait()

                        # Calls Twilio API to send a text message to linked phone number
                            message = clients.api.account.messages.create(to="+16177174048",
                                                    from_="+16179172104",
                                                    body="Alert! Voice authentication failed. An unknown person is trying to enter the building.")
                            print('Authentication Failed. Voice Not Detected.')
                            exit()

                    # If success in attempt
                        if 'success' in response:
                            engine.say('Authentication Successful. Voice Detected. You can enter the house.')
                            engine.runAndWait()
                            print(response)
                            print('Authentication Successful. Voice Detected. You can enter the house.')

                        #After message loop exits
                            exit()
                ## If the face is unknown

                if name == "Unknown":
                    if run_once == 0:
                        print('Unknown')
                        engine.say('Unknown face detected. Please move away from the camera')
                        engine.runAndWait()
                        # Sends text message to phone number
                        message = clients.api.account.messages.create(to="+16177174048",
                                                from_="+16179172104",
                                                body="Alert! An unknown person has failed face recognition")
                        run_once = 1
                    count = 0
                    print(count)

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

        status_list=status_list[-2:]

        if status_list[-1]==1 and status_list[-2]==0:
            times.append(datetime.now())
        if status_list[-1]==0 and status_list[-2]==1:
            times.append(datetime.now())

    cv2.imshow("Color Frame",frame)
    key = cv2.waitKey(1)
    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break
print(status_list)
print(times)

for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

# Writes recorded time and motion to csv file
df.to_csv("times.csv")
video.release()
cv2.destroyAllWindows()
