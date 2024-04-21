from typing import Counter
import cv2
import os
from flask import Flask, request, render_template, redirect, session, url_for
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import mysql.connector
import plotly
import plotly.graph_objs as go

DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = ''
DB_NAME = 'faceattendance'

# VARIABLES
MESSAGE = "WELCOME  " \
          " Instruction: to register your attendance kindly click on 'a' on keyboard"

#### Defining Flask App
app = Flask(__name__)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Gender,Branch,Time')


#### get a number of total registered users
def totalreg():
    # return len(os.listdir('static/faces'))
    connection = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
    cursor = connection.cursor()

    cursor.execute("SELECT COUNT(DISTINCT roll) FROM students")
    count = cursor.fetchone()[0]  
    return count


#### extract the face from an image
def extract_faces(img):
    if img != []:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []


#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []

    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# to fetch the last row
def extract_last_row():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    last_row = df.tail(1)
    return last_row

#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    genders = df['Gender']  # Added
    branches = df['Branch']  # Added
    times = df['Time']
    l = len(df)
    return names, rolls, genders, branches, times, l


#### Add Attendance of a specific user
def add_attendance(name, gender, branch):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    print(gender)
    gender = "Female"
    branch = "MCA"
    current_time = datetime.now().strftime("%H:%M:%S")
    connection = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
    cursor = connection.cursor()
    query = "INSERT INTO students (name, roll, gender, branch, time) VALUES (%s, %s, %s, %s, %s)"
    time = str(get_current_datetime())
    values = (username, userid, gender, branch, time)
    cursor.execute(query, values)
    connection.commit()
        
    cursor.close()
    connection.close()
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if str(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{gender},{branch},{current_time}')
    else:
        print("This user has already marked attendance for the day, but still marking it.")


################## ROUTING FUNCTIONS ##############################

#### Our main page
@app.route('/')
def home():
    names, rolls, genders, branches, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, genders=genders, branches=branches, times=times,
                           l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=MESSAGE)

def get_current_datetime():
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return current_datetime


@app.route('/contact')
def contact():
    
    return render_template('contact.html')

    # return re


@app.route('/liststudents')
def liststudents():
    return render_template('liststudents.html')



@app.route('/list_students')
def list_students():
    connection = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
    cursor = connection.cursor()

    cursor.execute("SELECT name, gender, branch, roll FROM students GROUP BY roll")
    data = cursor.fetchall()

    cursor.close()
    connection.close()

    return render_template('liststudents.html', table_data=data)




@app.route('/attendance')
def show_attendance():
    connection = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
    cursor = connection.cursor()

    cursor.execute("SELECT name, COUNT(*) as attendance_count FROM students GROUP BY name")
    data = cursor.fetchall()

    cursor.close()
    connection.close()

    names = [row[0] for row in data]
    attendance_count = [row[1] for row in data]
    bar_chart = create_graph(names, attendance_count)
    return render_template('attendance.html', table_data=data, plot=bar_chart)


def create_graph(names, attendance_count):
    data = [go.Bar(x=names, y=attendance_count)]
    layout = go.Layout(title='Overall Attendance')
    fig = go.Figure(data=data, layout=layout)
    graph = plotly.offline.plot(fig, output_type='div', include_plotlyjs=False)
    return graph

@app.route('/view_student/<int:roll>')
def view_student(roll):
 
    connection = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
    cursor = connection.cursor()

    cursor.execute("SELECT time FROM students WHERE roll = %s", (roll,))
    attendance_data = cursor.fetchall()

    cursor.close()
    connection.close()

    attendance_timestamps = [row[0] for row in attendance_data]

    attendance_days = [timestamp.date() for timestamp in attendance_timestamps]

    attendance_counts = Counter(attendance_days)

    x_data = list(attendance_counts.keys())
    y_data = list(attendance_counts.values())

    trace = go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines+markers',
        name='Attendance',
        marker=dict(color='blue'),
    )

    layout = go.Layout(
        title='Attendance Time Series',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Attendance Count'),
    )

    fig = go.Figure(data=[trace], layout=layout)

    graph_json = fig.to_json()

    return render_template('view_student.html', graph_json=graph_json)








@app.route('/start', methods=['GET'])
def start():
    ATTENDANCE_MARKED = False
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        names, rolls, genders, branches, times, l = extract_attendance()
        MESSAGE = 'This face is not registered with us, kindly register yourself first'
        print("Face not in the database, need to register")
        return render_template('home.html', names=names, rolls=rolls, genders=genders, branches=branches, times=times,
                               l=l, totalreg=totalreg, datetoday2=datetoday2, mess=MESSAGE)

    cap = cv2.VideoCapture(0)
    ret = True
    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            cv2.putText(frame, f'{identified_person}', (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
            if cv2.waitKey(1) == ord('a'):
                add_attendance(identified_person, 'Unknown', 'Unknown')  
                current_time_ = datetime.now().strftime("%H:%M:%S")
                print(f"Attendance marked for {identified_person}, at {current_time_} ")
                ATTENDANCE_MARKED = True
                break
        if ATTENDANCE_MARKED:
            break

        cv2.imshow('Attendance Check, press "q" to exit', frame)
        cv2.putText(frame, 'hello', (30, 30), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255))

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, genders, branches, times, l = extract_attendance()
    MESSAGE = 'Attendance taken successfully'
    last_row = extract_last_row()
    print(str(last_row))
    print("Attendance registered")
    return render_template('home.html', names=names, rolls=rolls, genders=genders, branches=branches, times=times,
                           l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=MESSAGE)




# this function is used to add new user details and caputure images 
# 


@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    gender = request.form['gender'] 
    branch = request.form['branch']  
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
            if j % 10 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if j == 500:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, genders, branches, times, l = extract_attendance()
    if totalreg() > 0:
        names, rolls, genders, branches, times, l = extract_attendance()
        MESSAGE = 'User added Successfully'
        connection = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
        cursor = connection.cursor()
        query = "INSERT INTO students (name, roll, gender, branch, time) VALUES (%s, %s, %s, %s, %s)"
        time = str(get_current_datetime())
        values = (newusername, newuserid, gender, branch, time)
        cursor.execute(query, values)
        connection.commit()
        
        cursor.close()
        connection.close()
        return render_template('home.html', names=names, rolls=rolls, genders=genders, branches=branches, times=times,
                               l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=MESSAGE)
    else:
        return redirect(url_for('home', names=names, rolls=rolls, genders=genders, branches=branches, times=times,
                                l=l, totalreg=totalreg(), datetoday2=datetoday2))


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True, port=1000)
