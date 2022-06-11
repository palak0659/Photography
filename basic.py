from flask import Flask, flash, request, redirect, url_for, render_template
import pickle
import cv2
import os
import urllib.request
import os
from werkzeug.utils import secure_filename
 
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__ )

UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/color')
def color():
    return render_template('color.html')

@app.route('/resize')
def resize():
    return render_template('resize.html')

@app.route('/add')
def add():
    return render_template('add.html')


@app.route('/add', methods = ['POST'])
def addtext():
    text = request.form['text']
    processed_text = text.upper()
    img = cv2.imread('static/uploads/download.jpg',cv2.IMREAD_UNCHANGED)
    cv2.putText(img,processed_text,(30,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),5)
    cv2.imshow('image',img)
    status = cv2.imwrite('static/uploads/addtext.png',img)
    print("Image written to file-system : ",status)
    cv2.waitKey(0)
    return render_template('add.html')

@app.route('/python')
def python():
    img = cv2.imread("Image/ball.jpg")
    img  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    backup= img.copy()
    for i in range(len(img[:,0,0])):
        for j in range(len(img[0,:,0])):
            R =int(img[i,j,0])
            G =int(img[i,j,1])
            B =int(img[i,j,2])

            sum_col = R+G+B

            if (sum_col >180) & (R>200) & (G>200) & (B>200):
                img[i,j,0] = img[i-1,j-1,0]
                img[i,j,1] = img[i-1,j-1,1]
                img[i,j,2] = img[i-1,j-1,2]


    for i in os.listdir('Image'):
        image = cv2.imread("Image/"+ i)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        lower = [np.mean(image[:,:,i] - np.std(image[:,:,i])/3 ) for i in range(3)]
        upper = [250, 250, 250]

        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        ret,thresh = cv2.threshold(mask, 40, 255, 0)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


        if len(contours) != 0:
            cv2.drawContours(output, contours, -1, 255, 3)

        
            c = max(contours, key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)

       
            cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),5)

        foreground = image[y:y+h,x:x+w]

        cv2.imshow('image',foreground)
        status = cv2.imwrite('static/uploads/resize.png',foreground)
    
        print("Image written to file-system : ",status)
        cv2.waitKey(0)

    return render_template('resize.html')

   

@app.route('/color', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('picture.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/colorful')
def colorful():
    file = open(r'C:/Users/dell/Colorization/colorization.py', 'r').read()
    exec(file)
    pic = cv2.imread('static/uploads/color.png',cv2.IMREAD_UNCHANGED)
    cv2.imshow('image',pic)
    cv2.waitKey(0)
    return render_template('color.html')


if __name__ == "__main__":
    app.run(debug=True)

