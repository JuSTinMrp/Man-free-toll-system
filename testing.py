import cv2
import pytesseract
import pandas as pd
import os
import sys
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
from PIL import Image
import time

camera = cv2.VideoCapture(0)
#camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

df = pd.DataFrame(columns=['License Plate'])        #pandas dataframe creation

#print(df)

while True:
    ret, frame = camera.read()   #ret, frame = 
    #frame = cv2.imread("plating.jpg")
    cv2.imshow("frame", frame)

    #frame = frame.resize((500, 500))
    #cv2.waitKey(0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    cv2.imshow("2 - Bilateral Filter", gray)

    edged = cv2.Canny(gray, 170, 200)
    cv2.imshow("4 - Canny Edges", edged)

    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)  

    for (x, y, w, h) in plates:
        cv2.rectangle(edged, (x, y), (x + w, y + h), (0, 255, 0), 2)

        plate_img = gray[y:y + h, x:x + w]

        _, plate_img = cv2.threshold(plate_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  #otsu thresholding method automatically
                                                                                              #finds the threshold value T

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))     #rectangular structuring element and size of the structuring element
        plate_img = cv2.dilate(plate_img, kernel, iterations=1)

        number_plate = pytesseract.image_to_string(plate_img, lang='eng', config='--oem 3 --psm 7')  #page segmentation mode = 11 char 
                                                                                              #including space

        number_plate = number_plate.strip().replace(' ', '').replace('\n', '').replace('\r', '').replace('\t','').replace('|', '').replace('=', '').replace('-', '')

        
        number_plate="KL48L7987"
        raw_data = {'date': [time.asctime( time.localtime(time.time()) )], 'vehicle_number': [number_plate], 'way' : ["IN"]}

        df = pd.DataFrame(raw_data, columns = ['date', 'way'])
        df.to_csv('data.csv')



        
        if number_plate != '':
            df = df.append({'License Plate': number_plate}, ignore_index=True)

            cv2.imwrite(os.path.join('images', number_plate + '.jpg'), plate_img)

            
            print(number_plate)
            #cv2.imshow('plate image', plate_img)
            #cv2.imshow('final plate', number_plate)

            

            break
        

           

    #cv2.imshow('Frame', frame)
    #cv2.imshow('Gray', gray)
    


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

print(df)



#try adaptive threshold method also finally....

#config='--oem 3 --psm 6 -c tessedit_char_whitelist = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')