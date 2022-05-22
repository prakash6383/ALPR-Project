import re
import cv2
import csv
import pytesseract
from datetime import datetime

# Reading video
video = cv2.VideoCapture('sample1.webm')

# tesseract software location
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# trained dataset haar cascade
nPlateCascade_file = 'Haarcascade/haarcascade_russian_plate_number.xml'
nPlateCascade_tracker = cv2.CascadeClassifier(nPlateCascade_file)

# storing data in csv sheet
def storedata(data):
    file = open("output/data.csv")
    reader = csv.reader(file)
    lines = len(list(reader))

    count = lines

    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    date = now.strftime('%H:%M:%S')

    with open('output/data.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow([str(count), data, time,  date])
        count += 1


# Getting video in frames
while True:
    read_successful, frame = video.read()
    video_crop = frame[1700:2160, 0:600]
    # video_crop = frame[1700:2160, 0:600]
    
    if read_successful:
        grayscaled_frame = cv2.cvtColor(video_crop, cv2.COLOR_BGR2GRAY)
    else:
        break

    nPlateCascade = nPlateCascade_tracker.detectMultiScale(grayscaled_frame)

    # print(nPlateCascade)
    for (x, y, w, h) in nPlateCascade:
        cv2.rectangle(video_crop, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # croped plate to tesseract
        crop = video_crop[y + 1:y + h, x + 1:x + w]

        # Removing the noice from the plate
        resize = cv2.resize(crop, None, fx=3, fy=3,
                            interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(resize, (5, 5), 0)
        gray = cv2.medianBlur(blur, 3)

        # coverting image to text in tesseract
        data = (pytesseract.image_to_string(gray))

        result = re.sub('[\W_]+', '', data)

        if not result:
            continue
        elif len(result) != 7:
            continue

        print(result)

        storedata(result)

        # showing the text in video
        cv2.putText(video_crop, result, (x, y + h + 25),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1)

    cv2.imshow('Output', video_crop)
    cv2.waitKey(1)
