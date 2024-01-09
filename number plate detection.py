import cv2 as cv
import numpy as np

plate_cascade = cv.CascadeClassifier("russian.xml")



cap =cv.VideoCapture(0)
cap.set(10,200)
count = 0
while True:
    ret, frame= cap.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    det = plate_cascade.detectMultiScale(gray, 1.4, 12)
    for (x,y,w,h) in det:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv.putText(frame, "Number plate", (x,y), cv.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)
        imgRoi = frame[y:y+h,x:x+w]
        cv.imshow("ROI", imgRoi)

    cv.imshow("Result", frame)
   
    
    if cv.waitKey(1) & 0xFF==ord('d'):
        cv.imwrite("images/Scanned/NoPlate_"+str(count)+".jpg",imgRoi)
        cv.rectangle(frame,(0,200),(640,300),(0,255,0),cv.FILLED)
        cv.putText(frame,"Scan Saved",(150,265),cv.FONT_HERSHEY_DUPLEX,
                    2,(0,0,255),2)
        cv.waitKey(500)
        count +=1
        break 
cv.destroyAllWindows()
cap.release()

