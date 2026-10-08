import cv2

facecas = cv2.CascadeClassifier ("haarcascade_frontalface_alt.xml")
capture= cv2.VideoCapture(0)
while True:
    ret,frame=capture.read()
    gray = cv2.cvtColor (frame,cv2.COLOR_BGR2GRAY)
    if not ret:
        print ("Could not read frame")
        break

    face= facecas.detectMultiScale (gray, 1.1, 5)

    _, thresh= cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    contours,hierarchy= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours (frame, contours, -1, (0,0,0), -1)#putting -1 in thickness removes background
    

    for (x,y,w,h) in face:
        cv2.rectangle (frame, (x, y), (x+w, y+h), (120,0,150), 2)
        cv2.putText (frame, "Face detected", (x,y-2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("Face detedction",frame)

    if (cv2.waitKey(1) & 0xFF) ==ord('q'): # & 0xFF is the bit wise and operator which corrects the value from linux or other sources 
       print ("Quitting...")
       break

capture.release()
cv2.destroyAllWindows()