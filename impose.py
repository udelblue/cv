import numpy as np
import cv2

kernelSize=21   # Kernel Bluring size 

# Edge Detection Parameter
parameter1=10
parameter2=40
intApertureSize=1

cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame1 = cap.read()    

    # Our operations on the frame come here
    frame = cv2.GaussianBlur(frame1, (kernelSize,kernelSize), 0, 0) 
    edge = cv2.Canny(frame,parameter1,parameter2,intApertureSize)  # Canny edge detection
    mask_edge = cv2.bitwise_not(edge)
    frame = cv2.bitwise_and(frame1,frame1,mask = mask_edge)   
    
    # Display the resulting frame
    cv2.imshow('Super Impose',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()