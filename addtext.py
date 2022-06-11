import cv2
import pickle
img = cv2.imread('C:/Users/dell/black/download.jpg',cv2.IMREAD_UNCHANGED)
name = input(">")
cv2.putText(img,name,(30,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),5)

cv2.imshow('image',img)
status = cv2.imwrite('black/images/name.png',img)
 
print("Image written to file-system : ",status)
cv2.waitKey(0)
