import numpy as np
import cv2

template_img=cv2.imread("C:/Users/khush/OneDrive/Desktop/templateimage.jpg")
test_img=cv2.imread("C:/Users/khush/OneDrive/Desktop/testimage.jpg")

gray=cv2.cvtColor(template_img,cv2.COLOR_BGR2GRAY)
Gray=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(Gray,127,255,cv2.THRESH_BINARY)
contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(Gray,contours,-1,(0,255,0),2)

def rotate_image(image, angle):
    (h,w) = image.shape[:2]
    center = (w//2,h//2)
    R=cv2.getRotationMatrix2D(center,angle,1.0)
    rotated_image=cv2.warpAffine(image,R,(w,h))
    return rotated_image

rotate_steps=36
step_angle=round((360/rotate_steps),0)

image_array= np.zeros((gray.shape[0],gray.shape[1],rotate_steps),dtype='uint8')
angles=[]
rotation_Angle=0
while rotation_Angle<360:
    angles.append(rotation_Angle)
    rotated_image=rotate_image(gray.copy(),rotation_Angle)
    image_array[:,:,len(angles)-1]=rotated_image
    rotation_Angle+=step_angle

for i in range(len(angles)):
    res=cv2.matchTemplate(Gray,image_array[:,:,i],cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    

cv2.imshow('image',Gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
