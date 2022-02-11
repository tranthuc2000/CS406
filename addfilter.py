import numpy as np
import cv2
import dlib
import math
hog_face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
imleft=cv2.imread('dogleft.jpg')
imright=cv2.imread('dogright1.jpg')
imnoise=cv2.imread('dognoise.jpg')
def rotate_image(mat, angle): #rotate imge
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h),borderValue=(255,255,255))
    return rotated_mat
def findlm(img): # find landmark point
    im_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    faces=hog_face_detector(im_gray)
    mypoints=[]
    for face in faces:
        x=face.left()
        y=face.top()
        w=face.right()-x
        h=face.bottom() - y
        #cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        landmarks=predictor(im_gray,face)
        mypoint=[]
        for n in range(68):
            x1=landmarks.part(n).x
            y1=landmarks.part(n).y
            #cv2.putText(im,str(n),(x1,y1-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(0,0,255),1)
            #cv2.circle(im,(x1,y1),2,(25,25,255),cv2.FILLED)
            mypoint.append([x1,y1])
        mypoint=np.array(mypoint)
        mypoints.append(mypoint)
    return mypoints
def putthug(im,mypoints): # add glass thuglife
    im_glass=cv2.imread('glassw.jpg')
    im_glass_gray=cv2.cvtColor(im_glass,cv2.COLOR_RGB2GRAY)
    for mypoint in mypoints:
        n=math.degrees(math.atan((mypoint[27][0]-mypoint[33][0])/(mypoint[27][1]-mypoint[33][1])))
        left = int(abs((mypoint[0][0]+mypoint[36][0])/2))
        right = abs((mypoint[45][0]+mypoint[16][0])/2)
        weye = int(right-left)
        we=weye
        im_glass_r=rotate_image(im_glass,n)
        im_glass_gray=cv2.cvtColor(im_glass_r,cv2.COLOR_RGB2GRAY)
        height_bc, width_bc, dif_bc =im_glass_r.shape
        newh=int(weye/width_bc*height_bc*1.2)
        he=newh
        top=int(mypoint[27][1]-newh//2)
        left=int(mypoint[27][0]-weye//2)
        if top <0:
            newh +=top
            top =0
        if left+weye > im.shape[1]:
            weye=im.shape[1] - left
        if top+newh >im.shape[0]:
            newh =im.shape[0]-top
        if left < 0:
            weye= weye+left
            left = 0
        if newh <0 or weye < 0 :
            return im
        im_glass_rsx=cv2.resize(im_glass_gray,(we,he))  
        im_glass_rsx=im_glass_rsx[he-newh:he,we-weye:we]
        ret,img_bcb=cv2.threshold(im_glass_rsx,100,255,cv2.THRESH_BINARY_INV)
        im_face=im[top:top+newh,left:left+weye]
        img_glass_rs=cv2.resize(im_glass_r,(we,he))
        img_glass_rs=img_glass_rs[he-newh:he,we-weye:we]
        img_bcb_inv=cv2.bitwise_not(img_bcb)
        img_fg=cv2.bitwise_and(im_face,im_face,mask=img_bcb_inv)
        im_glass_rsx=cv2.resize(im_glass,(we,he))
        img_bg=cv2.bitwise_and(img_glass_rs,img_glass_rs,mask=img_bcb)
        rs=cv2.add(img_fg,img_bg)
        im[top:top+newh,left:left+weye]=rs
    return im
def putglassblue(im,mypoints): # add blue glass
    im_glass=cv2.imread('glassww.jpg')
    im_glass_gray=cv2.cvtColor(im_glass,cv2.COLOR_RGB2GRAY)
    for mypoint in mypoints:
        n=math.degrees(math.atan((mypoint[27][0]-mypoint[33][0])/(mypoint[27][1]-mypoint[33][1])))
        left = int(abs((mypoint[0][0]+mypoint[36][0])/2))
        right = abs((mypoint[45][0]+mypoint[16][0])/2)
        weye = int((right-left)*1.2)
        im_glass_r=rotate_image(im_glass,n)
        im_glass_gray=cv2.cvtColor(im_glass_r,cv2.COLOR_RGB2GRAY)
        height_bc, width_bc, dif_bc =im_glass_r.shape
        newh=int(weye/width_bc*height_bc)
        we=weye
        he=newh
        top=int(mypoint[27][1]-newh//2)
        left=int(mypoint[27][0]-weye//2)
        if top <0:
            newh +=top
            top =0
        if left+weye > im.shape[1]:
            weye=im.shape[1] - left
        if top+newh >im.shape[0]:
            newh =im.shape[0]-top
        if left < 0:
            weye= weye+left
            left = 0
        if newh <0 or weye < 0 :
            return im
        im_glass_rsx=cv2.resize(im_glass_gray,(we,he))  
        im_glass_rsx=im_glass_rsx[he-newh:he,we-weye:we]
        ret,img_bcb=cv2.threshold(im_glass_rsx,240,255,cv2.THRESH_BINARY_INV)
        im_face=im[top:top+newh,left:left+weye]
        img_glass_rs=cv2.resize(im_glass_r,(we,he))
        img_glass_rs=img_glass_rs[he-newh:he,we-weye:we]
        img_bcb_inv=cv2.bitwise_not(img_bcb)
        img_fg=cv2.bitwise_and(im_face,im_face,mask=img_bcb_inv)
        im_glass_rsx=cv2.resize(im_glass,(we,he))
        img_bg=cv2.bitwise_and(img_glass_rs,img_glass_rs,mask=img_bcb)
        rs=cv2.add(img_fg,img_bg)
        im[top:top+newh,left:left+weye]=rs
    return im
def putnobita(im,mypoints): # add nobita glass
    im_glass=cv2.imread('glassmu.jpg')
    im_glass_gray=cv2.cvtColor(im_glass,cv2.COLOR_RGB2GRAY)

    for mypoint in mypoints:
       
        n=math.degrees(math.atan((mypoint[27][0]-mypoint[33][0])/(mypoint[27][1]-mypoint[33][1])))
        left = int(abs((mypoint[0][0]+mypoint[36][0])/2))
        right = abs((mypoint[45][0]+mypoint[16][0])/2)
        weye = int((right-left)*1.2)
        im_glass_r=rotate_image(im_glass,n)
        im_glass_gray=cv2.cvtColor(im_glass_r,cv2.COLOR_RGB2GRAY)
        height_bc, width_bc, dif_bc =im_glass_r.shape
        newh=int(weye/width_bc*height_bc)
        top1=int(mypoint[37][1])
        bottom=mypoint[41][1]
        
        heye=abs(int(4*(top1-bottom))) 
        he=newh
        we=weye 
        top=int(mypoint[27][1]-newh//2)
        left=int(mypoint[27][0]-weye//2)
        if top <0:
            newh +=top
            top =0
        if left+weye > im.shape[1]:
            weye=im.shape[1] - left
        if top+newh >im.shape[0]:
            newh =im.shape[0]-top
        if left < 0:
            weye= weye+left
            left = 0
        if newh <0 or weye < 0 :
            return im
        im_glass_rsx=cv2.resize(im_glass_gray,(we,he))  
        im_glass_rsx=im_glass_rsx[he-newh:he,we-weye:we]
        ret,img_bcb=cv2.threshold(im_glass_rsx,100,255,cv2.THRESH_BINARY_INV)
        im_face=im[top:top+newh,left:left+weye]
        img_glass_rs=cv2.resize(im_glass_r,(we,he))
        img_glass_rs=img_glass_rs[he-newh:he,we-weye:we]
        img_bcb_inv=cv2.bitwise_not(img_bcb)
        img_fg=cv2.bitwise_and(im_face,im_face,mask=img_bcb_inv)
        im_glass_rsx=cv2.resize(im_glass,(we,he))
        img_bg=cv2.bitwise_and(img_glass_rs,img_glass_rs,mask=img_bcb)
        rs=cv2.add(img_fg,img_bg)
        im[top:top+newh,left:left+weye]=rs
    return im
def addleft(im,im_glass,mypoint,n):
    w1=mypoint[22][0]-mypoint[17][0]
    hh=mypoint[41][1]-mypoint[19][1]
    w = w1
    h=w1
    x =mypoint[0][0] - h//3
    y =mypoint[19][1] - hh - w//2
    if y <0:
        h +=y
        y =0
    if x+w > im.shape[1]:
        w=im.shape[1] - x
    if y+h >im.shape[0]:
        h =im.shape[0]-y
    if x < 0:
        w= w+x
        x = 0
    if h <0 or w < 0:
        return im
    im_glass_r=rotate_image(im_glass,n)
    im_glass_gray=cv2.cvtColor(im_glass_r,cv2.COLOR_RGB2GRAY)
    im_glass_rsx=cv2.resize(im_glass_gray,(w1,w1))
    im_glass_rsx=im_glass_rsx[w1-h:w1,w1-w:w1]
    ret,img_bcb=cv2.threshold(im_glass_rsx,200,255,cv2.THRESH_BINARY_INV)
    im_face=im[y:y+h,x:x+w]
    img_glass_rs=cv2.resize(im_glass_r,(w1,w1))
    img_glass_rs=img_glass_rs[w1-h:w1,w1-w:w1]
    img_bcb_inv=cv2.bitwise_not(img_bcb)
    img_fg=cv2.bitwise_and(im_face,im_face,mask=img_bcb_inv)
    img_bg=cv2.bitwise_and(img_glass_rs,img_glass_rs,mask=img_bcb)
    rs=cv2.add(img_fg,img_bg)
    im[y:y+h,x:x+w]=rs
    return im
def addright(im,im_glass,mypoint,n): # add dog right
    w1=mypoint[22][0]-mypoint[17][0]
    hh=mypoint[46][1]-mypoint[24][1]
    w = w1
    h=w1
    x =mypoint[24][0]
    y =mypoint[24][1]-hh - w//2
    if y <0:
        h +=y
        y =0
    if x+w > im.shape[1]:
        w=im.shape[1] - x
    if y+h >im.shape[0]:
        h =im.shape[0]-y
    if x < 0:
        w= w+x
        x = 0
    if h <0 or w < 0:
        return im
    im_glass_r=rotate_image(im_glass,n)
    
    im_glass_gray=cv2.cvtColor(im_glass_r,cv2.COLOR_RGB2GRAY)
    im_glass_rsx=cv2.resize(im_glass_gray,(w1,w1)) 
    im_glass_rsx=im_glass_rsx[w1-h:w1,0:(w1-(w1-w))]

    ret,img_bcb=cv2.threshold(im_glass_rsx,200,255,cv2.THRESH_BINARY_INV)
    im_face=im[y:y+h,x:x+w]
    img_glass_rs=cv2.resize(im_glass_r,(w1,w1))
    img_glass_rs=img_glass_rs[w1-h:w1,0:(w1-(w1-w))]
    img_bcb_inv=cv2.bitwise_not(img_bcb)
    img_fg=cv2.bitwise_and(im_face,im_face,mask=img_bcb_inv)
    img_bg=cv2.bitwise_and(img_glass_rs,img_glass_rs,mask=img_bcb)
    rs=cv2.add(img_fg,img_bg)
    im[y:y+h,x:x+w]=rs
    return im
def addnoice(im,mypoint,im_glass,n): # add dog noice
    w=int(2.5*(mypoint[35][0]-mypoint[31][0]))
    h=int(5/6*w)
    center=[(mypoint[35][0]+mypoint[31][0])//2,(mypoint[33][1]+mypoint[30][1])//2]
    x =int(center[0]-w//2)
    y =int(center[1]-h*2//3)
    im_glass_r=rotate_image(im_glass,n)
    im_glass_gray=cv2.cvtColor(im_glass_r,cv2.COLOR_RGB2GRAY)
    height_bc, width_bc, dif_bc =im_glass_r.shape
    im_glass_rsx=cv2.resize(im_glass_gray,(w,h))  
    ret,img_bcb=cv2.threshold(im_glass_rsx,200,255,cv2.THRESH_BINARY_INV)
    im_face=im[y:y+h,x:x+w]
    img_glass_rs=cv2.resize(im_glass_r,(w,h))
    img_bcb_inv=cv2.bitwise_not(img_bcb)
    img_fg=cv2.bitwise_and(im_face,im_face,mask=img_bcb_inv)
    im_glass_rsx=cv2.resize(im_glass,(w,h))
    img_bg=cv2.bitwise_and(img_glass_rs,img_glass_rs,mask=img_bcb)
    rs=cv2.add(img_fg,img_bg)
    im[y:y+h,x:x+w]=rs
    return im

def adddog(im,mypoints): 
    for mypoint in mypoints:
        n=math.degrees(math.atan((mypoint[27][0]-mypoint[33][0])/(mypoint[27][1]-mypoint[33][1])))
        im=addleft(im,imleft,mypoint,n) 
        im=addright(im,imright,mypoint,n)
        im=addnoice(im,mypoint,imnoise,n)
    return im
 
img=cv2.imread('img6.jpg') # change link image here
mypoints=findlm(img)
img=adddog(img,mypoints) # add filter here
img=putthug(img,mypoints)
img=putnobita(img,mypoints)
img=putglassblue(img,mypoints)
cv2.imshow('final',img) # show here
cv2.waitKey(0)
cv2.destroyAllWindows()
