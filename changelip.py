
import cv2
import numpy as np
import dlib
hog_face_detector = dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
def empty(x):
    pass
cv2.namedWindow("BGR")
cv2.resizeWindow("BGR",500,100)
cv2.createTrackbar("BLUE","BGR",0,255,empty)
cv2.createTrackbar("GREEN","BGR",0,255,empty)
cv2.createTrackbar("RED","BGR",0,255,empty)
cv2.createTrackbar("BETA","BGR",0,10,empty)
cv2.createTrackbar("Kernel Gaussian","BGR",1,10,empty)
def createMask(img, landmarks):
    Points = landmarks
    Mask = np.zeros_like(img)
    cv2.fillConvexPoly(Mask, np.int32(Points), (255, 255, 255))
    Mask = np.uint8(Mask)
    return Mask

def creatEye(img,points,scale=5):
	bbox=cv2.boundingRect(points)
	x,y,w,h=bbox
	imgcrop=img[y:y+h,x:x+w]
	imgcrop=cv2.resize(imgcrop,(0,0),None,scale,scale)
	return imgcrop

def creatlips(img,mask,blue,green,red,K,beta):
    color=np.zeros_like(img)
    color[:]=blue,green,red
    mask=cv2.bitwise_and(mask,color)
    mask=cv2.GaussianBlur(mask,(K,K),cv2.BORDER_DEFAULT)
    
    img=cv2.addWeighted(img,1,mask,beta,0)
    return img

while True:
    img = cv2.imread('MicrosoftTeams-image.png')
    R=cv2.getTrackbarPos("RED","BGR")
    G= cv2.getTrackbarPos("GREEN","BGR")
    B= cv2.getTrackbarPos("BLUE","BGR")
    beta=cv2.getTrackbarPos("BETA","BGR")/10
    K = cv2.getTrackbarPos("Kernel Gaussian","BGR")*2+1
    imgcpy=img.copy()
    imggray = cv2.cvtColor(imgcpy, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(imggray)
    for face in faces:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        landmarks=predictor(imggray,face)
        top1=[]
        top2=[]
        top3=[]
        top4=[]
        top5=[]
        top6=[]
        top7=[]
        top8=[]
        top9=[]
        top10=[]
        mymidpoints=[]
        mymidpoints2=[]
        myleftpoints=[]
        myrightpoints=[]
        myrightpoints2=[]
        rightpoints=[]
        leftpoints=[]
        for n in range(68):
            if (n == 60 or n ==50 or n ==49): 
                x1=landmarks.part(n).x
                y1=landmarks.part(n).y
                top1.append([x1,y1])
            if (n == 60 or n ==50 or n ==62): 
                x1=landmarks.part(n).x
                y1=landmarks.part(n).y
                top2.append([x1,y1])
            if (n == 61 or n ==50 or n ==51): 
                x1=landmarks.part(n).x
                y1=landmarks.part(n).y
                top3.append([x1,y1])
            if (n == 61 or n ==51 or n ==62): 
                x1=landmarks.part(n).x
                y1=landmarks.part(n).y
                top4.append([x1,y1])
            if (n == 62 or n ==52 or n ==51): 
                x1=landmarks.part(n).x
                y1=landmarks.part(n).y
                top5.append([x1,y1])
            if (n == 63 or n ==52 or n ==62): 
                x1=landmarks.part(n).x
                y1=landmarks.part(n).y
                top6.append([x1,y1])
            if (n == 52 or n ==63 or n ==53): 
                x1=landmarks.part(n).x
                y1=landmarks.part(n).y
                top7.append([x1,y1])
            if (n == 53 or n ==64  or n == 63): 
                x1=landmarks.part(n).x
                y1=landmarks.part(n).y
                top8.append([x1,y1])
            if    n==48 or n == 60  or n == 49 : 
                x1=landmarks.part(n).x
                y1=landmarks.part(n).y
                top9.append([x1,y1])
            if n == 54  or n==64 or n == 53 : 
                x1=landmarks.part(n).x
                y1=landmarks.part(n).y
                top10.append([x1,y1])


            if n == 58  or n ==65 or n== 56 or n == 57: 
                x1=landmarks.part(n).x
                y1=landmarks.part(n).y
                mymidpoints.append([x1,y1])
            if n == 58  or n ==65 or n== 67 or n==66 : 
                x1=landmarks.part(n).x
                y1=landmarks.part(n).y
                mymidpoints2.append([x1,y1])
            if n == 67  or n==58 or n == 60  or n == 59 : 
                x1=landmarks.part(n).x
                y1=landmarks.part(n).y
                myleftpoints.append([x1,y1])

            if (n == 55 or n ==65 or n ==64): 
                x1=landmarks.part(n).x
                y1=landmarks.part(n).y
                myrightpoints.append([x1,y1])
            if (n == 55 or n ==56  or n == 65): 
                x1=landmarks.part(n).x
                y1=landmarks.part(n).y
                myrightpoints2.append([x1,y1])
            if    n==48 or n == 60  or n == 59 : 
                x1=landmarks.part(n).x
                y1=landmarks.part(n).y
                leftpoints.append([x1,y1])
            if n == 54  or n==64 or n == 55 : 
                x1=landmarks.part(n).x
                y1=landmarks.part(n).y
                rightpoints.append([x1,y1])
        top11=np.array(top1)
        top21=np.array(top2)
        top31=np.array(top3)
        top41=np.array(top4)
        top51=np.array(top5)
        top61=np.array(top6)
        top71=np.array(top7)
        top81=np.array(top8)
        top91=np.array(top9)
        top101=np.array(top10)
        mymidpoint=np.array(mymidpoints)
        mymidpoint2=np.array(mymidpoints2)
        myrightpoint2=np.array(myrightpoints2)
        myleftpoint=np.array(myleftpoints)
        myrightpoint=np.array(myrightpoints)
        leftpoint=np.array(leftpoints)
        rightpoint=np.array(rightpoints)
        left = createMask(imgcpy,leftpoint[:])
        right = createMask(imgcpy,rightpoint[:])
        top01 = createMask(imgcpy,top11)
        top02 = createMask(imgcpy,top21)
        top03 = createMask(imgcpy,top31)
        top04 = createMask(imgcpy,top41)
        top05 = createMask(imgcpy,top51)
        top06 = createMask(imgcpy,top61)
        top07 = createMask(imgcpy,top71)
        top08 = createMask(imgcpy,top81)
        top09 = createMask(imgcpy,top91)
        top010 = createMask(imgcpy,top101)
        mid = createMask(imgcpy,mymidpoint[:])
        mid2 = createMask(imgcpy,mymidpoint2[:])
        left1 = createMask(imgcpy,myleftpoint[:])
        right2 = createMask(imgcpy,myrightpoint[:])
        right3 = createMask(imgcpy,myrightpoint2[:])
        rss=cv2.add(left,right)
        rss=cv2.add(rss,top01)
        rss=cv2.add(rss,top02)
        rss=cv2.add(rss,top03)
        rss=cv2.add(rss,top04)
        rss=cv2.add(rss,top05)
        rss=cv2.add(rss,top06)
        rss=cv2.add(rss,top07)
        rss=cv2.add(rss,top08)
        rss=cv2.add(rss,top09)
        rss=cv2.add(rss,top010)
        rss=cv2.add(rss,mid)
        rss=cv2.add(rss,mid2)
        rss=cv2.add(rss,right2)
        rss=cv2.add(rss,right3)
        rss=cv2.add(rss,left1)
        imgcpy = creatlips(imgcpy,rss,B,G,R,K,beta)
    cv2.imshow('BGR',cv2.resize(imgcpy,(500,500)))
    cv2.waitKey(1)

