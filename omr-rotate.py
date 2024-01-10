import numpy as np
import cv2
import math

img = cv2.imread("images/omr-turned.jpg",0)
# img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles =cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,60,param1=60,param2=40,minRadius=200,maxRadius=0)

circles = np.uint16(np.around(circles))
counter=0
correctC=[]
xC=[]
yC=[]

for i in circles[0,:]: 
    #cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    #cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),2)
    cv2.putText(cimg,str(i[0])+","+str(i[1])+","+str(i[2]),(i[0],i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(255,0,0),1,cv2.LINE_AA)
    correctC.append((i[0],i[1],i[2]))
    xC.append(i[0])
    yC.append(i[1])
    counter+=1

print("Circle Count is : " + str(counter))

# xCS=sorted(xC)
# yCS=sorted(yC)
# xS=sorted(correctC, key=lambda correctC:correctC[0])

# q1=sorted(xS[:4],key=lambda correctC: correctC[1])
# q2=sorted(xS[4:8],key=lambda correctC: correctC[1])
# q3=sorted(xS[8:12],key=lambda correctC: correctC[1])
# q4=sorted(xS[12:16],key=lambda correctC: correctC[1])
# q5=sorted(xS[16:20],key=lambda correctC: correctC[1])
# q6=sorted(xS[20:24],key=lambda correctC: correctC[1])
# q7=sorted(xS[24:28],key=lambda correctC: correctC[1])
# q8=sorted(xS[28:32],key=lambda correctC: correctC[1])
# q9=sorted(xS[32:],key=lambda correctC: correctC[1])

# sortedTmp=[q1,q2,q3,q4,q5,q6,q7,q8,q9]
# sorted=[]

# for i in sortedTmp:
#     for j in i:
#         sorted.append(j)

# for i in range(36):
#     cv2.putText(cimg,str(i),(sorted[i][0],sorted[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),3,cv2.LINE_AA)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()