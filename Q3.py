import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
   
# Load the left and right images 
# in gray scale 
imgLeft = cv2.imread('./images/left.jpg',0) 
imgRight = cv2.imread('./images/right.jpg',0) 

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict,arucoParams)
(cornersl, ids, rejected) = detector.detectMarkers(imgLeft)


(cornersr, ids, rejected) = detector.detectMarkers(imgRight)


ptsLeft= []
ptsRight= []

for i in range(len(cornersl)):
    for j in range(len(cornersl[0][0])):
        ptsLeft.append(list(cornersl[i][0][j]))
        ptsRight.append(list(cornersr[i][0][j]))


print("Left = ", np.int32(ptsLeft))
print("Right = ", np.int32(ptsRight))


ptsLeft = np.int32(ptsLeft) 
ptsRight = np.int32(ptsRight) 
F, mask = cv2.findFundamentalMat(ptsLeft, 
                                 ptsRight, 
                                 cv2.FM_LMEDS) 
print(F)
# We select only inlier points 
ptsLeft = ptsLeft[mask.ravel() == 1] 
ptsRight = ptsRight[mask.ravel() == 1]  


def drawlines(img1, img2, lines, pts1, pts2, left = True): 
    
    r, c = img1.shape 
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) 
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) 
      
    for r, pt1, pt2 in zip(lines, pts1, pts2): 
          
        color = tuple(np.random.randint(0, 255, 
                                        3).tolist()) 
          
        x0, y0 = map(int, [0, -r[2] / r[1] ]) 
        x1, y1 = map(int,  
                     [c, -(r[2] + r[0] * c) / r[1] ]) 
        if left:  
            img1 = cv2.line(img1,  
                            (x0, y0), (x1, y1), color, 1) 
            img1 = cv2.circle(img1, 
                            tuple(pt1), 5, color, -1) 
            img2 = cv2.circle(img2,  
                            tuple(pt2), 5, color, -1) 
        else:
            img1 = cv2.circle(img1, 
                            tuple(pt1), 5, color, -1) 
            img2 = cv2.circle(img2,  
                            tuple(pt2), 5, color, -1)
    return img1, img2 

# Find epilines corresponding to points 
# in right image (second image) and 
# drawing its lines on left image 
linesLeft = cv2.computeCorrespondEpilines(ptsRight.reshape(-1, 1, 2),2, F) 
linesLeft = linesLeft.reshape(-1, 3) 
img5, img6 = drawlines(imgLeft, imgRight,  
                       linesLeft, ptsLeft, 
                       ptsRight) 
   
# Find epilines corresponding to  
# points in left image (first image) and 
# drawing its lines on right image 
linesRight = cv2.computeCorrespondEpilines(ptsLeft.reshape(-1, 1, 2),1, F) 
linesRight = linesRight.reshape(-1, 3) 
img3, img4 = drawlines(imgRight, imgLeft,  
                       linesRight, ptsRight, 
                       ptsLeft) 
   
plt.subplot(121), plt.imshow(img5) 
plt.subplot(122), plt.imshow(img3) 
plt.show()

# Compute essential matrix (E) from fundamental matrix (F)
E, _ = cv2.findEssentialMat(ptsLeft, ptsRight)

# Decompose essential matrix into rotation matrix (R) and translation vector (t)
_, R, t, _ = cv2.recoverPose(E, ptsLeft, ptsRight)

print("Rotation Matrix (R):")
print(R)
print("\nTranslation Vector (t):")
print(t)