import numpy as np
import cv2
# testing image corners
import numpy as np
import cv2 as cv
import glob
import math
from scipy.spatial.transform import Rotation as R
# Load previously saved data
with np.load('B.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

print(cv.__version__)
print("mtx:")
print(mtx)
print("dist:")
print(dist)

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None) #worked!

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

    if ret == True:

        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret,rvecs,tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        #r = R.from_rotvec(3,rvecs)
        # Convert 3x1 rotation vector to rotation matrix for further computation            
        rotation_matrix, jacobian = cv.Rodrigues(rvecs)
        # Projection Matrix
        tvecs_new = -np.matrix(rotation_matrix).T * np.matrix(tvecs)            
        #pmat = np.hstack((rotation_matrix, tvecs))
        pmat = np.hstack((rotation_matrix, tvecs_new))
        roll, pitch, yaw = cv.decomposeProjectionMatrix(pmat)[-1]
        # print(roll)
        print('Roll: {:.2f}\nPitch: {:.2f}\nYaw: {:.2f}'.format(float(roll), float(pitch), float(yaw)))
        #print(-np.matrix(rotation_matrix).T * np.matrix(tvecs))
        # print(cv.decomposeProjectionMatrix(pmat)[-1]) #another way to get Euler angles
          
        # C = np.matmul(-rotation_matrix.transpose(), tvecs)
        # # Orientation vector
        # O = np.matmul(rotation_matrix.T, np.array([0, 0, 1]).T)
        # camera_pose = C.squeeze()
        # camera_orientation = O

        # # project 3D points to image plane
        # imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        # img = draw(frame,corners2,imgpts)
        # # Resize image
        # frame = cv.resize(img, (540,960))                    
        # # Show image 
        # #cv2.imshow("output", frame)                               
    
    # # Display the resulting frame
    # frame = cv.resize(frame, (540,960))  
    # cv2.imshow('frame',frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    # k = cv2.waitKey(0) & 0xFF
    # if k == ord('s'):
    #     cv2.imwrite(fname[:6]+'.png', frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()