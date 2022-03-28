import depthai
import cv2
import math
import mediapipe as mp
import os
import numpy as np
from irisSeg import irisSeg
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


RANSAC = 0;
LEFT_IRIS       = [ 474, 475, 476, 477 ];
RIGHT_IRIS      = [ 469, 470, 471, 472 ];
points_eye_r    = [  7,  158, 157, 163 ];
points_eye_l    = [ 373, 384, 385, 390 ];
METHOD = 0;


# In this case the tip of the nose is the fixed point in the face (REF_POINT)
cam_res = [900, 900];
short_smooth = 8;
calib = 0;
X_calib, Y_calib = [], [];
cur_pointer = np.zeros(4);
nframe = 0;


WINDOWS_RES = [2560,1140];
big_image = np.zeros([WINDOWS_RES[1], WINDOWS_RES[0], 3],dtype=np.uint8);
# Create Full screen window
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

iris_2dl = np.zeros((short_smooth,4));
irrad_2dl = np.zeros((short_smooth,2));
cur_pointerl = np.zeros((short_smooth,4));

vec = np.zeros((4));

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6)



def initiate_dai():
    # Create Videostream (needed so window stays open)
    # Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
    pipeline = depthai.Pipeline()

    # First, we want the Color camera as the output
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(cam_res[0], cam_res[1])  # 300x300 will be the preview frame size, available as 'preview' output of the node
    cam_rgb.setInterleaved(False)
    #cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_4_K)
    #cam_rgb.initialControl.setManualExposure(9000, 1200)
    #cam_rgb.setFps(40)

    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)
    return pipeline;
    
pipeline = initiate_dai();

def draw_calib(image,WINDOWS_RES):

    # Show Spots for calibration
    p = np.zeros((9,2)).astype('int')
    
    p[0] = (5,5)
    p[1] = (int(WINDOWS_RES[0]/2)-5,5)
    p[2] = (int(WINDOWS_RES[0]-5),5)
    
    p[3] = (5,int(WINDOWS_RES[1]/2))
    p[4] = (int(WINDOWS_RES[0]/2)-5,int(WINDOWS_RES[1]/2))
    p[5] = (int(WINDOWS_RES[0]-5),int(WINDOWS_RES[1]/2))
    
    p[6] = (5,int(WINDOWS_RES[1]-5))
    p[7] = (int(WINDOWS_RES[0]/2-5),int(WINDOWS_RES[1]-5))
    p[8] = (int(WINDOWS_RES[0]-5),int(WINDOWS_RES[1]-5))    
    
    cv2.circle(image, (p[0,0], p[0,1]), 2, (255,0,0), 2)
    cv2.putText(image, '1', (p[0,0], int(p[0,1]+10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.circle(image, (p[1,0], p[1,1]), 2, (255,0,0), 2)
    cv2.putText(image, '2', (int(p[1,0]-25), int(p[1,1]+10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.circle(image, (p[2,0], p[2,1]), 2, (255,0,0), 2)
    cv2.putText(image, '3', (int(p[2,0]-10), int(p[2,1]+35)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.circle(image, (p[3,0], p[3,1]), 2, (255,0,0), 2)
    cv2.putText(image, '4', (p[3,0], int(p[3,1]+10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.circle(image, (p[4,0], p[4,1]), 2, (255,0,0), 2)
    cv2.putText(image, '5', (int(p[4,0]-25), int(p[4,1]+10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.circle(image, (p[5,0],p[5,1]), 2, (255,0,0), 2)
    cv2.putText(image, '6', (int(p[5,0]-10), int(p[5,1]+10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.circle(image, (p[6,0], p[6,1]), 2, (255,0,0), 2)
    cv2.putText(image, '7', (p[6,0],int(p[6,1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.circle(image, (p[7,0], p[7,1]), 2, (255,0,0), 2)
    cv2.putText(image, '8', (int(p[7,0]-25),int(p[7,1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.circle(image, (p[8,0], p[8,1]), 2, (255,0,0), 2)
    cv2.putText(image, '9', (int(p[8,0]-10),int(p[8,1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return p

def draw_parallel(image,hor_vert,coord, color):
    line_thickness = 1
    if hor_vert == "vertical":
        x1, y1 = int(coord) , 0
        x2, y2 = int(coord), int(image.shape[0])
    if hor_vert == "horizontal":
        x1, y1 = 0, int(coord)
        x2, y2 = int(image.shape[1]), int(coord)
    cv2.line(image, (x1, y1), (x2, y2), color, thickness=line_thickness)

if 'first_face_3d' in locals():
    del first_face_3d;

def RANSAC_calc(X,Y,Z):
    from sklearn import linear_model, datasets
    ransac = linear_model.RANSACRegressor(max_trials=950)
    ransac.fit(X.reshape(-1,1), Y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    X = X[inlier_mask];
    Y = Y[inlier_mask];
    Z = Z[inlier_mask];
    return X, Y, Z;

def calibrate(X_calib, Y_calib, RANSAC = 0, METHOD=0):

    if METHOD == 0:
    # Conventional 2nd order least squares function
        X = np.array(X_calib)[:,0]
        Y = np.array(X_calib)[:,1]
        Z = np.array(Y_calib)[:,0]

        A = np.array([X*0+1, X, Y, X*Y, X**2,Y**2]).T
        B = Z.flatten()

        coeff_lx, r, rank, s = np.linalg.lstsq(A, B)
                
        # RX
        X = np.array(X_calib)[:,2]
        Y = np.array(X_calib)[:,3]

        A = np.array([X*0+1, X, Y, X*Y, X**2,Y**2]).T
        B = Z.flatten()

        coeff_rx, r, rank, s = np.linalg.lstsq(A, B)
                
        # LY
        X = np.array(X_calib)[:,0]
        Y = np.array(X_calib)[:,1]
        Z = np.array(Y_calib)[:,1]
                

        A = np.array([X*0+1, X, Y, X*Y, X**2,Y**2]).T
        B = Z.flatten()
                
        coeff_ly, r, rank, s = np.linalg.lstsq(A, B)
                
        # RY
        X = np.array(X_calib)[:,2]
        Y = np.array(X_calib)[:,3]
                

        A = np.array([X*0+1, X, Y, X*Y, X**2,Y**2]).T
        B = Z.flatten()

        coeff_ry, r, rank, s = np.linalg.lstsq(A, B)
        
        
        return coeff_lx, coeff_ly, coeff_rx, coeff_ry

    if METHOD == 1:
        rf_estimators = 600;
        rf_max_depth  = 12;
        rf_max_features = 'log2';
        from sklearn.ensemble import ExtraTreesRegressor
        X = np.array(X_calib)[:,[0,2]];
        Y = np.array(Y_calib)[:,0];
            
        rf_x = ExtraTreesRegressor(n_estimators = rf_estimators, max_depth = rf_max_depth, max_features = rf_max_features)
        rf_x.fit(X,Y)
        
        X = np.array(X_calib)[:,[1,3]];
        Y = np.array(Y_calib)[:,1];

        rf_y = ExtraTreesRegressor(n_estimators = rf_estimators, max_depth = rf_max_depth, max_features = rf_max_features)
        rf_y.fit(X,Y)
        
        return rf_x, rf_y






import math

# ----------------------------------------------------------------------------

class LowPassFilter(object):

    def __init__(self, alpha):
        self.__setAlpha(alpha)
        self.__y = self.__s = None

    def __setAlpha(self, alpha):
        alpha = float(alpha)
        if alpha<=0 or alpha>1.0:
            raise ValueError("alpha (%s) should be in (0.0, 1.0]"%alpha)
        self.__alpha = alpha

    def __call__(self, value, timestamp=None, alpha=None):        
        if alpha is not None:
            self.__setAlpha(alpha)
        if self.__y is None:
            s = value
        else:
            s = self.__alpha*value + (1.0-self.__alpha)*self.__s
        self.__y = value
        self.__s = s
        return s

    def lastValue(self):
        return self.__y

# ----------------------------------------------------------------------------

class OneEuroFilter(object):

    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        if freq<=0:
            raise ValueError("freq should be >0")
        if mincutoff<=0:
            raise ValueError("mincutoff should be >0")
        if dcutoff<=0:
            raise ValueError("dcutoff should be >0")
        self.__freq = float(freq)
        self.__mincutoff = float(mincutoff)
        self.__beta = float(beta)
        self.__dcutoff = float(dcutoff)
        self.__x = LowPassFilter(self.__alpha(self.__mincutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__dcutoff))
        self.__lasttime = None
        
    def __alpha(self, cutoff):
        te    = 1.0 / self.__freq
        tau   = 1.0 / (2*math.pi*cutoff)
        return  1.0 / (1.0 + tau/te)

    def __call__(self, x, timestamp=None):
        if self.__lasttime and timestamp:
            self.__freq = 1.0 / (timestamp-self.__lasttime)
        self.__lasttime = timestamp
        prev_x = self.__x.lastValue()
        dx = 0.0 if prev_x is None else (x-prev_x)*self.__freq # FIXME: 0.0 or value?
        edx = self.__dx(dx, timestamp, alpha=self.__alpha(self.__dcutoff))
        cutoff = self.__mincutoff + self.__beta*math.fabs(edx)
        return self.__x(x, timestamp, alpha=self.__alpha(cutoff))






#first_face_3d = np.load('C:/Users/SebastianG/Nextcloud/_SEBASTIAN/Forschung/Eye Tracking/AET-v2/FACE_ROT/face_3d_middle.npy')

with depthai.Device(pipeline) as device:
    # From this point, the Device will be in "running" mode and will start sending data via XLink
    q_rgb = device.getOutputQueue("rgb")
    frame = None;
    img_h, img_w = cam_res[1], cam_res[0];
    ref_l = [0,0];
    ref_r = [0,0];
    l_cx, l_cy, r_cx, r_cy = 0, 0, 0, 0;
    
    nframe = 0;

    config = {
            'freq': 120,       # Hz
            'mincutoff': 1.0,  # FIXME
            'beta': 1.0,       # FIXME
            'dcutoff': 1.0     # this one should be ok
            }

    f_cry = OneEuroFilter(**config)
    f_crx = OneEuroFilter(**config)
    f_cly = OneEuroFilter(**config)
    f_clx = OneEuroFilter(**config)


    
    while True:
        in_rgb = q_rgb.tryGet()
        big_image.fill(255);
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
            
        if frame is not None:

            results = face_mesh.process(frame)            
            
            if results.multi_face_landmarks:
                
                landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in results.multi_face_landmarks[0].landmark]
                )
                landmarks[:,0] = landmarks[:,0]#*img_w;
                landmarks[:,1] = landmarks[:,1]#*img_h;
                
                if 'first_face_3d' not in locals():
                    first_face_3d = landmarks;



                # Essential muscle Ts
                ret, M, mask = cv2.estimateAffine3D(landmarks,first_face_3d);
                M = np.mat(M);
                #trans_mat = np.vstack([ M, [0,0,0,1]])

                #for point in landmarks:#np.mean(face_3d,axis=2):
                #    point = np.append(point,1)
                #    point = M.dot(point).T;
                #    cv2.circle(frame, (int(point[0]*cam_res[0]), int(point[1])*cam_res[1]), 2, (0,0,255), 1)

                #angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(M[:,:3])




                ref_l = np.mean(landmarks[points_eye_l,:2], axis = 0)
                ref_r = np.mean(landmarks[points_eye_r,:2], axis = 0)                

                (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(landmarks[LEFT_IRIS,:2].astype('float32'))
                (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(landmarks[RIGHT_IRIS,:2].astype('float32'))
                
                center_left = np.array([l_cx, l_cy], dtype=np.float32)
                center_right = np.array([r_cx, r_cy], dtype=np.float32)
                
                iris_2dl = np.vstack((iris_2dl,np.array([center_left, center_right]).flatten()))
                iris_2dl = np.delete(iris_2dl, 0, 0);
                
                irrad_2dl = np.vstack((irrad_2dl,np.array([l_radius, r_radius]).flatten()))
                irrad_2dl = np.delete(irrad_2dl, 0, 0);
               
                cv2.circle(frame, ( int(center_left[0]*img_w), int(center_left[1]*img_h) ) , int(1), (255,0,255), 2, cv2.LINE_AA)
                cv2.circle(frame, ( int(center_right[0]*img_w), int(center_right[1]*img_h) ) , int(1), (0,0,255), 2, cv2.LINE_AA)
                cv2.circle(frame, ( int(ref_l[0]*img_w),int(ref_l[1]*img_h) ) , int(1), (0,255,0), 2, cv2.LINE_AA)
                               
                               
            y_offset = 550;
            x_offset = 1250;
            big_image[y_offset:y_offset+480, x_offset:x_offset+480] = cv2.resize(frame,(480,480));
            
            calibp = draw_calib(big_image,WINDOWS_RES)
            
            
            vec[0] = f_clx((ref_l[0] - l_cx), nframe);
            vec[1] = f_cly((ref_l[1] - l_cy), nframe);
            vec[2] = f_crx((ref_r[0] - r_cx), nframe);
            vec[3] = f_cry((ref_r[1] - r_cy), nframe);


            if calib == 1:  
                if METHOD == 0:
                    cur_pointer[0] = coeff_lx[0] + coeff_lx[1]*vec[0] + coeff_lx[2]*vec[1] + coeff_lx[3]*vec[0]*vec[1] + coeff_lx[4]*(vec[0]**2) + coeff_lx[5]*(vec[1]**2);
                    cur_pointer[1] = coeff_rx[0] + coeff_rx[1]*vec[2] + coeff_rx[2]*vec[3] + coeff_rx[3]*vec[2]*vec[3] + coeff_rx[4]*(vec[2]**2) + coeff_rx[5]*(vec[3]**2);
                    cur_pointer[2] = coeff_ly[0] + coeff_ly[1]*vec[0] + coeff_ly[2]*vec[1] + coeff_ly[3]*vec[0]*vec[1] + coeff_ly[4]*(vec[0]**2) + coeff_ly[5]*(vec[1]**2);
                    cur_pointer[3] = coeff_ry[0] + coeff_ry[1]*vec[2] + coeff_ry[2]*vec[3] + coeff_ry[3]*vec[2]*vec[3] + coeff_ry[4]*(vec[2]**2) + coeff_ry[5]*(vec[3]**2);
                    
                    cur_pointerl = np.vstack((cur_pointerl,np.array([cur_pointer[0], cur_pointer[1],cur_pointer[2],cur_pointer[3]]).flatten()))
                    cur_pointerl = np.delete(cur_pointerl, 0, 0);
                    
                    draw_parallel(big_image,'vertical',int((np.mean(cur_pointerl[:,1]) + np.mean(cur_pointerl[:,0])) / 2), (0,0,255))
                    draw_parallel(big_image,'horizontal',int((np.mean(cur_pointerl[:,2]) + np.mean(cur_pointerl[:,3])) / 2), (0,0,255))

                if METHOD == 1:
                    pred_x = rf_x.predict(vec[[0,2]].reshape(1,-1));
                    pred_y = rf_y.predict(vec[[1,3]].reshape(1,-1));
                    draw_parallel(big_image,'vertical',int(pred_x), (0,0,255))
                    draw_parallel(big_image,'horizontal',int(pred_y), (0,0,255))

                
                

            cv2.imshow('window', big_image)
            cv2.imshow('orig', frame)

        key = cv2.waitKey(1);
        if key == ord('q'):
            break
        
        # Calibrate horizontal axis
        if key == ord('1'):
            Y_calib.append([calibp[0,0],calibp[0,1]])
            X_calib.append(list(vec))
                

        if key == ord('2'):
            Y_calib.append([calibp[1,0],calibp[1,1]])
            X_calib.append(list(vec))

        if key == ord('3'):
            Y_calib.append([calibp[2,0],calibp[2,1]])
            X_calib.append(list(vec))

        if key == ord('4'):
            Y_calib.append([calibp[3,0],calibp[3,1]])
            X_calib.append(list(vec))

        if key == ord('5'):
            Y_calib.append([calibp[4,0],calibp[4,1]])
            X_calib.append(list(vec))

        if key == ord('6'):
            Y_calib.append([calibp[5,0],calibp[5,1]])
            X_calib.append(list(vec))

        if key == ord('7'):
            Y_calib.append([calibp[6,0],calibp[6,1]])
            X_calib.append(list(vec))

        if key == ord('8'):
            Y_calib.append([calibp[7,0],calibp[7,1]])
            X_calib.append(list(vec))

        if key == ord('9'):
            Y_calib.append([calibp[8,0],calibp[8,1]])
            X_calib.append(list(vec))

        if key == ord('x'):
            if METHOD == 0:
                coeff_lx, coeff_ly, coeff_rx, coeff_ry = calibrate(X_calib, Y_calib, 0, METHOD)
            if METHOD == 1:
                rf_x, rf_y = calibrate(X_calib, Y_calib, 0, METHOD)
            
        if key == ord('c'):
            calib = 1;

        if key == ord('f'):
            first_face_3d = landmarks;
        nframe += 1

cv2.destroyAllWindows()



