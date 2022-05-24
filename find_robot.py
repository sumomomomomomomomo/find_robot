import numpy as np
import cv2
import time
import math
import sys
import os

def pre_img(img):
    """この関数は、画像の前処理を行う関数。
    ガウシアンフィルターによる平滑化ののち、
    hsv画像への変換を行っている。
    入力は、RGB画像、出力は、hsv画像
    """
    img = cv2.GaussianBlur(img,(5,5),0)     #ガウシアンフィルター
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV) #rgb→hsv変換
    return hsv

def mask(hsv,thr):
    """この関数は、画像のマスクを作成する関数。
    入力は、hsv画像と閾値、出力は二値画像
    """
    hMax=thr[0] #h 最大
    sMax=thr[1] #s 最大
    vMax=thr[2] #v 最大
    hMin=thr[3] #h 最小
    sMin=thr[4] #s 最小
    vMin=thr[5] #v 最小
    
    lower=np.array([hMin,sMin,vMin])
    upper=np.array([hMax,sMax,vMax])
    mask=cv2.inRange(hsv,lower,upper)   #マスク作成

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)         #黒増やす
    dilation = cv2.dilate(erosion,kernel,iterations = 3)    #白増やす

    inv = cv2.bitwise_not(dilation)     #白黒反転
    return inv

def find_circle(mask,thr):
    """この関数は、円を検出する関数
    入力に、hsv画像と閾値、出力に、円の中心と半径のリスト
    """
    dp=thr[0]           #解像度
    minDist=thr[1]      #円同士の距離
    param1=thr[2]       #ヒステリス処理上限
    param2=thr[3]       #低い→誤検出，高い→未検出
    minRadius=thr[4]    #半径下限
    maxRadius=thr[5]    #半径上限

    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,dp=dp,minDist=minDist,param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)   #円検出
    if circles is None:
        print('no circle')
        circles = np.array([[[0,0,0]]])
        return circles
    else:
        print(circles)
        return circles

def draw_circle(img,circles):
    """この関数は、円を描画する関数
    入力に、rgb画像と検出された円のリスト、出力に、rgb画像
    """
    for i in range(circles.shape[1]):
        cv2.circle(img,center=(int(circles[i,0]),int(circles[i,1])),radius=int(circles[i,2]),color=(255,255,0),thickness=2) #円の描画
        cv2.circle(img,center=(int(circles[i,0]),int(circles[i,1])),radius=2,color=(0,0,255),thickness=3)                   #中心の描画
    return img    

def circle_comb(circle_g,circle_p):
    """この関数は、円の組み合わせを行う
    入力に、検出された緑とピンクの円のリスト、
    出力に、組み合わされたロボ1とロボ2のリスト
    """
    d = [[0] * 4 for i in range(2)]
    k=[0]*6

    for i in range(2):
        for j in range(4):
            d[i][j]=(circle_p[0,i,0]-circle_g[0,j,0])**2 + (circle_p[0,i,1]-circle_g[0,j,1])**2

    k[0]=d[0][0]+d[0][1]+d[1][2]+d[1][3]
    k[1]=d[0][0]+d[0][2]+d[1][1]+d[1][3]
    k[2]=d[0][0]+d[0][3]+d[1][2]+d[1][1]
    k[3]=d[0][1]+d[0][2]+d[1][0]+d[1][3]
    k[4]=d[0][1]+d[0][3]+d[1][0]+d[1][2]
    k[5]=d[0][2]+d[0][3]+d[1][0]+d[1][1]

    min_d = min(k)
    if min_d == k[0]:
        circle_1=[circle_p[0,0],circle_g[0,0],circle_g[0,1]]
        circle_2=[circle_p[0,1],circle_g[0,2],circle_g[0,3]]
    elif min_d == k[1]:
        circle_1=[circle_p[0,0],circle_g[0,0],circle_g[0,2]]
        circle_2=[circle_p[0,1],circle_g[0,1],circle_g[0,3]]
    elif min_d == k[2]:
        circle_1=[circle_p[0,0],circle_g[0,0],circle_g[0,3]]
        circle_2=[circle_p[0,1],circle_g[0,2],circle_g[0,1]]
    elif min_d == k[3]:
        circle_1=[circle_p[0,0],circle_g[0,1],circle_g[0,2]]
        circle_2=[circle_p[0,1],circle_g[0,0],circle_g[0,3]]
    elif min_d == k[4]:
        circle_1=[circle_p[0,0],circle_g[0,1],circle_g[0,3]]
        circle_2=[circle_p[0,1],circle_g[0,0],circle_g[0,2]]
    elif min_d == k[5]:
        circle_1=[circle_p[0,0],circle_g[0,2],circle_g[0,3]]
        circle_2=[circle_p[0,1],circle_g[0,0],circle_g[0,1]]
    else:
        print('error:not comb')
    return circle_1,circle_2 

def find_robo(robo):
    """この関数は、三点の中心を計算する関数"""
    robo=np.array(robo)
    robo_cx=(robo[0,0]+robo[1,0]+robo[2,0])/3
    robo_cy=(robo[0,1]+robo[1,1]+robo[2,1])/3
    return  (robo_cx,robo_cy)

def draw_robo(img,robo_center,pink_center):
    """この関数は、ロボットの外径と向きを描画する関数"""
    x=pink_center[0]
    y=pink_center[1]
    x -= robo_center[0]
    y -= robo_center[1]
    x_f=x*math.cos(math.radians(35))-y*math.sin(math.radians(35))
    y_f=y*math.cos(math.radians(35))+x*math.sin(math.radians(35))
    x_f=x_f*1.5
    y_f=y_f*1.5
    x_f += robo_center[0]
    y_f += robo_center[1]
    cv2.line(img, (int(robo_center[0]),int(robo_center[1])), (int(x_f),int(y_f)), (255, 100, 10),thickness=4)
    cv2.circle(img,(int(robo_center[0]),int(robo_center[1])),radius=3,color=(0,0,255),thickness=3)
    cv2.circle(img,(int(robo_center[0]),int(robo_center[1])),radius=100,color=(0,0,255),thickness=2)
    return img

def draw_circle2(img,circles):
    """円の描画関数"""
    for i in range(circles.shape[1]):
        cv2.circle(img,center=(int(circles[0,i,0]),int(circles[0,i,1])),radius=int(circles[0,i,2]),color=(255,255,0),thickness=2)
        cv2.circle(img,center=(int(circles[0,i,0]),int(circles[0,i,1])),radius=2,color=(0,0,255),thickness=3)
    return img

if __name__ == "__main__":

    #画像取り込み
    #frame=cv2.imread("tworobo.jpg",1)
    #frame=cv2.imread("onerobo.jpg",1)
    #frame = cv2.imread("input_2.jpg",1)

    #動画取り込み
    cap = cv2.VideoCapture(r'C:\\\.mp4')
    if (cap.isOpened()== False):  
        print("ビデオファイルを開くとエラーが発生しました")

    #閾値
    green_thr=np.array([99,157,149,67,52,78])
    pink_thr=np.array([179,130,230,160,50,160])
    c_thr=np.array([2,30,100,14,13,20])

    cv2.namedWindow('f')

    #画像のとき
    #while (True):
    #動画のとき
    while (cap.isOpened()):
        #動画のとき
        ret,image = cap.read()
        frame=image[623:1378,530:2041]
        output=frame.copy()
        output1=frame.copy()
        output2=frame.copy()
        output3=frame.copy()

        #前処理
        frame = pre_img(frame)
        #緑マーカー
        frame_g=mask(frame,green_thr)
        circle_g=find_circle(frame_g,c_thr)
        #ピンクマーカー
        frame_p=mask(frame,pink_thr)
        circle_p=find_circle(frame_p,c_thr)
        
        #描画
        #output3=draw_circle2(output3,circle_g)
        #output3=draw_circle2(output3,circle_p)
        #cv2.namedWindow('all marker')
        #qcv2.imshow('all marker',output3)

        #ロボットの検出、描画
        if (circle_p.shape[1]==2) and (circle_g.shape[1]==4):
            print("two robo")
            robo_1,robo_2=circle_comb(circle_g,circle_p)
            robo_1=np.array(robo_1)
            center_1=np.array(robo_1)
            print(robo_1,center_1)
            robo_2=np.array(robo_2)
            robo_1_c=find_robo(robo_1)
            robo_2_c=find_robo(robo_2)
            output=draw_circle(output,robo_1)#robo_1
            output=draw_circle(output,robo_2)
            output=draw_robo(output,robo_1_c,robo_1[0])
            output=draw_robo(output,robo_2_c,robo_2[0])
        elif (circle_p.shape[1]==2) and (circle_g.shape[1]==3):
            print("two robo")
            #robo_1,robo_2=circle_comb(circle_g,circle_p)
        
        elif (circle_p.shape[1]==1) and (circle_g.shape[1]==2):
            print("one robo")
            robo_1=[circle_p[0,0],circle_g[0,0],circle_g[0,1]]
            robo_1_c=find_robo(robo_1)
            center_1_x=robo_1_c[0]
            center_1_y=robo_1_c[1]
            print(center_1_x,center_1_y)
            robo_1=np.array(robo_1)
            output=draw_circle(output,robo_1)
            output=draw_robo(output,robo_1_c,robo_1[0])
        else:
            print("no robo")
            key =cv2.waitKey(1) & 0xFF
            if key==ord("q"):
                break

        cv2.imshow('f',output)
        key =cv2.waitKey(1) & 0xFF
        if key==ord("q"):
            break
