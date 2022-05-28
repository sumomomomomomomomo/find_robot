import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class HSVThreshold:
    """HSV画像の処理用データクラス

    Args:
        hue_max (float): 色相の最大値
        hue_min (float): 色相の最小値
        sat_max (float): 彩度の最大値
        sat_min (float): 彩度の最小値
        val_max (float): 明度の最大値
        val_min (float): 明度の最小値
    """
    hue_max: float
    hue_min: float
    sat_max: float
    sat_min: float
    val_max: float
    val_min: float


@dataclass
class HoughCirclesParameters():
    """HoughCircles用のパラメータクラス

    Args:
        dp (float): 解像度
        min_dist (float): 円同士の最小距離
        param1 (float): ヒステリス処理上限
        param2 (float): 低い→誤検出，高い→未検出
        min_radius (float): 最小半径
        max_radius (float): 最大半径
    """

    dp: float
    min_dist: float
    param1: float
    param2: float
    min_radius: float
    max_radius: float


def pre_img(img: np.ndarray) -> np.ndarray:
    """RGB画像を読み込み、ガウシアンフィルターによる平滑、hsvに変換する関数

    Args:
        img (np.ndarray): RGB画像

    Returns:
        np.ndarray: hsv画像
    """

    img = cv2.GaussianBlur(img, (5, 5), 0)  # ガウシアンフィルター
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # rgb→hsv変換
    return hsv


def mask(hsv: np.ndarray, thr: HSVThreshold) -> np.ndarray:
    """この関数は、hsv画像をマスクする関数

    Args:
        hsv (np.ndarray): hsv画像
        thr (HSVThreshold): HSV閾値

    Returns:
        np.ndarray: 2値画像
    """

    lower = np.array([thr.hue_min, thr.sat_min, thr.val_min])
    upper = np.array([thr.hue_max, thr.sat_max, thr.val_max])
    mask = cv2.inRange(hsv, lower, upper)  # マスク作成

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)  # 黒増やす
    dilation = cv2.dilate(erosion, kernel, iterations=3)  # 白増やす

    inv = cv2.bitwise_not(dilation)  # 白黒反転
    return inv


def find_circle(
        mask: np.ndarray,
        params: HoughCirclesParameters) -> Optional[np.ndarray]:
    """この関数は、円を検出する関数
    入力に、hsv画像と閾値、出力に、円の中心と半径のリスト
    """
    dp = params.dp  # 解像度
    min_dist = params.min_dist  # 円同士の距離
    param1 = params.param1  # ヒステリス処理上限
    param2 = params.param2  # 低い→誤検出，高い→未検出
    min_radius = params.min_radius  # 半径下限
    max_radius = params.max_radius  # 半径上限

    circles = cv2.HoughCircles(
        mask,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius)  # 円検出

    print(circles)
    return circles


def draw_circle(img, circles):
    """この関数は、円を描画する関数
    入力に、rgb画像と検出された円のリスト、出力に、rgb画像
    """
    for i in range(circles.shape[1]):
        cv2.circle(img, center=(int(circles[i, 0]), int(circles[i, 1])), radius=int(
            circles[i, 2]), color=(255, 255, 0), thickness=2)  # 円の描画
        cv2.circle(img, center=(int(circles[i, 0]), int(
            circles[i, 1])), radius=2, color=(0, 0, 255), thickness=3)  # 中心の描画
    return img


def circle_comb(circle_g, circle_p):
    """この関数は、円の組み合わせを行う
    入力に、検出された緑とピンクの円のリスト、
    出力に、組み合わされたロボ1とロボ2のリスト
    """
    d = [[0] * 4 for i in range(2)]
    k = [0] * 6

    for i in range(2):
        for j in range(4):
            d[i][j] = (circle_p[0, i, 0] - circle_g[0, j, 0])**2 + \
                (circle_p[0, i, 1] - circle_g[0, j, 1])**2

    k[0] = d[0][0] + d[0][1] + d[1][2] + d[1][3]
    k[1] = d[0][0] + d[0][2] + d[1][1] + d[1][3]
    k[2] = d[0][0] + d[0][3] + d[1][2] + d[1][1]
    k[3] = d[0][1] + d[0][2] + d[1][0] + d[1][3]
    k[4] = d[0][1] + d[0][3] + d[1][0] + d[1][2]
    k[5] = d[0][2] + d[0][3] + d[1][0] + d[1][1]

    min_d = min(k)
    if min_d == k[0]:
        circle_1 = [circle_p[0, 0], circle_g[0, 0], circle_g[0, 1]]
        circle_2 = [circle_p[0, 1], circle_g[0, 2], circle_g[0, 3]]
    elif min_d == k[1]:
        circle_1 = [circle_p[0, 0], circle_g[0, 0], circle_g[0, 2]]
        circle_2 = [circle_p[0, 1], circle_g[0, 1], circle_g[0, 3]]
    elif min_d == k[2]:
        circle_1 = [circle_p[0, 0], circle_g[0, 0], circle_g[0, 3]]
        circle_2 = [circle_p[0, 1], circle_g[0, 2], circle_g[0, 1]]
    elif min_d == k[3]:
        circle_1 = [circle_p[0, 0], circle_g[0, 1], circle_g[0, 2]]
        circle_2 = [circle_p[0, 1], circle_g[0, 0], circle_g[0, 3]]
    elif min_d == k[4]:
        circle_1 = [circle_p[0, 0], circle_g[0, 1], circle_g[0, 3]]
        circle_2 = [circle_p[0, 1], circle_g[0, 0], circle_g[0, 2]]
    elif min_d == k[5]:
        circle_1 = [circle_p[0, 0], circle_g[0, 2], circle_g[0, 3]]
        circle_2 = [circle_p[0, 1], circle_g[0, 0], circle_g[0, 1]]
    else:
        print('error:not comb')
    return circle_1, circle_2


def find_robo(robo):
    """この関数は、三点の中心を計算する関数"""
    robo = np.array(robo)
    robo_cx = (robo[0, 0] + robo[1, 0] + robo[2, 0]) / 3
    robo_cy = (robo[0, 1] + robo[1, 1] + robo[2, 1]) / 3
    return (robo_cx, robo_cy)


def draw_robo(img, robo_center, pink_center):
    """この関数は、ロボットの外径と向きを描画する関数"""
    x = pink_center[0]
    y = pink_center[1]
    x -= robo_center[0]
    y -= robo_center[1]
    x_f = x * math.cos(math.radians(35)) - y * math.sin(math.radians(35))
    y_f = y * math.cos(math.radians(35)) + x * math.sin(math.radians(35))
    x_f = x_f * 1.5
    y_f = y_f * 1.5
    x_f += robo_center[0]
    y_f += robo_center[1]
    cv2.line(img, (int(robo_center[0]), int(robo_center[1])), (int(
        x_f), int(y_f)), (255, 100, 10), thickness=4)
    cv2.circle(
        img, (int(
            robo_center[0]), int(
            robo_center[1])), radius=3, color=(
                0, 0, 255), thickness=3)
    cv2.circle(
        img, (int(
            robo_center[0]), int(
            robo_center[1])), radius=100, color=(
                0, 0, 255), thickness=2)
    return img


def draw_circle2(img, circles):
    """円の描画関数"""
    for i in range(circles.shape[1]):
        cv2.circle(img, center=(int(circles[0, i, 0]), int(circles[0, i, 1])), radius=int(
            circles[0, i, 2]), color=(255, 255, 0), thickness=2)
        cv2.circle(img, center=(int(circles[0, i, 0]), int(
            circles[0, i, 1])), radius=2, color=(0, 0, 255), thickness=3)
    return img


if __name__ == "__main__":

    # 画像取り込み
    # frame=cv2.imread("tworobo.jpg",1)
    # frame=cv2.imread("onerobo.jpg",1)
    # frame = cv2.imread("input_2.jpg",1)

    # 動画取り込み
    cap = cv2.VideoCapture(r'C:\programs\find_robot\test.mp4')
    if cap.isOpened() is False:
        raise IOError("ビデオファイルを開くとエラーが発生しました")

    # 閾値
    green_thr = HSVThreshold(
        hue_max=99,
        hue_min=67,
        sat_max=157,
        sat_min=52,
        val_max=149,
        val_min=78)
    pink_thr = HSVThreshold(
        hue_max=179,
        hue_min=160,
        sat_max=130,
        sat_min=50,
        val_max=230,
        val_min=160)
    hough_circles_params = HoughCirclesParameters(
        dp=2,
        min_dist=30,
        param1=100,
        param2=14,
        min_radius=13,
        max_radius=20)

    cv2.namedWindow('f')

    # 画像のとき
    # while (True):
    # 動画のとき
    while (cap.isOpened()):
        # 動画のとき
        ret, image = cap.read()
        frame = image[623:1378, 530:2041]
        output = frame.copy()
        output1 = frame.copy()
        output2 = frame.copy()
        output3 = frame.copy()

        # 前処理
        frame = pre_img(frame)
        # 緑マーカー
        frame_green = mask(frame, green_thr)
        circle_green = find_circle(frame_green, hough_circles_params)
        # ピンクマーカー
        frame_pink = mask(frame, pink_thr)
        circle_pink = find_circle(frame_pink, hough_circles_params)

        # 描画
        # output3=draw_circle2(output3,circle_g)
        # output3=draw_circle2(output3,circle_p)
        # cv2.namedWindow('all marker')
        # qcv2.imshow('all marker',output3)

        # ロボットの検出、描画
        # ここまで書いておいて悪いけど、ここはもっと良い書き方を考えたほうが良いと思う
        if (circle_pink.shape[1] == 2) and (circle_green.shape[1] == 4):
            print("two robo")
            robo_1, robo_2 = circle_comb(circle_green, circle_pink)
            robo_1 = np.array(robo_1)
            center_1 = np.array(robo_1)
            print(robo_1, center_1)
            robo_2 = np.array(robo_2)
            robo_1_c = find_robo(robo_1)
            robo_2_c = find_robo(robo_2)
            output = draw_circle(output, robo_1)  # robo_1
            output = draw_circle(output, robo_2)
            output = draw_robo(output, robo_1_c, robo_1[0])
            output = draw_robo(output, robo_2_c, robo_2[0])
        elif (circle_pink.shape[1] == 2) and (circle_green.shape[1] == 3):
            print("two robo")
            # robo_1,robo_2=circle_comb(circle_g,circle_p)

        elif (circle_pink.shape[1] == 1) and (circle_green.shape[1] == 2):
            print("one robo")
            robo_1 = [circle_pink[0, 0],
                      circle_green[0, 0], circle_green[0, 1]]
            robo_1_c = find_robo(robo_1)
            center_1_x = robo_1_c[0]
            center_1_y = robo_1_c[1]
            print(center_1_x, center_1_y)
            robo_1 = np.array(robo_1)
            output = draw_circle(output, robo_1)
            output = draw_robo(output, robo_1_c, robo_1[0])
        else:
            print("no robo")
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.imshow('f', output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
