import cv2
import time,board,busio
import numpy as np
import adafruit_mlx90640
import matplotlib.pyplot as plt
import serial,time
import RPi.GPIO as GPIO
import math
import sys
from scipy import ndimage
from time import sleep

 
servoPin = 13 #GPI0 13
SERVO_MAX_DUTY = 12 #서브모터의 최대 주기
SERVO_MIN_DUTY = 3 #서브모터의 최소 주기 

max_angle_deviation = 2 #조향각 최대 편차
past_steering_angle = 1 #과거 조향각
ser = serial.Serial("/dev/serial0", 115200,timeout=0) # mini UART serial device

arduino_steering_angle = serial.Serial('/dev/ttyAMA1',115200,timeout = 0) #조향각 UART통신 설정
arduino_angle_distance = serial.Serial('/dev/ttyAMA2',115200,timeout = 0) #각도 및 거리 UART통신 설정
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000) # I2C 통신 설정
mlx = adafruit_mlx90640.MLX90640(i2c) # MLX90640 SETUP
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ # MLX90640 갱신속도


GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPin,GPIO.OUT)

servo = GPIO.PWM(servoPin,50) #Lidar 서보모터 설정
servo.start(0)

mlx_shape = (24,32) # mlx90640 shape
mlx_frame = np.zeros(mlx_shape[0]*mlx_shape[1]) # 768 pts


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,360)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)

 

fig = plt.figure(figsize=(6,4)) # figure 설정
ax = fig.add_subplot(111) # subplot 추가
fig.subplots_adjust(0.05,0.05,0.95,0.95) # padding 제거
therm1 = ax.imshow(np.zeros(mlx_shape),cmap=plt.cm.winter,vmin=25,vmax=45) # preemptive image
cbar = fig.colorbar(therm1) # colorbar 설정
cbar.set_label('Temperature [$^{\circ}$C]',fontsize=14) # colorbar label
fig.canvas.draw() # draw figure to copy background
ax_background = fig.canvas.copy_from_bbox(ax.bbox) # copy background
fig.show()#'''

def setServoPos(degree):

  if degree > 180:
    degree = 180
    
  duty = SERVO_MIN_DUTY+(degree*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0)
  
  servo.ChangeDutyCycle(duty)

def plot_update():

    fig.canvas.restore_region(ax_background)
    mlx.getFrame(mlx_frame) # read mlx90640
    data_array = np.fliplr(np.reshape(mlx_frame,mlx_shape))

    global data_analysis #mlx90640 탐지영역
    data_analysis = data_array[:,16:17] #기준 탐지영역 slicing
    global Temp_max #온도 최댓값
    Temp_max= np.max(data_analysis)

    global Temp_min 
    Temp_min= np.min(data_analysis)

    therm1.set_array(data_array) # set data
    therm1.set_clim(vmin=np.min(data_array),vmax=np.max(data_array)) # set bounds
    cbar.on_mappable_changed(therm1) # update colorbar range
    ax.draw_artist(therm1) # draw new thermal image
    fig.canvas.blit(ax.bbox) # draw background
    fig.canvas.flush_events() # show the new image

    return

def read_tfluna_data():
    
    if ser.isOpen() == False:
        ser.open()

    while True:
        counter = ser.in_waiting # 수신버퍼에 남아있는 바이트 수
                               
        if counter > 8:
            bytes_serial = ser.read(9) # read 9 bytes
            ser.reset_input_buffer() # reset buffer

            if bytes_serial[0] == 0x59 and bytes_serial[1] == 0x59: # check first two bytes
                distance = bytes_serial[2] + bytes_serial[3]*256 # distance in next two bytes
                print(distance)
                ser.close()

                return distance

def Searching_index(): # 탐지영역의 index 추출
    max_val = np.max(data_analysis)
    points = np.where(np.logical_and(data_analysis > max_val-2 , data_analysis<= max_val))
    np_points = np.array(points)
    x_top = np.min(np_points[0])
    x_bottom = np.max(np_points[0])
    return x_top,x_bottom

def Calculation_theta(bt_index): #index를 이용한 각도 계산

    index_to_theta = 42.5 + 35 / 23 * (23 - bt_index)
    mlx_bottom = 60 * math.tan(index_to_theta * math.pi / 180)
    lidar_bottom = mlx_bottom + 1.75
    val = lidar_bottom / 51.75
    radian = math.atan2(lidar_bottom , 51.75)
    degree = radian * 180/math.pi

    return degree

def DetectLineSlope(src): # 차선 인지 및 조향각 계산

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    can = cv2.Canny(blur, 50, 200, None, 3)
    height = can.shape[0]  # 카메라 해상도가 360 640이어서 360이 출력되는 듯
    rectangle = np.array([[(0, height), (0, 170), (360, 170), (360, 360)]])
    mask = np.zeros_like(can) #360 * 640
    cv2.fillPoly(mask, rectangle, 255, cv2.LINE_4)
    masked_image = cv2.bitwise_and(can, mask)
    ccan = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR)


    line_arr = cv2.HoughLinesP(masked_image, 1, np.pi / 180, 20, minLineLength=15, maxLineGap=10)
    line_R = np.empty((0, 5) , int)  
    line_L = np.empty((0, 5) , int)  

    if line_arr is not None:
        line_arr2 = np.empty((len(line_arr), 5), int)

        for i in range(0, len(line_arr)):
            temp = 0
            l = line_arr[i][0]
            line_arr2[i] = np.append(line_arr[i], np.array((np.arctan2(l[1] - l[3], l[0] - l[2]) * 180) / np.pi))# degree transform
            if line_arr2[i][1] > line_arr2[i][3]:
                temp = line_arr2[i][0], line_arr2[i][1]
                line_arr2[i][0], line_arr2[i][1] = line_arr2[i][2], line_arr2[i][3]
                line_arr2[i][2], line_arr2[i][3] = temp
            if line_arr2[i][0] < 180 and (abs(line_arr2[i][4]) < 150 and abs(line_arr2[i][4]) > 60):
                line_L = np.append(line_L, line_arr2[i])
            elif line_arr2[i][0] > 180 and (abs(line_arr2[i][4]) < 150 and abs(line_arr2[i][4]) > 60):
                line_R = np.append(line_R, line_arr2[i])

    line_L = line_L.reshape(int(len(line_L) / 5), 5)
    line_R = line_R.reshape(int(len(line_R) / 5), 5)

    try:

        line_L = line_L[line_L[:, 0].argsort()[-1]]
        degree_L = line_L[4]
        cv2.line(ccan, (line_L[0], line_L[1]), (line_L[2], line_L[3]), (255, 0, 0), 10, cv2.LINE_AA)

    except:
        degree_L = 0

    try:
        line_R = line_R[line_R[:, 0].argsort()[0]]
        degree_R = line_R[4]
        cv2.line(ccan, (line_R[0], line_R[1]), (line_R[2], line_R[3]), (255, 0, 0), 10, cv2.LINE_AA)

    except:
        degree_R = 0

    try:
        if (degree_L != 0) and (degree_R != 0): #좌우 차선을 인지한 경우
            mid_x1 = int((line_L[0] + line_R[0]) / 2)
            mid_x2 = int((line_L[2] + line_R[2]) / 2)
            mid_y1 = int((line_L[1] + line_R[1]) / 2)
            mid_y2 = int((line_L[3] + line_R[3]) / 2)
            
        elif degree_L == 0: #오른쪽 차선만 인지한 경우
            mid_x1 = int(line_R[0]-100)
            mid_x2 = int(line_R[2]-100)
            mid_y1 = int(line_R[1])
            mid_y2 = int(line_R[3])

        else: #왼쪽 차선만 인지한 경우
            mid_x1 = int(line_L[0]+100)
            mid_x2 = int(line_L[2]+100)
            mid_y1 = int(line_L[1])
            mid_y2 = int(line_L[3])

    except: #차선을 인지 못한 경우
        mid_x1 = 0
        mid_x2 = 0
        mid_y1 = 0
        mid_y2 = 0
        
    cv2.line(ccan, (mid_x1, mid_y1), (mid_x2, mid_y2), (0, 0, 255), 10, cv2.LINE_AA)


    global slope 

    if mid_x1 != mid_x2:
        slope = math.atan((mid_y2-mid_y1)/(mid_x1-mid_x2)) * 180 / math.pi
        
        if slope < 0 :
            slope = 180 + slope

    elif mid_x1 == mid_x2 and mid_y1 != mid_y2:
        slope = 90
        
    else:
        slope = 180

    steering_angle = slope - 90
    global past_steering_angle
    compensation_angle = 8

    if(-compensation_angle < steering_angle < compensation_angle):
        steering_angle = -2
        past_steering_angle = steering_angle
        print('Go')

    elif(steering_angle>0):

        steering_angle = steering_angle - compensation_angle

        if(abs(steering_angle-past_steering_angle)<= max_angle_deviation):
            past_steering_angle = steering_angle

        elif((steering_angle-past_steering_angle)>0):
            steering_angle = past_steering_angle + max_angle_deviation    
            past_steering_angle = steering_angle

        else:
            steering_angle = past_steering_angle - max_angle_deviation
            past_steering_angle = steering_angle

        if(steering_angle > 40):
            steering_anlge = 40
            past_steering_angle = steering_angle

        print('Left',int(steering_angle))

   
    else:
        steering_angle = steering_angle + compensation_angle
       
        if(abs(steering_angle - past_steering_angle) <= max_angle_deviation):
            past_steering_angle = steering_angle

        elif((past_steering_angle - steering_angle)>0):
            steering_angle = past_steering_angle - max_angle_deviation
            past_steering_angle = steering_angle
            
        else:
            steering_angle = past_steering_angle + max_angle_deviation
            past_steering_angle = steering_angle
        print('Right',int(-steering_angle))

 
    steering_angle = int(steering_angle)
    steering_angle = str(steering_angle) + ' '
    #print(steering_angle)
    arduino_steering_angle.write(steering_angle.encode())
    

    mimg = cv2.addWeighted(src, 1, ccan, 1, 0)
    
    return mimg

def transmitting_angle_distance():
    stop = 'a'
    stop = str(stop)
    print('stop')
    arduino_angle_distance.write(stop.encode())
    
    top_index , bottom_index = Searching_index()
    top_theta = Calculation_theta(top_index)
    bottom_theta = Calculation_theta(bottom_index)
    sleep(3)
    
    setServoPos(top_theta)
    sleep(1)
    top_distance = read_tfluna_data()
    
    if(top_distance < 100):
        top_distance = '0'+str(int(top_distance))
    else:
        top_distance = str(int(top_distance))
    
    imformation = str(int(top_theta))+ top_distance 
    setServoPos(0)
    sleep(2)
    
    setServoPos(bottom_theta)
    sleep(2)
    bottom_distance =  read_tfluna_data()
    
    if(bottom_distance < 100):
        bottom_distance = '0'+str(int(bottom_distance))
    else:
        bottom_distance = str(int(bottom_distance))
    
    imformation = imformation + str(int(bottom_theta)) + bottom_distance + '\n'
    arduino_angle_distance.write(imformation.encode())
    print(imformation)
    
    setServoPos(0)
    sleep(3)

while cap.isOpened():

    try: 
        ret,frame = cap.read()

        if ret:
            cv2.imshow('ImageWindow', DetectLineSlope(frame)) #차선 감지를 표시한 카메라 
            DetectLineSlope(frame)
            plot_update() # update plot

            if (Temp_max-Temp_min > 6.5):
                transmitting_angle_distance()
                
    except:
        print('error')
        continue
    key =  cv2.waitKey(23)
    if key & 0xff == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
