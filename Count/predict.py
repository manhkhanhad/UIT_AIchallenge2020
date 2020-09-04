import numpy as np
import math
from shapely.geometry import Polygon
IMG_HEIGHT = 720 
IMG_WIDTH = 1280 

class predictROI:
      
      """
      #KIEM TRA OBJECT CO TRONG MOT KHONG
      def DienTichTamGiac(self,x,y,z):
            a = math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
            b = math.sqrt((y[0] - z[0])**2 + (y[1] - z[1])**2)
            c = math.sqrt((z[0] - x[0])**2 + (z[1] - x[1])**2)
            p = (a+b+c)/2
            return math.sqrt(p*(p-a)*(p-b)*(p-c))

      def DienTichDaGiac(self):
            s = 0
            self.region.astype(int)
            for i in range(-1,self.region.shape[0]-1):
                  s += (self.region[i][0] - self.region[i+1][0])*((self.region[i][1] + self.region[i+1][1]))
            return abs(s/2)

      def is_in_region(self,x,y):
            Tong_tam_giac = 0
            point = np.array([x,y])
            for i in range(0,self.region.shape[0]-1):
                  Tong_tam_giac += self.DienTichTamGiac(self.region[i],self.region[i+1],point)

            Tong_tam_giac += self.DienTichTamGiac(self.region[0],self.region[-1],point)

            epsilon = 10**-6
            if (Tong_tam_giac - self.DienTichDaGiac()) <= epsilon:
                  return True
            else:
                  return False
      #-----------------------------------------------
      """

      

      def __init__(self, num, data, region):
            self.data = data
            self.currentFrame = num
            self.region = region
            #ROI = np.array([2, 1], [7, 1], [7, 4], [2, 4])
      
      def is_in_region(self,x_cen,y_cen):
            width = self.data[-1][3]
            height = self.data[-1][4]
            x_min = x_cen - width/2
            x_max = x_cen + width/2
            y_min = y_cen - height/2
            y_max = y_cen + height/2
            print(x_min,y_min,x_min,y_max,x_max,y_max,x_max,y_min)
            bbox = Polygon([(x_min,y_min),(x_min,y_max),(x_max,y_max),(x_max,y_min)])
            pts = []
            for point in self.region:
                  pts.append((point[0],point[1]))
            roi = Polygon(pts)
            if bbox.intersects(roi):
                  return True
            else:
                  return False
      
      def Euclid_distance(self, x, y, a, b):
            return math.sqrt(((a - x)**2 + (b - y)**2))
      
      #đạo hàm tìm vận tốc và gia tốc 
      def deritivate(self):
            v = np.zeros(len(self.data) - 1)

            t = np.zeros(len(self.data) - 1)
            
            for i in range(1, len(self.data)):
                  t[i-1] = self.data[i][0] - self.data [i-1][0]

            for i in range(1, len(self.data)):
                  v[i-1] = self.Euclid_distance(self.data[i][1], self.data[i][2], self.data[i-1][1], self.data[i-1][2]) / t[i-1]

            a = np.zeros(len(v) - 1)
            
            for i in range(1, len(v)):
                  a[i-1] = (v[i] - v[i-1]) / (t[i])
            
            capacity = 10
            if (len(a) < capacity):
                  capacity = 1
            if len(a) == 0:
                  return None,None
            return np.mean(v[-capacity:]), np.mean(a[-capacity:])    
      
      def predictTime(self):
            l, r = 0, 100
            X,Y = None,None
            vectorX = self.data[-1][1] - self.data[-2][1]
            vectorY = self.data[-1][2] - self.data[-2][2]

            gcd = math.sqrt(vectorX ** 2 + vectorY ** 2)
            if gcd == 0:
                  return None,None,None
            vectorX = vectorX / gcd
            vectorY = vectorY / gcd

            #vận tốc trung bình và gia tốc 
            v, a = self.deritivate()

            #print(v, a, vectorX, vectorY)

            #ans là thời gian để ra khỏi ROI
            ans = -1

            while l <= r:
                  time = (l + r) // 2
                  
                  #công thức tính toạ độ (x, y) xuất hiện cuối cùng 
                  if a == None:
                       return None,None,None 
                  x = self.data[-1, 1] + vectorX * abs(v * time + 1/2 * a * time * time)
                  y = self.data[-1, 2] + vectorY * abs(v * time + 1/2 * a * time * time)

                  #print(l, r, time, x, y)
                  
                  #nếu (x, y) nằm trong ROI thì cần thêm thời gian để ra khỏi ROI
                  #nếu (x, y) ra khỏi ROI cần thời gian ít hơn để ra khỏi ROI
                  if (self.is_in_region(x,y)):
                        l = time + 1
                  else: 
                        ans = time
                        X, Y = x, y
                        r = time - 1
                  #print(l, r)
            
            #print(X, Y)
            if X == None or not(0<X<IMG_WIDTH and 0<Y<IMG_HEIGHT): 
                  return None,None,None
            return ans + self.currentFrame,X,Y
#lịch sử các điểm và mất track tại điểm cuối
a = np.array([[46, 686, 255], [47, 686, 255]])
region = np.array([[1,566],[484,170],[756,164],[910,719],[0,718]])
#predict tại frame thứ 50 với bộ lịch sử a
test = predictROI(50, a, region)