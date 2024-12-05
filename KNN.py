import csv#Thư viện để làm việc với file csv dùng để lưu trữ dữ liệu dưới dạng bảng
import numpy as np
import math
import time 
#Hàm đọc file 
def loadData(path):
    f = open(path, "r")# mở file path chế độ mở chỉ đọc
    data = csv.reader(f)#đọc từng dòng của file csv thành các list
    data = np.array(list(data))#Chuyển dữ liệu từ file thành một mảng numpy
    data = np.delete(data, 0, 0)#xóa đi dòng [0,0]
    data = np.delete(data, 0, 1)#xóa đi dòng [0,1]
    np.random.shuffle(data)#Trộn ngẫu nhiên các hang dữ liệu 
    f.close()
    trainSet = data[:105]#lấy 100 dòng đầu tiên của mảng để làm tập huấn
    testSet = data[105:]#lấy các dữ liệu còn lại để làm tập kiểm tra
    return trainSet, testSet
#Hàm để tính khoảng cách giữa 2 điểm dữ liệu
def khoangCach(pointA, pointB, numOfFeature=4):#pointA, pointB là 2 điểm dữ liệu đầu vào
    tmp = 0
    for i in range(numOfFeature):#numOffeature tượng trung cho 4 thuộc tính của loài hoa đó 
        #SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,(chiều dài, rộng cánh đài, chiều dài, rộng cánh hoa)
        tmp += (float(pointA[i]) - float(pointB[i])) ** 2
    return math.sqrt(tmp)#tính khoảng cách theo euclidean 
#Hàm thực hiện thuật toán K-Nearest Neighbor. 
# Thuật toán này tìm k điểm dữ liệu gần nhất với một điểm dữ liệu mới và dự đoán nhãn của điểm mới dựa trên nhãn của k điểm gần nhất đó.
def KNN(trainSet, point, k):
    distances = []
    for item in trainSet:
        distances.append({ # áp dụng numpy
            "label": item[-1],
            "value": khoangCach(item, point)#trả về  khoảng cách của điểm dữ liệu cần xét đến các điểm trong trainset
        })
    distances.sort(key=lambda x: x["value"])#Sắp xếp các điểm dữ liệu đó theo thứ tự tăng dần 
    labels = [item["label"] for item in distances]#Lấy các label từ danh sách distances 
    return labels[:k]
#Hàm tìm K điểm dữ liệu gần nhất với điểm dữ liệu cần dự đoán
def timKDiemGanNhat(arr):
    labels = set(arr)
    ans = ""
    maxOccur = 0
    for label in labels:
        num = arr.count(label)#Đếm số nhãn trong danh sách dùng hàm count
        if num > maxOccur:
            maxOccur = num
            ans = label
    return ans# trả về số nhãn xuất hiện nhiều nhất

start_time = time.time()  # Lấy thời gian bắt đầu
if __name__ == "__main__":
    trainSet, testSet = loadData("./Iris.csv")#đoc file 
    # print(testSet)
    check = 0
    for test in testSet:
        knn = KNN(trainSet, test, 10)#trả về 10 label điểm gần nhất 
        answer = timKDiemGanNhat(knn)#trả về label xuât hiện nhiều nhất
        # if(test[-1]==answer):#test[-1] là lấy cột phần tử label
        #     check+=1
        check += test[-1] == answer #nếu label này bằng test thì check +=1                                    
        # print("label: {} -> predicted: {}".format(item[-1], answer))
    end_time = time.time()  # Lấy thời gian kết thúc
    print("Số phần tử test đúng: ",check,"/",len(testSet))
    print(f'Accuracy: {(check/len(testSet))*100:.2f}',"%")#chỉ số test đúng/ tổng chỉ số test
    print(f'Time    : {end_time - start_time:.4f} seconds')
