import numpy as np
import pandas as pd
from collections import Counter
import time
class KNN:
    def __init__(self,K) :
        self.K=K
        self.X_train=None#dữ liệu huấn luyện
        self.Y_train=None#nhãn dữ liệu huấn luyện
    def duLieuHL(self, X, y):
        self.X_train = X#dữ liệu x
        self.y_train = y#nhãn y tương ứng với dữ liệu x
    @staticmethod
    #Hàm tính khoảng cách
    def khoangCach(x1,x2):
        return np.linalg.norm(x1-x2)
    #Dự đoán nhãn dự liệu cho 1 điểm dữ liệu
    def predict(self, x):
        kc = [self.khoangCach(x, x_train) for x_train in self.X_train]
        top_idx = np.argsort(kc)[:self.K]#sắp xếp kc theo thứ tự tăng dần
        k_nearests = self.y_train[top_idx]#nhãn của k1 điểm dữ liệu gần nhất
        label = Counter(k_nearests).most_common(1)[0][0]#hàm most_common trả về phần tử xuất hiện nhiều nhất
        #(1)là lấy phần tử đầu tiên và duy nhất trong danh sách đó 
        #[0] Phần tử này là một tuple (cặp) gồm phần tử và số lần xuất hiện của nó.
        #[0] Bây giờ chúng ta đã có một tuple. [0] lần nữa được dùng để lấy phần tử đầu tiên của tuple đó, chính là phần tử xuất hiện nhiều nhất.
        return label#trả về điểm dữ liệu gần nhất
    #Dự đoán nhãn cho 1 tập điểm dữ liệu
    def predict_batch(self, X):
        y_pred = [self.predict(x) for x in X]
        return y_pred#danh sách dự đoán các nhãn
def train_test_split(X, y, test_size=0.3):
    indices = np.arange(X.shape[0])#Tạo một mảng gồm các chỉ số từ 0 đến số lượng hàng trong X.
    np.random.shuffle(indices)#Trộn ngẫu nhiên các chỉ số trong mảng indices.
    test_size = int(len(X) * test_size)#Tính toán số lượng mẫu dữ liệu cho tập kiểm tra dựa trên test_size.
    train_indices = indices[:-test_size]#ấy các chỉ số từ đầu mảng đến trước chỉ số của phần tử thứ test_size
    test_indices = indices[-test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)
start_time = time.  time()  # Lấy thời gian bắt đầu
data = pd.read_csv('Iris.csv')
# Lấy các đặc trưng và nhãn
X = data.iloc[:, :-1].values  # Lấy tất cả các cột trừ cột cuối cùng
y = data.iloc[:, -1].values   # Lấy cột cuối cùng
# print(X)
label_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
#np.unique để tìm ra giá trị duy nhất trong mảng
#Hamd enumerate để lấy chỉ số và giá trị
y = np.array([label_mapping[label] for label in y])#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)#phân chia thành 70-30
# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (70-30)
model = KNN(10)
model.duLieuHL(X_train, y_train)
y_pred = model.predict_batch(X_test)#Dự đoán nhãn cho tập dữ liệu x_test
acc = accuracy_score(y_test, y_pred)                                
end_time = time.time()  # Lấy thời gian kết thúc
print("Số phần tử test đúng:",np.sum(y_pred==y_test),"/",len(y_test))
print(f'Accuracy: {acc*100:.2f}',"%")
print(f'Time    : {end_time - start_time:.4f} seconds')