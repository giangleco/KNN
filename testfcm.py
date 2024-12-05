import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class FuzzyCMeans:
    def __init__(self, n_clusters=3, m=2, max_iter=150, error=1e-5):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.centers = None
        self.U = None

    def fit(self, X):
        N = X.shape[0]
        C = self.n_clusters

        # Khởi tạo ngẫu nhiên ma trận mức độ thành viên U
        U = np.random.dirichlet(np.ones(C), size=N)

        for iteration in range(self.max_iter):
            U_old = U.copy()

            # Tính toán các tâm cụm
            centers = self._calculate_centers(X, U)

            # Cập nhật ma trận mức độ thành viên U
            U = self._update_U(X, centers)

            # Kiểm tra điều kiện hội tụ
            if np.linalg.norm(U - U_old) < self.error:
                break

        self.centers = centers
        self.U = U

    def _calculate_centers(self, X, U):
        um = U ** self.m
        return (um.T @ X) / um.sum(axis=0)[:, None]

    def _update_U(self, X, centers):
        power = 2 / (self.m - 1)
        temp = np.zeros((X.shape[0], self.n_clusters))

        for i, x in enumerate(X):
            for j, c in enumerate(centers):
                temp[i, j] = np.linalg.norm(x - c)

        temp = temp ** power
        denominator = temp.sum(axis=1)[:, None]
        return 1 / (temp / denominator)

    def predict(self, X):
        return np.argmax(self.U, axis=1)
# Đọc dữ liệu từ file Iris.csv
data = pd.read_csv('Iris.csv')
X = data.iloc[:, :-1].values  # Lấy tất cả các cột trừ cột cuối cùng

# Tạo đối tượng FCM và huấn luyện mô hình
fcm = FuzzyCMeans(n_clusters=3)
fcm.fit(X)

# Dự đoán cụm cho các điểm dữ liệu
labels = fcm.predict(X)

# Hiển thị các cụm
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(fcm.centers[:, 0], fcm.centers[:, 1], marker='x', s=200, c='red')
plt.title('Fuzzy C-Means Clustering')
plt.show()
