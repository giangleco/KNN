Ứng dụng thuật toán K-nearst neighbors trên bộ dữ liệu iris(bộ dữ liêu các loài hòa)
I.Giới thiệu chung:
1.Khái niệm
 K-nearest neighbor là một trong những thuật toán supervised-learning đơn giản nhất (mà hiệu quả trong một vài trường hợp) trong Machine Learning. Khi training, thuật toán này không học một điều gì từ dữ liệu training (đây cũng là lý do thuật toán này được xếp vào loại lazy learning), mọi tính toán được thực hiện khi nó cần dự đoán kết quả của dữ liệu mới. K-nearest neighbor có thể áp dụng được vào cả hai loại của bài toán Supervised learning là Classification(phân loại) và Regression(hồi quy). 
2.Ứng dụng trên bộ dữ liệu iris
    Phân loại 3 loài hoa Iris-setosa,Iris-versicolor Iris-virginica dựa trên các thuộc tính SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
 ![alt text](image.png)
 II.Cài đặt
 1.Cài đặt môi trường ảo
  python -m venv venv
  source venv/bin/activate  # Trên macOS/Linux
  venv\Scripts\activate     # Trên Window
  2.Cài đặt các thư viện cần thiết
  -pip install numpy
  -pip install pandas
 3.Các bược thực hiện
 *Để thực hiện bài toán kNN cần 6 bước chính như sau:
    B1. Ta có D là tập các điểm dữ liệu đã được gắn nhãn và A là dữ liệu chưa được phân loại.
    B2. Đo khoảng cách (Euclidean, Manhattan, Minkowski, Minkowski hoặc Trọng số) từ dữ liệu mới A đến tất cả các dữ liệu khác -đã được phân loại trong D.
    B3. Chọn k (k là tham số mà bạn định nghĩa) khoảng cách nhỏ nhất.
    B4. Kiểm tra danh sách các lớp có khoảng cách ngắn nhất và đếm số lượng của mỗi lớp xuất hiện.
    B5. Lấy đúng lớp (lớp xuất hiện nhiều lần nhất).
    B6. Lớp của dữ liệu mới là lớp mà bạn đã nhận được ở bước 5
4.Các công thức tính khoảng cách
![alt text](image-1.png)
    • Euclidean: Phù hợp khi các chiều của dữ liệu có ý nghĩa tương đương và không có cấu trúc đặc biệt. 
    • Manhattan: Phù hợp khi các chiều của dữ liệu có ý nghĩa khác nhau hoặc khi có cấu trúc mạng lưới trong dữ liệu. 
    • Minkowski: Cung cấp một sự linh hoạt cao, cho phép điều chỉnh độ nhạy cảm của khoảng cách đối với các chiều khác nhau của dữ liệu.
III.Các khởi động chương trình
-Chạy file OOP_KNN.py 
python3 OOP_KNN.py
-chạy file KNN.py
python3 KKN.py
IV. Hỗ trợ
1.các công cụ hỗ trợ
-chat gpt(https://chatgpt.com/)
-gemini(https://gemini.google.com/app?hl=vi)
-youtube
2.Tài liệu tham khảo
-https://machinelearningcoban.com/2017/01/08/knn/
-https://codelearn.io/sharing/thuat-toan-k-nearest-neighbors-knn
-https://viblo.asia/p/knn-k-nearest-neighbors-1-djeZ14ejKWz