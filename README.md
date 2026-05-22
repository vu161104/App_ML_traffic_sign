# 🚦 Nhận Diện Biển Báo Giao Thông

Phân loại biển báo giao thông Việt Nam sử dụng **HOG + SVM**.

## Cấu trúc

```
├── app.py                          # Flask API
├── index.html                      # Giao diện web
├── hog_svm_model.pkl               # Model đã huấn luyện
├── HOG_SVM_TrafficSign_v3 (2).ipynb  # Notebook huấn luyện
├── requirements.txt
├── Procfile
└── .python-version
```

## Cài đặt & Chạy

```bash
pip install -r requirements.txt
python app.py
```

Truy cập: http://127.0.0.1:5000

## API

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/` | Giao diện web |
| GET | `/health` | Kiểm tra server |
| POST | `/predict` | Dự đoán từ ảnh (`form-data: image`) |
| GET | `/labels` | Danh sách nhãn |

## 10 Lớp biển báo

| ID | Biển báo |
|----|---------|
| 0 | Ưu tiên tại giao lộ |
| 1 | Đường ưu tiên |
| 2 | Nhường đường |
| 3 | Dừng lại (Stop) |
| 4 | Bắt buộc rẽ phải |
| 5 | Bắt buộc rẽ trái |
| 6 | Đi thẳng |
| 7 | Đi thẳng hoặc rẽ phải |
| 8 | Đi thẳng hoặc rẽ trái |
| 9 | Vòng xuyến |

## Demo

https://drive.google.com/file/d/1RwiFA8OIsDp1ncK83_pbzBJxiOwwJHnp/view?usp=sharing
