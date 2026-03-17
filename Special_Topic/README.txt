網頁後面加/admin進入編輯菜單畫面

下一步計畫
1.給/admin加上密碼鎖
2.融合人臉辨識系統


架構圖:
Project Root/
├── app.py                                  (核心 Backend 邏輯)
├── menu.db                                 (SQLite Database)
├── face_detection_yunet_2023mar.onnx       (AI 模型：Face Detection)
├── face_recognition_sface_2021dec.onnx     (AI 模型：Face Recognition)
├── static/
│   └── uploads/                            (使用者上傳的照片儲存區)
└── templates/
    ├── admin.html                          (前端介面：後台管理)
    ├── customer.html                       (前端介面：Kiosk 點餐首頁)
    └── register.html                       (前端介面：註冊與上傳照片)