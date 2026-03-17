import time
import os
import cv2
import numpy as np

from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
app = Flask(__name__)
app.secret_key = 'kiosk_secret_key_123' # 加入這行：Session 必須要有加密金鑰才能運作
from flask_sqlalchemy import SQLAlchemy

# --- 1. 取得絕對路徑與基本設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, 'menu.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

upload_path = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = upload_path
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

# --- 2. 載入模型 (確保路徑為純英文) ---
yunet_path = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")
sface_path = os.path.join(BASE_DIR, "face_recognition_sface_2021dec.onnx")

detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

# --- 3. 資料庫模型 ---
class MenuItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Integer, nullable=False)
    description = db.Column(db.String(200))

# 會員模型：綁定姓名、電話與照片路徑
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    photo_path = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()

def get_face_feature(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    detector.setInputSize((img.shape[1], img.shape[0]))
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0: return None
    face_align = recognizer.alignCrop(img, faces[0])
    return recognizer.feature(face_align)

# --- 4. 路由設定 (保留原本的 admin 與 checkout) ---
COUPONS = {"WELCOME50": 50, "SAVE100": 100}

@app.route('/')
def customer_index():
    items = MenuItem.query.all()
    return render_template('customer.html', items=items)

@app.route('/checkout', methods=['POST'])
def checkout():
    total = int(request.form.get('total_price', 0))
    coupon_code = request.form.get('coupon_code', '').upper()
    discount = COUPONS.get(coupon_code, 0)
    final_amount = max(0, total - discount)
    
    # 取得當前使用者名稱，並在結帳後清除 Session (模擬 Kiosk 點餐完畢回到首頁)
    user_name = session.get('user_name', '顧客')
    session.pop('user_name', None)
    
    return f"<h1>下單成功，{user_name}！</h1><p>原價: {total}</p><p>折抵: {discount}</p><h2>實付金額: {final_amount}</h2><a href='/'>返回首頁</a>"

@app.route('/logout')
def logout():
    # 手動取消點餐，清除 Session
    session.pop('user_name', None)
    return redirect(url_for('customer_index'))
@app.route('/admin')
def admin_index():
    items = MenuItem.query.all()
    return render_template('admin.html', items=items)

@app.route('/admin/add', methods=['POST'])
def add_item():
    name = request.form.get('name')
    price = request.form.get('price')
    desc = request.form.get('description')
    if name and price:
        new_item = MenuItem(name=name, price=int(price), description=desc)
        db.session.add(new_item)
        db.session.commit()
    return redirect(url_for('admin_index'))

@app.route('/admin/delete/<int:id>')
def delete_item(id):
    item_to_delete = MenuItem.query.get_or_404(id)
    db.session.delete(item_to_delete)
    db.session.commit()
    return redirect(url_for('admin_index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '')
        phone = request.form.get('phone', '')
        action = request.form.get('action')
        captured_photo = request.form.get('captured_photo', '') # 接收隱藏欄位傳來的暫存檔名
        
        if not name or not phone:
            return "<h1>請填寫完整資料</h1><a href='/register'>返回</a>", 400

        # --- 分支 A：開啟相機，只拍照不寫入資料庫 ---
        if action == 'webcam':
            cap = cv2.VideoCapture(0)
            win_name = 'Auto Capture - Please look at the camera'
            cv2.namedWindow(win_name)
            
            # 建立暫存檔名
            temp_filename = f"temp_{phone}_{int(time.time())}.jpg"
            temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            success_capture = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                display_frame = frame.copy()
                cv2.putText(display_frame, "Detecting face...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                detector.setInputSize((frame.shape[1], frame.shape[0]))
                _, faces = detector.detect(frame)
                
                if faces is not None and len(faces) > 0:
                    face_align = recognizer.alignCrop(frame, faces[0])
                    feature = recognizer.feature(face_align)
                    
                    if feature is not None:
                        # 儲存暫存畫面
                        cv2.imwrite(temp_filepath, frame)
                        success_capture = True
                        
                        cv2.putText(display_frame, "Face Detected! Captured...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow(win_name, display_frame)
                        cv2.waitKey(1000) 
                        break

                cv2.imshow(win_name, display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1: break

            cap.release()
            cv2.destroyAllWindows()

            if not success_capture:
                return "<h1>相機已關閉或未偵測到人臉</h1><a href='/register'>返回重新註冊</a>", 400

            # 拍照成功：帶著資料與暫存照片返回註冊介面
            return render_template('register.html', name=name, phone=phone, captured_photo=temp_filename)

        # --- 分支 B：真正寫入資料庫 ---
        elif action == 'register':
            photo = request.files.get('photo')
            source_path = ""
            
            # 判斷使用者是選檔案，還是沿用剛拍好的照片
            if photo and photo.filename != '':
                # 處理上傳檔案
                ext = os.path.splitext(photo.filename)[1] or '.jpg'
                source_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_upload_{phone}{ext}")
                photo.save(source_path)
            elif captured_photo:
                # 沿用剛才相機拍的暫存檔
                source_path = os.path.join(app.config['UPLOAD_FOLDER'], captured_photo)
                if not os.path.exists(source_path):
                    return "<h1>找不到拍攝的照片，請重新操作！</h1><a href='/register'>返回</a>", 400
            else:
                return "<h1>請上傳照片或使用相機拍攝！</h1><a href='/register'>返回</a>", 400

            # 再次驗證特徵 (防呆)
            feature = get_face_feature(source_path)
            if feature is None:
                os.remove(source_path)
                return "<h1>照片中未偵測到人臉，請重新提供！</h1><a href='/register'>返回</a>", 400

            # 正式取得資料庫 ID 並建立 User
            new_user = User(name=name, phone=phone, photo_path="temp")
            db.session.add(new_user)
            db.session.flush() 

            # 將暫存檔案重新命名為正式檔名
            ext = os.path.splitext(source_path)[1]
            final_filename = f"member_{new_user.id}_{phone}{ext}"
            final_filepath = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
            
            # 搬移/重新命名檔案 (確保這裡只執行一次)
            os.rename(source_path, final_filepath)

            # 更新路徑並正式 Commit
            new_user.photo_path = final_filepath
            db.session.commit()
            
            # ================= 新增：註冊完自動登入 =================
            # 將剛註冊好的名字存入 Session，讓系統知道這位客人已登入
            session['user_name'] = new_user.name
            
            # 直接跳轉回首頁
            return redirect(url_for('customer_index'))

    # GET 請求：顯示空白表單
    return render_template('register.html', name='', phone='', captured_photo='')
# --- 5. 更新的人臉辨識登入邏輯 ---
@app.route('/face_login')
def face_login():
    users = User.query.all()
    whitelist = []
    
    # 建立動態白名單，綁定姓名與特徵
    for u in users:
        feat = get_face_feature(u.photo_path)
        if feat is not None:
            whitelist.append({"name": u.name, "feature": feat})
            
    if not whitelist:
        return "<h1>系統中尚未有任何有效的會員特徵，請先註冊或重新上傳照片！</h1><a href='/register'>前往註冊</a>", 400

    cap = cv2.VideoCapture(0)
    win_name = 'Face Login - Press Q to Exit'
    cv2.namedWindow(win_name)
    recognized_user = None
    login_success = False # 新增成功標記

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = detector.detect(frame)

        if faces is not None:
            for face in faces:
                face_align = recognizer.alignCrop(frame, face)
                feature = recognizer.feature(face_align)
                
                best_score = 0
                best_name = "Unknown"
                
                # 與資料庫內的會員進行特徵比對
                for w_user in whitelist:
                    score = recognizer.match(w_user["feature"], feature, cv2.FaceRecognizerSF_FR_COSINE)
                    if score > best_score:
                        best_score = score
                        if score > 0.36: # 餘弦相似度閾值
                            best_name = w_user["name"]

                coords = face[:-1].astype(np.int32)
                
                # 判斷是否登入成功並繪製畫面
                if best_name != "Unknown":
                    recognized_user = best_name
                    login_success = True
                    color = (0, 255, 0) # 綠框
                    cv2.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), color, 2)
                    # 顯示名字與成功訊息 (使用英文避免亂碼)
                    cv2.putText(frame, f"{best_name} - Login Success!", (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    break # 找到匹配者，跳出人臉偵測迴圈
                else:
                    color = (0, 0, 255) # 紅框
                    cv2.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), color, 2)
                    cv2.putText(frame, f"Unknown: {best_score:.2f}", (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow(win_name, frame)

        # 成功辨識後的自動中斷處理
        if login_success:
            cv2.waitKey(1500) # 畫面停留 1.5 秒讓使用者確認
            break # 跳出相機讀取迴圈

        # 手動退出機制
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1: break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()
    
    # 根據結果渲染網頁
# ... (前面的 cap.release() 與 cv2.destroyAllWindows() 保留) ...
    
    # 根據結果處理登入狀態
    if recognized_user:
        # 登入成功，將名字存入 session 並導向首頁
        session['user_name'] = recognized_user
        return redirect(url_for('customer_index'))
    else:
        return "<h1>未能辨識身份或已取消</h1><a href='/'>返回首頁</a>"
if __name__ == '__main__':
    app.run(debug=True)