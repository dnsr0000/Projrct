import time
import os
import cv2
import numpy as np
from datetime import datetime
from sqlalchemy import text

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import io
import csv

app = Flask(__name__)
app.secret_key = 'kiosk_secret_key_123'

# Flask 應用程式初始化
# app.secret_key 用於 session 加密

# --- 1. 取得絕對路徑與基本設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, 'menu.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ================= 新增：分類上傳資料夾 =================
menu_path = os.path.join(BASE_DIR, 'static', 'menu')
member_path = os.path.join(BASE_DIR, 'static', 'member')
app.config['UPLOAD_FOLDER_MENU'] = menu_path
app.config['UPLOAD_FOLDER_MEMBER'] = member_path

# 確保兩個資料夾都存在
os.makedirs(app.config['UPLOAD_FOLDER_MENU'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER_MEMBER'], exist_ok=True)
# =======================================================

db = SQLAlchemy(app)

# --- 2. 載入模型 ---
yunet_path = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")
sface_path = os.path.join(BASE_DIR, "face_recognition_sface_2021dec.onnx")

detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

# --- 3. 資料庫模型 ---
# MenuItem: 餐點資料表
# User: 會員資料表（包含人臉照片路徑）
# Order + OrderItem: 訂單與訂單細項
class MenuItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Integer, nullable=False)
    description = db.Column(db.String(200))
    category = db.Column(db.String(80), nullable=False, default='未分類')
    is_recommended = db.Column(db.Boolean, default=False)
    is_new = db.Column(db.Boolean, default=False)
    image_path = db.Column(db.String(200), nullable=True)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    photo_path = db.Column(db.String(200), nullable=False)
    points = db.Column(db.Integer, default=0)

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    table_number = db.Column(db.String(50))
    total_price = db.Column(db.Integer, nullable=False)
    payment_method = db.Column(db.String(50))
    status = db.Column(db.String(20), default='Pending')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    items = db.relationship('OrderItem', backref='order', lazy=True)

class OrderItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('order.id'), nullable=False)
    item_name = db.Column(db.String(100), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Integer, nullable=False)

class Setting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.String(200), nullable=False)

# 助手函數：後台設定參數存取

def get_setting(key, default=None):
    s = Setting.query.filter_by(name=key).first()
    if s:
        return s.value
    if default is not None:
        set_setting(key, default)
        return default
    return None


def set_setting(key, value):
    s = Setting.query.filter_by(name=key).first()
    if s:
        s.value = str(value)
    else:
        s = Setting(name=key, value=str(value))
        db.session.add(s)
    db.session.commit()

with app.app_context():
    db.create_all()
    insp = db.inspect(db.engine)
    if 'menu_item' in insp.get_table_names():
        cols = [c['name'] for c in insp.get_columns('menu_item')]
        with db.engine.begin() as conn:
            if 'category' not in cols:
                conn.execute(text("ALTER TABLE menu_item ADD COLUMN category VARCHAR(80) DEFAULT '未分類'"))
            if 'is_recommended' not in cols:
                conn.execute(text("ALTER TABLE menu_item ADD COLUMN is_recommended BOOLEAN DEFAULT 0"))
            if 'is_new' not in cols:
                conn.execute(text("ALTER TABLE menu_item ADD COLUMN is_new BOOLEAN DEFAULT 0"))
    if 'setting' in insp.get_table_names():
        cols = [c['name'] for c in insp.get_columns('setting')]
        with db.engine.begin() as conn:
            if 'name' not in cols:
                conn.execute(text("ALTER TABLE setting ADD COLUMN name VARCHAR(100) DEFAULT ''"))
            if 'value' not in cols:
                conn.execute(text("ALTER TABLE setting ADD COLUMN value VARCHAR(200) DEFAULT ''"))

    if 'user' in insp.get_table_names():
        cols = [c['name'] for c in insp.get_columns('user')]
        with db.engine.begin() as conn:
            if 'points' not in cols:
                conn.execute(text("ALTER TABLE user ADD COLUMN points INTEGER DEFAULT 0"))

# --- 輔助函數 ---
# 讀取圖片，進行人臉偵測並回傳特徵向量
# 用於人臉登入 / 註冊比對

def get_face_feature(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    detector.setInputSize((img.shape[1], img.shape[0]))
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0: return None
    face_align = recognizer.alignCrop(img, faces[0])
    return recognizer.feature(face_align)

# --- 店家後台管理 (整合登入與管理畫面) ---
# 需要 admin 登入，POST 提交時檢查帳密（簡單示例）
@app.route('/admin', methods=['GET', 'POST'])
def admin_index():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == '1234' and password == '1234':
            session['admin_logged_in'] = True
            return redirect(url_for('admin_index'))
        else:
            return "<h1>帳號或密碼錯誤！</h1><a href='/admin'>重新登入</a>", 401

    if not session.get('admin_logged_in'):
        return render_template('admin.html')

    # 訂單搜尋與日期篩選
    status_filter = request.args.get('status', 'all')
    payment_filter = request.args.get('payment_method', 'all')
    search_term = request.args.get('search', '').strip()
    start_date = request.args.get('start_date', '').strip()
    end_date = request.args.get('end_date', '').strip()

    # 基本資料
    items = MenuItem.query.all()
    users = User.query.all()

    # 訂單查詢
    order_q = Order.query
    if status_filter != 'all':
        order_q = order_q.filter(Order.status == status_filter)
    if payment_filter != 'all':
        order_q = order_q.filter(Order.payment_method == payment_filter)

    if search_term:
        if search_term.isdigit():
            order_q = order_q.filter((Order.id == int(search_term)) | (Order.table_number.contains(search_term)))
        else:
            order_q = order_q.filter(Order.table_number.contains(search_term))

    if start_date:
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            order_q = order_q.filter(Order.created_at >= start_dt)
        except Exception:
            pass
    if end_date:
        try:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            order_q = order_q.filter(Order.created_at <= end_dt)
        except Exception:
            pass

    orders = order_q.order_by(Order.created_at.desc()).limit(200).all()

    # 營收報表（依分頁結果計算）
    revenue_orders = Order.query
    if start_date:
        try:
            revenue_orders = revenue_orders.filter(Order.created_at >= datetime.strptime(start_date, '%Y-%m-%d'))
        except Exception:
            pass
    if end_date:
        try:
            revenue_orders = revenue_orders.filter(Order.created_at <= datetime.strptime(end_date, '%Y-%m-%d'))
        except Exception:
            pass
    revenue_orders = revenue_orders.all()

    total_revenue = sum(o.total_price for o in revenue_orders)
    total_orders = len(revenue_orders)
    # 熱門品項統計
    item_counter = {}
    for o in revenue_orders:
        for item in o.items:
            item_counter[item.item_name] = item_counter.get(item.item_name, 0) + item.quantity
    top_items = sorted(item_counter.items(), key=lambda x: x[1], reverse=True)[:5]

    points_redemption_enabled = get_setting('points_redemption_enabled', '1')
    points_to_cash = get_setting('points_to_cash_ratio', '10')
    points_earning_enabled = get_setting('points_earning_enabled', '1')
    points_earning_rate = get_setting('points_earning_rate', '1')

    return render_template('admin.html', items=items, users=users, orders=orders,
                           status_filter=status_filter, payment_filter=payment_filter,
                           search_term=search_term, start_date=start_date, end_date=end_date,
                           total_revenue=total_revenue, total_orders=total_orders, top_items=top_items,
                           points_redemption_enabled=points_redemption_enabled,
                           points_to_cash=points_to_cash,
                           points_earning_enabled=points_earning_enabled,
                           points_earning_rate=points_earning_rate)

@app.route('/admin/edit/<int:item_id>', methods=['GET'])
# 顯示菜單編輯表單

def admin_edit_item(item_id):
    if not session.get('admin_logged_in'): return redirect(url_for('admin_index'))
    item = MenuItem.query.get_or_404(item_id)
    return render_template('admin_edit_item.html', item=item)

@app.route('/admin/update/<int:item_id>', methods=['POST'])
# 處理菜單編輯儲存，支援更新圖片上傳與標記

def admin_update_item(item_id):
    if not session.get('admin_logged_in'): return redirect(url_for('admin_index'))
    item = MenuItem.query.get_or_404(item_id)
    item.name = request.form.get('name', item.name)
    item.price = int(request.form.get('price', item.price))
    item.description = request.form.get('description', item.description)
    item.category = request.form.get('category', item.category)
    item.is_recommended = request.form.get('recommended') == 'on'
    item.is_new = request.form.get('is_new') == 'on'
    image = request.files.get('image')
    if image and image.filename:
        ext = os.path.splitext(image.filename)[1] or '.jpg'
        image_filename = f"menu_{item.id}_{int(time.time())}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER_MENU'], image_filename)
        image.save(filepath)
        if item.image_path:
            old_path = os.path.join(app.config['UPLOAD_FOLDER_MENU'], item.image_path)
            if os.path.exists(old_path): os.remove(old_path)
        item.image_path = image_filename
    db.session.commit()
    return redirect(url_for('admin_index'))

@app.route('/admin_logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('customer_index'))

# --- 新增菜單 (儲存至 menu 資料夾) ---
@app.route('/admin/add', methods=['POST'])
# 新增菜單項目，包含圖片、分類、推薦、新品

def add_item():
    if not session.get('admin_logged_in'): return redirect(url_for('admin_index'))
    
    name = request.form.get('name')
    price = request.form.get('price')
    desc = request.form.get('description')
    category = request.form.get('category', '未分類')
    recommended = request.form.get('recommended') == 'on'
    image = request.files.get('image')
    
    if name and price:
        image_filename = ""
        if image and image.filename != '':
            ext = os.path.splitext(image.filename)[1] or '.jpg'
            image_filename = f"menu_{int(time.time())}{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER_MENU'], image_filename)
            image.save(filepath)
            
        is_new = request.form.get('is_new') == 'on'
        new_item = MenuItem(name=name, price=int(price), description=desc, category=category, is_recommended=recommended, is_new=is_new, image_path=image_filename)
        db.session.add(new_item)
        db.session.commit()
        
    return redirect(url_for('admin_index'))

# --- 刪除菜單 (從 menu 資料夾刪除) ---
@app.route('/admin/delete/<int:id>')
# 刪除菜單項目，連帶刪除本地圖片檔案

def delete_item(id):
    if not session.get('admin_logged_in'): return redirect(url_for('admin_index'))
    item_to_delete = MenuItem.query.get_or_404(id)
    
    if item_to_delete.image_path:
        filepath = os.path.join(app.config['UPLOAD_FOLDER_MENU'], item_to_delete.image_path)
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"刪除餐點照片失敗: {e}")
            
    db.session.delete(item_to_delete)
    db.session.commit()
    return redirect(url_for('admin_index'))

@app.route('/admin/update_order_status/<int:order_id>', methods=['POST'])
# 更新訂單狀態（Pending / Completed / Cancelled）
def update_order_status(order_id):
    if not session.get('admin_logged_in'): return redirect(url_for('admin_index'))
    new_status = request.form.get('status', 'Pending')
    order = Order.query.get_or_404(order_id)
    order.status = new_status
    db.session.commit()
    return redirect(url_for('admin_index'))

@app.route('/admin/update_settings', methods=['POST'])
def admin_update_settings():
    if not session.get('admin_logged_in'): return redirect(url_for('admin_index'))
    redemption_enabled = request.form.get('points_redemption_enabled', '0')
    to_cash = request.form.get('points_to_cash_ratio', '10')
    earning_enabled = request.form.get('points_earning_enabled', '0')
    earning_rate = request.form.get('points_earning_rate', '1')
    set_setting('points_redemption_enabled', '1' if redemption_enabled == 'on' else '0')
    set_setting('points_to_cash_ratio', str(max(1, int(to_cash))))
    set_setting('points_earning_enabled', '1' if earning_enabled == 'on' else '0')
    set_setting('points_earning_rate', str(max(0, int(earning_rate))))
    return redirect(url_for('admin_index'))

@app.route('/admin/edit_order/<int:order_id>')
# 顯示訂單編輯表單

def admin_edit_order(order_id):
    if not session.get('admin_logged_in'): return redirect(url_for('admin_index'))
    order = Order.query.get_or_404(order_id)
    return render_template('admin_edit_order.html', order=order)

@app.route('/admin/update_order/<int:order_id>', methods=['POST'])
# 處理訂單編輯後，更新價格、品項、狀態，並回到後台列表

def admin_update_order(order_id):
    if not session.get('admin_logged_in'): return redirect(url_for('admin_index'))
    order = Order.query.get_or_404(order_id)
    item_names = request.form.getlist('item_name')
    quantities = request.form.getlist('quantity')
    prices = request.form.getlist('price')
    new_items = []
    total = 0
    for name, qty, price in zip(item_names, quantities, prices):
        if not name.strip():
            continue
        q = int(qty or 0)
        p = int(price or 0)
        if q <= 0 or p < 0:
            continue
        new_items.append((name.strip(), q, p))
        total += q * p

    # 刪除舊明細，重建新明細
    OrderItem.query.filter_by(order_id=order.id).delete()
    for name, q, p in new_items:
        db.session.add(OrderItem(order_id=order.id, item_name=name, quantity=q, price=p))

    order.total_price = total
    order.payment_method = request.form.get('payment_method', order.payment_method)
    order.status = request.form.get('status', order.status)
    db.session.commit()
    return redirect(url_for('admin_index', highlight_order=order.id))

@app.route('/admin/export_orders')
def export_orders():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_index'))

    orders = Order.query.order_by(Order.created_at.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['OrderID', 'MemberOrTable', 'TotalPrice', 'PaymentMethod', 'Status', 'CreatedAt', 'ItemName', 'Quantity', 'Price'])

    for order in orders:
        if order.items:
            for item in order.items:
                writer.writerow([order.id, order.table_number or '訪客', order.total_price, order.payment_method, order.status, order.created_at.strftime('%Y-%m-%d %H:%M:%S'), item.item_name, item.quantity, item.price])
        else:
            writer.writerow([order.id, order.table_number or '訪客', order.total_price, order.payment_method, order.status, order.created_at.strftime('%Y-%m-%d %H:%M:%S'), '', '', ''])

    response = Response(output.getvalue(), mimetype='text/csv')
    response.headers['Content-Disposition'] = 'attachment; filename=orders_export.csv'
    return response

# --- 刪除會員並銷毀照片 ---
@app.route('/admin/delete_user/<int:id>')
def delete_user(id):
    if not session.get('admin_logged_in'): return redirect(url_for('admin_index'))
    user_to_delete = User.query.get_or_404(id)
    
    try:
        # photo_path 本來就是存絕對路徑，所以直接刪除即可
        if os.path.exists(user_to_delete.photo_path):
            os.remove(user_to_delete.photo_path)
    except Exception as e:
        print(f"刪除照片失敗: {e}")
        
    db.session.delete(user_to_delete)
    db.session.commit()
    return redirect(url_for('admin_index'))

# --- 客戶首頁 ---
@app.route('/')
# 客戶首頁：顯示菜單與篩選功能

def customer_index():
    items = MenuItem.query.all()
    categories = sorted({item.category or '未分類' for item in items})
    is_member = bool(session.get('user_name'))
    user_points = 0
    points_to_cash = int(get_setting('points_to_cash_ratio', '10'))
    points_redemption_enabled = get_setting('points_redemption_enabled', '1') == '1'
    points_earning_enabled = get_setting('points_earning_enabled', '1') == '1'
    if is_member and session.get('user_id'):
        member = User.query.get(session.get('user_id'))
        user_points = member.points if member else 0
    return render_template('customer.html', items=items, categories=categories, is_member=is_member, user_points=user_points, points_to_cash=points_to_cash, points_redemption_enabled=points_redemption_enabled, points_earning_enabled=points_earning_enabled)

# --- 結帳與登出 ---
@app.route('/logout')
def logout():
    session.pop('user_name', None)
    return redirect(url_for('customer_index'))

@app.route('/submit_order', methods=['POST'])
# 客戶端送出訂單 API：接收 JSON 訂單、寫入訂單與訂單細項

def submit_order():
    data = request.json
    user_name = session.get('user_name', data.get('table_number', '一般顧客'))
    user_id = session.get('user_id')
    points_redemption_enabled = get_setting('points_redemption_enabled', '1') == '1'
    points_to_cash = int(get_setting('points_to_cash_ratio', '10'))
    points_earning_enabled = get_setting('points_earning_enabled', '1') == '1'
    points_earning_rate = int(get_setting('points_earning_rate', '1'))
    member_discount_enabled = get_setting('member_discount_enabled', '1') == '1'
    member_discount_rate = float(get_setting('member_discount_rate', '0.10'))

    use_points = int(data.get('use_points', 0)) if data.get('use_points') else 0
    total_price = data['total_price']
    discount_points = 0
    discount_amount = 0

    if points_redemption_enabled and user_id and use_points > 0:
        user = User.query.get(user_id)
        if user:
            allow_points = min(use_points, user.points)
            discount_amount = allow_points // points_to_cash
            discount_points = allow_points if discount_amount > 0 else 0
            total_price = max(0, total_price - discount_amount)
            user.points = user.points - discount_points

    if user_id and points_earning_enabled:
        user = User.query.get(user_id)
        if user:
            earn = (sum(item['price'] * item['quantity'] for item in data['items']) // 10) * points_earning_rate
            user.points += earn

    new_order = Order(
        table_number=user_name,
        total_price=total_price,
        payment_method=data.get('payment_method', 'Cash')
    )
    db.session.add(new_order)
    db.session.flush()

    order_total_for_points = 0
    for item in data['items']:
        order_item = OrderItem(
            order_id=new_order.id,
            item_name=item['name'],
            quantity=item['quantity'],
            price=item['price']
        )
        db.session.add(order_item)
        order_total_for_points += item['price'] * item['quantity']

    if user_id and points_earning_enabled:
        user = User.query.get(user_id)
        if user:
            earn = (order_total_for_points // 10) * points_earning_rate
            user.points += earn

    db.session.commit()
    # 不登出會員，保持單次登入
    return jsonify({'message': f'訂單已成功送出！感謝 {user_name} 的光臨。', 'order_id': new_order.id, 'discount': discount_amount, 'used_points': discount_points})

# --- 會員註冊 (儲存至 member 資料夾) ---
@app.route('/register', methods=['GET', 'POST'])
# 會員註冊：支援上傳照片或即時拍照後註冊

def register():
    if request.method == 'POST':
        name = request.form.get('name', '')
        phone = request.form.get('phone', '')
        action = request.form.get('action')
        captured_photo = request.form.get('captured_photo', '')
        
        if not name or not phone:
            return "<h1>請填寫完整資料</h1><a href='/register'>返回</a>", 400

        if action == 'webcam':
            cap = cv2.VideoCapture(0)
            win_name = 'Auto Capture - Please look at the camera'
            cv2.namedWindow(win_name)
            
            temp_filename = f"temp_{phone}_{int(time.time())}.jpg"
            # 路徑改為 UPLOAD_FOLDER_MEMBER
            temp_filepath = os.path.join(app.config['UPLOAD_FOLDER_MEMBER'], temp_filename)
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
            return render_template('register.html', name=name, phone=phone, captured_photo=temp_filename)

        elif action == 'register':
            photo = request.files.get('photo')
            source_path = ""
            
            if photo and photo.filename != '':
                ext = os.path.splitext(photo.filename)[1] or '.jpg'
                # 路徑改為 UPLOAD_FOLDER_MEMBER
                source_path = os.path.join(app.config['UPLOAD_FOLDER_MEMBER'], f"temp_upload_{phone}{ext}")
                photo.save(source_path)
            elif captured_photo:
                # 路徑改為 UPLOAD_FOLDER_MEMBER
                source_path = os.path.join(app.config['UPLOAD_FOLDER_MEMBER'], captured_photo)
                if not os.path.exists(source_path):
                    return "<h1>找不到拍攝的照片，請重新操作！</h1><a href='/register'>返回</a>", 400
            else:
                return "<h1>請上傳照片或使用相機拍攝！</h1><a href='/register'>返回</a>", 400

            feature = get_face_feature(source_path)
            if feature is None:
                os.remove(source_path)
                return "<h1>照片中未偵測到人臉，請重新提供！</h1><a href='/register'>返回</a>", 400

            existing_users = User.query.order_by(User.id).all()
            available_id = 1
            for u in existing_users:
                if u.id == available_id: available_id += 1
                else: break

            new_user = User(id=available_id, name=name, phone=phone, photo_path="temp")
            db.session.add(new_user)
            db.session.flush() 

            ext = os.path.splitext(source_path)[1]
            final_filename = f"member_{new_user.id}_{phone}{ext}"
            # 路徑改為 UPLOAD_FOLDER_MEMBER
            final_filepath = os.path.join(app.config['UPLOAD_FOLDER_MEMBER'], final_filename)
            
            os.rename(source_path, final_filepath)
            new_user.photo_path = final_filepath
            db.session.commit()
            
            session['user_name'] = new_user.name
            session['user_id'] = new_user.id
            return redirect(url_for('customer_index'))

    return render_template('register.html', name='', phone='', captured_photo='')

# --- 人臉登入 ---
@app.route('/face_login')
# 人臉登入：比對會員照片特徵，成功則設置 session 登入

def face_login():
    users = User.query.all()
    whitelist = []
    
    for u in users:
        feat = get_face_feature(u.photo_path)
        if feat is not None:
            whitelist.append({"name": u.name, "feature": feat})
            
    if not whitelist:
        return "<h1>系統中尚未有任何有效的會員特徵，請先註冊！</h1><a href='/register'>前往註冊</a>", 400

    cap = cv2.VideoCapture(0)
    win_name = 'Face Login - Press Q to Exit'
    cv2.namedWindow(win_name)
    recognized_user = None
    login_success = False

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
                
                for w_user in whitelist:
                    score = recognizer.match(w_user["feature"], feature, cv2.FaceRecognizerSF_FR_COSINE)
                    if score > best_score:
                        best_score = score
                        if score > 0.36:
                            best_name = w_user["name"]

                coords = face[:-1].astype(np.int32)
                if best_name != "Unknown":
                    recognized_user = best_name
                    login_success = True
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), color, 2)
                    cv2.putText(frame, f"{best_name} - Login Success!", (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    break 
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), color, 2)
                    cv2.putText(frame, f"Unknown: {best_score:.2f}", (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow(win_name, frame)
        if login_success:
            cv2.waitKey(1500)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1: break

    cap.release()
    cv2.destroyAllWindows()
    
    if recognized_user:
        user_obj = User.query.filter_by(name=recognized_user).first()
        if user_obj:
            session['user_name'] = user_obj.name
            session['user_id'] = user_obj.id
        else:
            session['user_name'] = recognized_user
        return redirect(url_for('customer_index'))
    else:
        return "<h1>未能辨識身份</h1><a href='/'>返回首頁</a>"

if __name__ == '__main__':
    app.run(debug=True)