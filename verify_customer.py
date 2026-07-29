import os
os.chdir(r'c:\Users\e1000\Projrct')
from app import app

client = app.test_client()
with client.session_transaction() as s:
    s['user_name'] = '123'

resp = client.get('/')
html = resp.get_data(as_text=True)
assert resp.status_code == 200, resp.status_code
assert 'id="modifierModal"' in html, '缺少 modifier modal'
assert 'handleAddToCart(this)' in html, '缺少加入購物車 handler'
assert '.form-control, .form-select' in html, '缺少深色主題表單樣式'
assert '客製化選擇' in html, '缺少客製化彈窗內容'
assert '購物車' in html, '未呈現已登入 customer 頁面'
print('customer template checks passed')
