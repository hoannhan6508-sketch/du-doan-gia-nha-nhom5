import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Dự đoán giá nhà Việt Nam", page_icon="🏡", layout="centered")

FEATURES = ['Diện tích (m2)', 'Số lượng phòng ngủ', 'Tuổi']
TARGET = 'Giá trị nhà( tỷ đồng)'
DATA_PATH = Path(__file__).parent / 'Nhom5_GiaNhaVietNam_train.xlsx'

@st.cache_resource
def load_model():
    df = pd.read_excel(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = DecisionTreeRegressor(random_state=42, max_depth=4)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        'MAE': mean_absolute_error(y_test, pred),
        'MSE': mean_squared_error(y_test, pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
        'R²': r2_score(y_test, pred),
    }
    return model, metrics, len(df)

st.title('🏡 DỰ ĐOÁN GIÁ NHÀ VIỆT NAM')
st.write('Nhập diện tích, số phòng ngủ và tuổi nhà để ước tính giá trị căn nhà bằng mô hình Decision Tree Regression.')
st.markdown('---')

try:
    model, metrics, n_rows = load_model()
except Exception as e:
    st.error(f'Không thể tải dữ liệu huấn luyện: {e}')
    st.stop()

st.subheader('🏠 Thông tin căn nhà')
with st.form('house_form'):
    area = st.number_input('Diện tích (m²)', min_value=1.0, max_value=2000.0, value=100.0, step=5.0)
    bedrooms = st.number_input('Số lượng phòng ngủ', min_value=0, max_value=30, value=3, step=1)
    age = st.number_input('Tuổi nhà (năm)', min_value=0, max_value=200, value=10, step=1)
    submitted = st.form_submit_button('Dự đoán giá nhà', type='primary')

if submitted:
    sample = pd.DataFrame([[area, bedrooms, age]], columns=FEATURES)
    prediction = float(model.predict(sample)[0])
    st.success(f'Giá nhà ước tính: {prediction:.2f} tỷ đồng')
    st.caption('Kết quả là ước tính từ mô hình học máy, không phải giá giao dịch hay định giá chính thức.')

with st.expander('📊 Thông tin mô hình và kết quả đánh giá'):
    st.write('**Thuật toán:** Decision Tree Regression')
    st.write('**Tham số:** max_depth = 4, random_state = 42')
    st.write(f'**Tổng số mẫu:** {n_rows} căn nhà; chia train/test nội bộ theo tỷ lệ 80:20.')
    st.caption('Các chỉ số bên dưới được tính trên tập kiểm tra nội bộ, không phải tập kiểm tra độc lập 6 mẫu dùng trong phần báo cáo.')
    c1, c2 = st.columns(2)
    c1.metric('MAE', f"{metrics['MAE']:.4f}")
    c2.metric('MSE', f"{metrics['MSE']:.4f}")
    c1.metric('RMSE', f"{metrics['RMSE']:.4f}")
    c2.metric('R²', f"{metrics['R²']:.4f}")

st.info('Lưu ý: Dữ liệu hiện chỉ có 23 mẫu. Kết quả dự đoán mang tính tham khảo và phục vụ mục đích học tập, không thay thế định giá bất động sản chuyên nghiệp.')
st.caption('Đồ án: Phân tích và dự đoán giá nhà Việt Nam – Nhóm 5')
