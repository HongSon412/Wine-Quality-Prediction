import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from PIL import Image


# Đọc dữ liệu
df = pd.read_csv("wine.csv")

# Tiền xử lý dữ liệu
df['quality_cat'] = df['quality'].astype('category').cat.codes
df1 = df.drop('quality',axis=1)
X = df1.drop('quality_cat',axis=1)
Y = df1['quality_cat']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Chuẩn hóa dữ liệu
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Huấn luyện mô hình Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Ứng dụng Streamlit
wine = Image.open("image12.jpg")
st.image(wine, width = 200)
st.title("Wine Quality Prediction")


# Tạo form nhập thông số
st.sidebar.header("Input Wine Parameters")
fixed_acidity = st.sidebar.number_input("Fixed Acidity", min_value=0.0, max_value=15.0, value=7.0)
volatile_acidity = st.sidebar.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.3)
citric_acid = st.sidebar.number_input("Citric Acid", min_value=0.0, max_value=1.5, value=0.3)
residual_sugar = st.sidebar.number_input("Residual Sugar", min_value=0.0, max_value=20.0, value=6.0)
chlorides = st.sidebar.number_input("Chlorides", min_value=0.0, max_value=0.1, value=0.045)
free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", min_value=0, max_value=100, value=30)
total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", min_value=0, max_value=400, value=115)
density = st.sidebar.number_input("Density", min_value=0.9, max_value=1.1, value=0.995)
pH = st.sidebar.number_input("pH", min_value=2.0, max_value=4.0, value=3.2)
sulphates = st.sidebar.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.6)
alcohol = st.sidebar.number_input("Alcohol", min_value=8.0, max_value=15.0, value=10.5)

# Dự đoán
if st.sidebar.button("Predict Quality"):
    input_data = np.array([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
    ]])
    input_data = sc.transform(input_data)  # Chuẩn hóa đầu vào
    prediction = lr.predict(input_data)
    result = "Good" if prediction[0] == 1 else "Bad"
    
    # Hiển thị kết quả với phong cách đẹp hơn
    st.subheader("Prediction Result:")
    if result == "Good":
        st.success("🌟 **The wine quality is GOOD!** 🌟")
        st.write("🍷 This wine has high potential for enjoyment.")
    else:
        st.warning("⚠️ **The wine quality is BAD!** ⚠️")
        st.write("💔 Unfortunately, this wine might not meet expectations.")
        
st.markdown("""**👈 Please input some value from the sidebar** to see some examples """)
