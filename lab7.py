import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

file_path = 'ITA105_Lab_7.csv'
df = pd.read_csv(file_path)

print("--- ĐANG THỰC HIỆN BÀI 1 ---")

num_cols = df.select_dtypes(include=[np.number]).columns
skew_series = df[num_cols].skew().sort_values(ascending=False, key=abs)
skew_df = pd.DataFrame({'Cột': skew_series.index, 'Skewness': skew_series.values})

print("\nTop 10 cột lệch nhất (hoặc toàn bộ nếu ít hơn 10):")
print(skew_df.head(10))

top_3_skewed = skew_df['Cột'].head(3).tolist()
plt.figure(figsize=(18, 5))
for i, col in enumerate(top_3_skewed):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Phân phối {col}\nSkewness: {df[col].skew():.2f}')
plt.tight_layout()
plt.show()

print("\nNhận xét Bài 1:")
print("- Các cột như LotArea có độ lệch cực cao do sự tồn tại của các bất động sản diện tích lớn (outliers).")
print("- Phân phối lệch phải khiến mô hình dễ bị sai lệch bởi các giá trị cực lớn.")
print("- Đề xuất: Sử dụng Log-transform cho LotArea/SalePrice và Yeo-Johnson cho các cột có giá trị âm.")

print("\n--- ĐANG THỰC HIỆN BÀI 2 ---")

pos_col1, pos_col2, neg_col = 'LotArea', 'SalePrice', 'NegSkewIncome'

results_2 = []

lot_log = np.log1p(df[pos_col1])
lot_bc, lmbda_lot = stats.boxcox(df[pos_col1])
yj_transformer = PowerTransformer(method='yeo-johnson')
lot_yj = yj_transformer.fit_transform(df[[pos_col1]]).flatten()

income_yj = yj_transformer.fit_transform(df[[neg_col]]).flatten()

print(f"\nSo sánh Skewness cho {pos_col1}:")
print(f"- Gốc: {df[pos_col1].skew():.4f}")
print(f"- Sau Log: {pd.Series(lot_log).skew():.4f}")
print(f"- Sau Box-Cox (λ={lmbda_lot:.2f}): {pd.Series(lot_bc).skew():.4f}")
print(f"- Sau Power (Y-J): {pd.Series(lot_yj).skew():.4f}")

plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1); sns.histplot(df[pos_col1], kde=True); plt.title(f"Gốc: {pos_col1}")
plt.subplot(2, 2, 2); sns.histplot(lot_yj, kde=True, color='green'); plt.title(f"Y-J Transformed: {pos_col1}")
plt.subplot(2, 2, 3); sns.histplot(df[neg_col], kde=True); plt.title(f"Gốc: {neg_col}")
plt.subplot(2, 2, 4); sns.histplot(income_yj, kde=True, color='orange'); plt.title(f"Y-J Transformed: {neg_col}")
plt.tight_layout()
plt.show()

print("\n--- ĐANG THỰC HIỆN BÀI 3 ---")

X = df.select_dtypes(include=[np.number]).drop(columns=['SalePrice'])
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_a = LinearRegression().fit(X_train, y_train)
pred_a = model_a.predict(X_test)
rmse_a = np.sqrt(mean_squared_error(y_test, pred_a))

y_train_log = np.log(y_train)
model_b = LinearRegression().fit(X_train, y_train_log)
pred_b_log = model_b.predict(X_test)
pred_b_real = np.exp(pred_b_log)
rmse_b = np.sqrt(mean_squared_error(y_test, pred_b_real))

pt_X = PowerTransformer(); pt_y = PowerTransformer()
X_train_pt = pt_X.fit_transform(X_train)
X_test_pt = pt_X.transform(X_test)
y_train_pt = pt_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
model_c = LinearRegression().fit(X_train_pt, y_train_pt)
pred_c_pt = model_c.predict(X_test_pt)
pred_c_real = pt_y.inverse_transform(pred_c_pt.reshape(-1,1)).flatten()
rmse_c = np.sqrt(mean_squared_error(y_test, pred_c_real))

print("\nKẾT QUẢ SO SÁNH MÔ HÌNH:")
results_3 = pd.DataFrame({
    'Mô hình': ['Gốc (A)', 'Log Target (B)', 'Power Trans (C)'],
    'RMSE (Giá trị thực)': [rmse_a, rmse_b, rmse_c]
})
print(results_3)

print("\n--- ĐANG THỰC HIỆN BÀI 4 ---")

df['log_price_index'] = np.log(df['SalePrice'])

print("\nINSIGHT NGHIỆP VỤ:")
print("1. Tại sao biến đổi? Để nhìn rõ các phân khúc giá rẻ và tầm trung mà không bị che lấp bởi các 'siêu biệt thự'.")
print("2. Metric 'log_price_index' giúp quản lý sự biến động giá theo tỷ lệ phần trăm thay vì giá trị tuyệt đối, phù hợp để so sánh các khu vực khác nhau.")
print("3. Khuyến nghị: Doanh nghiệp nên tập trung vào các khu vực có log_price_index ổn định để giảm rủi ro thị trường.")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1); sns.boxplot(x=df['SalePrice']); plt.title("Raw SalePrice (Nhiều Outliers)")
plt.subplot(1, 2, 2); sns.boxplot(x=df['log_price_index'], color='salmon'); plt.title("Log Price Index (Cân đối hơn)")
plt.show()

print("\n--- TẤT CẢ CÁC BÀI ĐÃ HOÀN THÀNH ---")