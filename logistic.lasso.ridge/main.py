import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from pprint import pprint

# # تولید داده‌های تصادفی
# np.random.seed(0)
# X = np.array([[i] for i in np.random.randint(low=10, size=10)])
# y = np.array([[i] for i in np.random.randint(low=10, size=10)])

# تولید داده‌های تصادفی
np.random.seed(0)
# X_1 = abs(np.random.rand(10, 1))
# y_1 = abs(X_1 > np.random.randn(10, 1))

# x_init = np.linspace(0, 1, 10)
# y_init = abs(np.random.uniform(-1, 1, 10))

X_1 = np.linspace(0, 1, 10)  # ایجاد یک دسته از اعداد صحیح متناوب بین 0 و 1
y_1 = X_1 + 0.2 * abs(np.random.uniform(-1, 1, 10)) + 0.2 # اضافه کردن نوفه تصادفی به داده‌ها

X_2 = X_1 + 0.1
y_2 = X_1 + 0.2 * abs(np.random.uniform(-1, 1, 10)) - 0.2 # اضافه کردن نوفه تصادفی به داده‌ها

print(y_1)
print(y_2)
# exit()

X = np.append(X_1, X_2)
y = np.append(y_1, y_2)

X = np.concatenate((X.reshape(-1, 1),), axis=1)
y = np.concatenate((y.reshape(-1, 1),), axis=1).ravel()



pprint(X)
print("===============\n")
pprint(y)
# exit()
# تعریف مدل‌ها
ridge_model = Ridge(alpha=1.0, copy_X=True, solver="svd")
lasso_model = Lasso(alpha=1.0, copy_X=True)
logistic_model = LogisticRegression()

# آموزش مدل‌ها
ridge_model.fit(X, y)
lasso_model.fit(X, y)
logistic_model.fit(X, (y > 0.5).astype(int))

# تولید خطوط تطابقی با y=x
x_range = np.linspace(0, 1, 100).reshape(-1, 1)
y_pred_ridge = ridge_model.predict(x_range)
y_pred_lasso = lasso_model.predict(x_range)
print(y_pred_lasso)
# exit()
# نمایش داده‌ها و خطوط تطابقی
plt.scatter(X, y, label="data")
plt.plot(x_range, x_range, 'k--', label="y=x", linewidth=2)
plt.plot(x_range, y_pred_ridge, 'g', label="Ridge")
plt.plot(x_range, y_pred_lasso, 'b', label="Lasso")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("رسم خطوط با Ridge و Lasso")
plt.show()

# logistic vis

# # نمایش داده‌ها و خطوط تطابقی
# plt.scatter(X, y_binary, label="data")
# plt.plot(x_range, y_pred, 'g', label="Logistic Regression")
# plt.legend()
# plt.xlabel("X")
# plt.ylabel("Y (Binary)")
# plt.title("رسم خط با Logistic Regression")
# plt.show()