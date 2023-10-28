import numpy as np 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# تعداد نمونه‌ها
n_samples = 20

# ایجاد داده‌های X به صورت تقریبی برابر با y=x
X = np.linspace(0, 1, n_samples).reshape(-1, 1)

# ایجاد نوفه تصادفی با توزیع نرمال
noise = np.random.normal(0, 0.1, (n_samples, 1))

# ایجاد داده‌های Y با توجه به تقارن نسبت به y=x و تبدیل به دسته‌های 0 و 1
y = X + noise

print(X, end="\n=============\n")
print(y)
y_binary = (y > 0.5).astype(int)

# تعریف مدل Logistic Regression و آموزش آن
logistic_model = LogisticRegression()
logistic_model.fit(X, y_binary)

# تولید خطوط تطابقی با y=x
x_range = np.linspace(0, 1, 100).reshape(-1, 1)
y_pred = logistic_model.predict(x_range)

# نمایش داده‌ها و خطوط تطابقی
plt.scatter(X, y_binary, label="data")
plt.plot(x_range, y_pred, 'g', label="Logistic Regression")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y (Binary)")
plt.title("linear Logistic Regression")
plt.show()
