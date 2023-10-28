import numpy as np 
import matplotlib.pyplot as plt

# ابعاد ماتریس
n = 5  # تعداد سطرها یا ستون‌ها

# ایجاد ماتریس اول (متقارن نسبت به y=x)
matrix1 = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i == j:
            matrix1[i, j] = i  # عنصر متقارن نسبت به خط y=x

# ایجاد ماتریس دوم (متقارن نسبت به خط y=-x)
matrix2 = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i == n - j - 1:
            matrix2[i, j] = i  # عنصر متقارن نسبت به خط y=-x

# تبدیل ماتریس‌ها به آرایه‌های یک بعدی
matrix1_flat = matrix1.flatten()
matrix2_flat = matrix2.flatten()

# نمایش ماتریس اول (متقارن نسبت به y=x) با scatter
plt.scatter(range(n**2), matrix1_flat)
plt.title('ماتریس اول (متقارن نسبت به y=x)')
plt.show()

# نمایش ماتریس دوم (متقارن نسبت به خط y=-x) با scatter
plt.scatter(range(n**2), matrix2_flat)
plt.title('ماتریس دوم (متقارن نسبت به y=-x)')
plt.show()
