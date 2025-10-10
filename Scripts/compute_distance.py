import numpy as np

# 定义两个点
P1 = np.array([6.64715, -71.5779, 9.82994])
P2 = np.array([5.57889, -63.5147, 10.1271])

# 计算欧氏距离
distance = np.linalg.norm(P2 - P1)

print("The Euclidean distance between the points is:", distance)
