# 手部触觉数据：指尖力矩阵 + 掌心压力矩阵

std_msgs/Header header

# finger_force 第 0 维对应的手指名，顺序与矩阵行一致
string[] finger_names

# 指尖力矩阵，layout.dim = [finger][row][col]
std_msgs/Float32MultiArray finger_force

# 掌心压力矩阵，layout.dim = [row][col]
std_msgs/Float32MultiArray palm_force
