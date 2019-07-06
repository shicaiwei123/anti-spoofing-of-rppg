import  numpy as  np

# list逆序访问
# import  numpy as  np
# a=[[1,2,3],[2,3,4]]
# a=np.array(a)
# b=a.shape[1]
# print(b)


# # 转recg类型
# import dlib
# faces=[[208,79,291,291]]
#
# #坐标
# coordinate=faces[0]
# x1=coordinate[0]
# y1=coordinate[1]
# x2=x1+coordinate[2]
# y2=y1+coordinate[3]
#
# #类型转变
# rect=dlib.rectangle(x1,y1,x2,y2)
# print(rect)


# #初始化高维数组
# a=[[[1]*3]*4]*5
# b=np.array(a)
#
# c=[[[2]*3]*4]*5
# d=np.array(c)
#
# e=np.vstack((b,c))
# print(e)
#
#
# print(b)

# 求相关
a=np.array([1,7,3,90,5])
b=np.array([5,6,7,8,9])
ab=np.array([a,b])
c=np.corrcoef(ab)
print(c)


