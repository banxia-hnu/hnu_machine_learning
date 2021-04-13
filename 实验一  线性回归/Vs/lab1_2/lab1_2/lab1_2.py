import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def LoadFile(filename):
    data = np.loadtxt(filename, delimiter=',', unpack=True, usecols=(0, 1, 2))
    x = np.transpose(np.array(data[0]))
    y = np.transpose(np.array(data[1]))
    z = np.transpose(np.array(data[2]))
    return x, y, z

if __name__ == '__main__':
    area, room, price = LoadFile('ex1data2.txt')
    num=len(area) #样本数量
    x0=np.ones(num)
    #归一化处理，这里使用线性归一化
    x1=(area-np.average(area))/(area.max()-area.min())
    x2=(room-np.average(room))/(room.max()-room.min())
    #堆叠属性数组，构造属性矩阵
    #从(16,)到(16,3),因为新出现的轴是第二个轴所以axis为1
    X=np.stack((x0,x1,x2),axis=1)
    #print(X)
    #得到形状为一列的数组
    Y=price.reshape(-1,1)
    #print(Y)
    learn_rate=0.001    #设置超参数
    iter=1500   #迭代次数
    display_step=50    #每50次迭代显示一下效果
    
    #设置模型参数初始值
    W=[[0],
       [0],
       [0]]
    #训练模型
    mse=[]
    for i in range(0,iter+1):
        #求偏导
        dL_dW=np.matmul(np.transpose(X),np.matmul(X,W)-Y)   #XT(XW-Y)
        #更新模型参数
        W=W-learn_rate*dL_dW
        #得到估计值
        PRED=np.matmul(X,W)
        #计算损失(均方误差)
        Loss=np.mean(np.square(Y-PRED))/2
        mse.append(Loss)
        if i % display_step==0:
            print("i:%i,Loss:%f"%(i,mse[i]))
    xx0=np.ones(1)
    xx1=(1650.0-np.average(area))/(area.max()-area.min())
    xx2=(3.0-np.average(room))/(room.max()-room.min())
    XX=[xx0,xx1,xx2]
    print("房屋面积为1650平方英尺房间数量为3时预测房屋的价格:%f"%(np.matmul(XX,W)))   
    
    #结果可视化
    plt.rcParams['font.sans-serif'] =['SimHei']
    plt.figure(figsize=(12,4))
    #损失变化可视化
    plt.subplot(1,2,1)
    plt.plot(mse)
    plt.xlabel("迭代次数",fontsize=14)
    plt.ylabel("损失值",fontsize=14)
    #估计值与标签值比较可视化
    plt.subplot(1,2,2)
    PRED=PRED.reshape(-1)
    plt.plot(price,color="red",marker="o",label="数据集")
    plt.plot(PRED,color="blue",marker="o",label="预测房价")
    plt.xlabel("sample",fontsize=14)
    plt.ylabel("price",fontsize=14)
    plt.legend()
    plt.show()
    
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(area, room, price, color="red")
    ax.set_zlabel('price', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('room', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('area', fontdict={'size': 15, 'color': 'red'})
    ax.scatter(area, room, PRED,color="b")
    XX, YY = np.meshgrid(area, room)
    ax.plot_surface(XX,
                    YY,
                    Z=W[:,0][0]*x0+W[:,0][1]*((XX-np.average(area))/(area.max()-area.min()))
                       +W[:,0][2]*((YY-np.average(room))/(room.max()-room.min())),
                    color='g',
                    alpha=0.9
                   )
   
    plt.show()
 
    
    