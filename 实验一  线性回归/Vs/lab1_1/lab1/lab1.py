import numpy as np
import matplotlib.pyplot as plt 
#设置字体
plt.rcParams['font.sans-serif'] =['SimHei']


def LoadFile(filename):
    data = np.loadtxt(filename, delimiter=',', unpack=True, usecols=(0, 1))
    x = np.transpose(np.array(data[0]))
    y = np.transpose(np.array(data[1]))
    return x, y

#加载样本数据

if __name__ == '__main__':
    x, y = LoadFile('ex1data1.txt')
    learn_rate=0.01  #设置超参数,学习率
    iter=1500    #迭代次数
    display_step=50  #每50次迭代显示一下效果
    #np.random.seed(612) #随机
    #w=np.random.randn()
    #b=np.random.randn()
    
    #初始化为0
    w=0
    b=0
    
    #训练模型
    mse=[] #存放每次迭代的损失值
    for i in range(0,iter+1):
        #求偏导
        dL_dw=np.mean(x*(w*x+b-y))
        dL_db=np.mean(w*x+b-y)
        #更新模型参数
        w=w-learn_rate*dL_dw
        b=b-learn_rate*dL_db
        #得到估计值
        pred=w*x+b
        #计算损失(均方误差)
        Loss=np.mean(np.square(y-pred))/2
        mse.append(Loss)
        #显示模型
        #plt.plot(x,pred)
        if i%display_step==0:
            print("i:%i,Loss:%f,w:%f,b:%f"%(i,mse[i],w,b))
            
            
    print("城市人口为35000时的预测餐车利润:%f"%(3.5*w+b))
    print("城市人口为70000时的预测餐车利润:%f"%(7*w+b))
    #模型和数据可视化
    plt.figure(figsize=(20,4))
    plt.subplot(1,3,1)
    #绘制散点图
    #张量和数组都可以作为散点函数的输入提供点坐标
    plt.scatter(x,y,color="red",label="数据集")
    plt.scatter(x,pred,color="green",label="梯度下降法")
    plt.plot(x,pred,color="blue")

    #设置坐标轴的标签文字和字号
    plt.xlabel("城市人口（万人）",fontsize=14)
    plt.ylabel("餐车利润（万美元）",fontsize=14)

    #在左上方显示图例
    plt.legend(loc="upper left")

    #损失变化可视化
    plt.subplot(1,3,2)
    plt.plot(mse)
    plt.xlabel("迭代次数",fontsize=14)
    plt.ylabel("损失值",fontsize=14)
    #估计值与标签值比较可视化
    plt.subplot(1,3,3)
    plt.plot(y,color="red",marker="o",label="数据集")
    plt.plot(pred,color="blue",marker="o",label="预测利润")
    plt.legend()
    plt.xlabel("sample",fontsize=14)
    plt.ylabel("price",fontsize=14)
    #显示整个绘图
    plt.show()


