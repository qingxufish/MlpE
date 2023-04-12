import numpy as np
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt

x_axis_data = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
y_axis_hits1 = [
0.4008,
0.3976,
0.4032,
0.4044,
0.4097,
0.4073,
0.4105,
0.4079,
0.4158,
0.4086,
0.4121,
0.4073,
0.4123,
0.4105
]
y_axis_hits3 = [
0.4458,
0.4517,
0.4537,
0.4521,
0.4609,
0.4587,
0.4585,
0.458,
0.4635,
0.4579,
0.463,
0.4558,
0.4631,
0.4655
]
y_axis_hits10 = [
0.5145,
0.5246,
0.5268,
0.5137,
0.5313,
0.5254,
0.5227,
0.5295,
0.5294,
0.5303,
0.5332,
0.5322,
0.527,
0.5303
]
y_axis_mrr = [
0.4368,
0.4385,
0.4418,
0.4409,
0.4483,
0.446,
0.4472,
0.4466,
0.4525,
0.4471,
0.4507,
0.4465,
0.4493,
0.4498
]
#画图
#A, = plt.plot(x_axis_data, y_axis_hits1, 'b*--', alpha=0.5, linewidth=1, label='Hits@1')
#B, = plt.plot(x_axis_data, y_axis_hits3, 'r*--', alpha=0.5, linewidth=1, label='Hits@3')
#C, = plt.plot(x_axis_data, y_axis_hits10, 'g*--', alpha=0.5, linewidth=1, label='Hits@10')
D, = plt.plot(x_axis_data, y_axis_mrr, 'b*--', alpha=0.5, linewidth=1, label='MRR')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'  : 10,
}

font2 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'  : 10,
}

# 图例
legend = plt.legend(handles=[D], prop=font1)

## 设置坐标轴间隔
x_major_locator=MultipleLocator(1)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.001)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
plt.xlim(1, 16)
#把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白

#设置坐标刻度值的大小以及刻度值的字体
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Times New Roman')
    label.set_fontsize(8)

#设置横纵坐标的名称以及对应字体格式
plt.xlabel('convolutional kernel size', font2)
plt.ylabel('MRR', font2)

# 设置网格
plt.grid(linestyle='--', color = 'gray', linewidth = 0.5)
# 保存
plt.savefig("RMSE.png", dpi = 600, bbox_inches = 'tight', pad_inches = 0.03)
plt.show()