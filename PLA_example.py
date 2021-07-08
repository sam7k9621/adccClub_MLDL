import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(color_codes=True)


dataset = np.array([
((1, 5.0, 1.5), -1),
((1, 4.8, 1.2), -1),
((1, 5.1, 1.9), -1),
((1, 4.9, 1.8), -1),

((1, 6.1, 3.8), 1),
((1, 6.3, 3.9), 1),
((1, 6.8, 4.1), 1),
((1, 6.4, 4.2), 1)],
dtype="object")

def plothist(w, pt):
    ps = [v[0] for v in dataset]
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_aspect('equal', adjustable='box')
    plt.ylim(-2, 8)
    plt.xlim(-2, 8)
    plt.xlabel('sepal length (cm)')
    plt.ylabel('petal length (cm)') 
    
    x_axis = np.linspace(-2,8)
    y_axis = np.linspace(-2,8)
    ax1.scatter([v[1] for v in ps[:4]], [v[2] for v in ps[:4]], s=10, c='r', marker="x", label='X')
    ax1.scatter([v[1] for v in ps[4:]], [v[2] for v in ps[4:]], s=10, c='b', marker="o", label='O')
    ax1.plot(x_axis, 0*x_axis, 'k')
    ax1.plot(0*y_axis, y_axis, 'k')
    
    pt_l = np.linspace(0, pt[1])
    if pt[1] > 4:
        ax1.plot(pt_l, pt_l*pt[2]/pt[1], 'b--')
        slope = +1
    else:
        ax1.plot(pt_l, pt_l*pt[2]/pt[1], 'r--')
        slope = -1
    
    new_w = w + slope*pt
    pt_a = np.linspace(0, new_w[1])
    ax1.plot(pt_a, pt_a*new_w[2]/new_w[1], 'y--')
    
    if any(w):
        pt_a = np.linspace(0, w[1])
        ax1.plot(pt_a, pt_a*w[2]/w[1], 'g--')
        a,b = -w[1]/w[2], -w[0]/w[2]
        ax1.plot(x_axis, a*x_axis + b, 'g-')
    
    
    plt.legend(loc='upper left');
    plt.show()
    
def check_error(w, dataset):
    result = None
    error = 0
    for x, s in dataset:
        x = np.array(x)
        if int(np.sign(w.T.dot(x))) != s:
            result =  x, s
            return result
    
    return result

#PLA演演算法實作

def pla(dataset):
    w = np.zeros(3)
    while check_error(w, dataset) is not None:
        x, s = check_error(w, dataset) 
        plothist(w, x)
        w += s * x
    return w


#執行

ps = [v[0] for v in dataset]
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_aspect('equal', adjustable='box')
plt.ylim(-2, 8)
plt.xlim(-2, 8)
plt.xlabel('sepal length (cm)')
plt.ylabel('petal length (cm)')

x_axis = np.linspace(-2,8)
y_axis = np.linspace(-2,8)
ax1.scatter([v[1] for v in ps[:4]], [v[2] for v in ps[:4]], s=10, c='r', marker="x", label='X')
ax1.scatter([v[1] for v in ps[4:]], [v[2] for v in ps[4:]], s=10, c='b', marker="o", label='O')
ax1.plot(x_axis, 0*x_axis, 'k')
ax1.plot(0*y_axis, y_axis, 'k')
plt.legend(loc='upper left');
plt.show()
    


w = pla(dataset)


print(w)
ps = [v[0] for v in dataset]
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_aspect('equal', adjustable='box')
plt.ylim(-2, 8)
plt.xlim(-2, 8)
plt.xlabel('sepal length (cm)')
plt.ylabel('petal length (cm)') 

x_axis = np.linspace(-2,8)
y_axis = np.linspace(-2,8)
ax1.scatter([v[1] for v in ps[:4]], [v[2] for v in ps[:4]], s=10, c='r', marker="x", label='X')
ax1.scatter([v[1] for v in ps[4:]], [v[2] for v in ps[4:]], s=10, c='b', marker="o", label='O')
ax1.plot(x_axis, 0*x_axis, 'k')
ax1.plot(0*y_axis, y_axis, 'k')

a,b = -w[1]/w[2], -w[0]/w[2]
ax1.plot(x_axis, a*x_axis + b, 'g-')


plt.legend(loc='upper left');
plt.show()




