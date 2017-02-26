from numpy.linalg import norm
import numpy as np
from matplotlib import pyplot as plt, style
style.use('fivethirtyeight')


class SVM:

    def __init__(self, visualize=True):
        self.visualize = visualize
        self.colors = {1:'r',-1:'b'}
        if self.visualize:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    def fit(self, data):
        self.data = data
        opt_mags = {}
        transforms = [[1,1],[1,-1],[-1,1],[-1,-1]]
        self.max_val = max([f for yi in self.data for xi in self.data[yi] for f in xi])
        self.min_val = min([f for yi in self.data for xi in self.data[yi] for f in xi])
        steps = [self.max_val*0.1,self.max_val*0.01,self.max_val*0.001]
        start_w = self.max_val*10
        b_range = 5

        for step in steps:
            w = np.array([start_w,start_w])
            step_optimized = False
            while not step_optimized:
                for b in np.arange(-(b_range*self.max_val),b_range*self.max_val,step*b_range):
                    for trans in transforms:
                        wt = w * trans
                        constraints = True
                        for yi in self.data:
                            for xi in self.data[yi]:
                                if not yi*(np.dot(xi,wt)+b) >= 1:
                                    constraints = False
                        if constraints:
                            opt_mags[norm(wt)] = [wt,b]
                if w[0] <= 0:
                    step_optimized = True
                else:
                    w = w - step
            mags = [m for m in sorted(opt_mags)]
            choice = opt_mags[mags[0]]
            self.w = choice[0]
            self.b = choice[1]
            start_w = self.w[0]+step*10
            print('Optimized a step')

    def predict(self,features):
        class_ = np.sign(np.dot(np.array(features),self.w)+self.b)
        if class_ != 0 and self.visualize:
            self.ax.scatter(features[0], features[1], color=self.colors[class_],marker='*',s=50)
        return class_

    def graph(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in self.data[i]] for i in self.data]

        # hyperplane = x.w+b
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_val*0.9,self.max_val*1.1)
        hpy_x_min = datarange[0]
        hpy_x_max = datarange[1]

        # Positive SV (x.w+b) = 1
        psv1 = hyperplane(hpy_x_min,self.w,self.b,1)
        psv2 = hyperplane(hpy_x_max,self.w,self.b,1)
        self.ax.plot([hpy_x_min,hpy_x_max], [psv1,psv2],'k',linewidth=2)

        # Negative SV (x.w+b) = -1
        nsv1 = hyperplane(hpy_x_min,self.w,self.b,-1)
        nsv2 = hyperplane(hpy_x_max,self.w,self.b,-1)
        self.ax.plot([hpy_x_min,hpy_x_max], [nsv1,nsv2],'k',linewidth=2)

        # Decision Boundary SV (x.w+b) = 0
        db1 = hyperplane(hpy_x_min,self.w,self.b,0)
        db2 = hyperplane(hpy_x_max,self.w,self.b,0 )
        self.ax.plot([hpy_x_min,hpy_x_max], [db1,db2],'y--',linewidth=3)

        plt.show()



train = {1: np.array([[1,2],[2,3],[3,1],[2,2]]),
         -1:np.array([[6,8],[8,6],[6,6],[7,8]])}
test = [4,8]
clf = SVM()
clf.fit(train)
predict_values = [[-1,2],[1,3],[4,2],[6,3],[8,-4],[3,5],[8,9],[9,0],[1,4],[6,2],[7,4],[5,0],[8,7],[4,4],[1,-1],[0,6]]
for p in predict_values:
    clf.predict(p)
clf.graph()
