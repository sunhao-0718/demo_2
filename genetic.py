import numpy as np
import pandas as pd
import math
import random
import sys
import codecs
from datetime import datetime
import matplotlib.pyplot as plt
# Stackel_berg均衡解
# 设立两个对象解决两层目标规划


# 建立上层函数的遗传算法
class Genetic(object):
    def __init__(self, pop_size, pc, pm, gen, sub_gen, sub_pop_size, a, M, alpha1, alpha2):
        super(Genetic, self).__init__()
        self.pop_size = pop_size  # 基因的数量
        self.pc = pc  # 交叉的可能性
        self.pm = pm  # 变异的可能性
        self.gen = gen  # 上层迭代的次数
        self.a = a  # 选择率
        self.M = M  # 蒙特卡洛模拟数量
        self.sub_gen = sub_gen  # 下层迭代次数
        self.sub_pop_size = sub_pop_size  # 下层基因数量
        self.alpha1 = alpha1  # 概率值alpha_1的值
        self.alpha2 = alpha2  # 概率值alpha_2的值

        self.opt_x = 0  # 最优解x的值
        self.opt_y = 0  # 最优解y的值
        self.f_val = float('-inf')  # 上层函数最优值
        self.sub_val = float('-inf')  # 下层函数最优值

        self.f_val_list = []
        self.sub_val_list = []
        self.opt_x_list = []
        self.opt_y_list = []
        self.gen_list = []
        self.time_list = []

    # 进行上层遗传算法
    def process(self):
        start_time = datetime.now()
        # 产生随机数
        random_value = self.__generate_random()

        # 初始化
        x = 7 * np.random.random(self.pop_size) + 5
        y = np.zeros(self.pop_size)
        genetic_y = SubGenetic(self.sub_pop_size, self.pc, self.pm, self.sub_gen, self.a, self.alpha1, self.alpha2)
        # 建立选择空间
        q = np.zeros(self.pop_size)
        q[0] = self.a
        for i in range(1, self.pop_size):
            q[i] = q[i-1] + self.a*math.pow((1-self.a), i)

        # 开始进行遗传
        for i in range(1, self.gen):
            print('上层遗传算法的第%s代' % i)
            # 评估
            print('evaluate')
            object_value = np.zeros(self.pop_size)
            for j in range(self.pop_size):
                optimal_y, sub_val = genetic_y.process(x[j], random_value, self.M)
                y[j] = optimal_y
                object_value[j] = random_value.apply(self.__main_func, axis=1, args=(x[j], optimal_y,)).mean()

            opt_value = object_value.max()
            max_index = np.argmax(object_value)

            if opt_value > self.f_val:
                self.f_val = opt_value
                self.opt_x = x[max_index]
                self.opt_y = y[max_index]

            # 选择
            print('select')
            selection = np.array([x, object_value]).T
            selection = selection[np.argsort(-selection[:, -1])]
            for j in range(0, self.pop_size):
                sr = random.random()*q[-1]
                x[j] = selection[np.sum(q < sr), 0]

            # 交叉
            print('crossover')
            is_pc = np.random.random(self.pop_size) < self.pc
            crossover = np.array([x, is_pc]).T
            crossover = crossover[np.argsort(-crossover[:, -1])]
            for j in range(0, self.pop_size, 2):
                if crossover[j, 1] is True:
                    r = random.random()
                    x[j] = crossover[j, 0]*r + crossover[j+1, 0]*(1-r)
                    x[j+1] = crossover[j+1, 0]*r + crossover[j, 0]*(1-r)

            # 突变
            print('mutation')
            is_pm = np.random.random(self.pop_size) < self.pm
            mutation = np.array([x, is_pm]).T
            for j in range(self.pop_size):
                if mutation[j, 1] is True:
                    x[j] = random.random()*7 + 5
            self.sub_val = random_value.apply(genetic_y.sub_func, axis=1, args=(self.opt_x, self.opt_y,)).mean()

            if i % (self.gen/30) == 0:
                self.f_val_list.append(self.f_val)
                self.sub_val_list.append(self.sub_val)
                self.gen_list.append(i)
                self.time_list.append(datetime.now() - start_time)
                self.opt_x_list.append(self.opt_x)
                self.opt_y_list.append(self.opt_y)

    def __main_func(self, series, x, y):
        x1 = series[0]
        x2 = series[1]
        value = (30 + x1 - 2*x - y)*x - math.pow(x - 10 - x2, 2)
        return value

    def __generate_random(self):
        x1 = np.random.uniform(2, 4, self.M)
        x2 = np.random.exponential(4, self.M)
        x3 = np.random.uniform(3, 4, self.M)
        x4 = np.random.exponential(3, self.M)
        x = pd.DataFrame([x1, x2, x3, x4]).T
        return x

    def plot_gen(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax1.plot(self.gen_list, self.f_val_list)
        ax1.title('f_val - iterations')
        ax1.xlabel('iterations')
        ax1.ylabel('f_val')
        ax2.plot(self.gen_list, self.sub_val_list)
        ax2.title('sub_val - iterations')
        ax2.xlabel('iterations')
        ax2.ylabel('sub_val')
        ax2.plot(self.gen_list, self.time_list)
        ax3.title('time - iterations')
        ax3.xlabel('iterations')
        ax3.ylabel('sub_val')
        plt.savefig('./result.png')
        plt.show()

    def save_result(self, result_save_path):
        with codecs.open(result_save_path, 'w', encoding='utf-8') as f:
            for i in range(len(self.gen_list)):
                f.write('第%s次迭代:\n' % self.gen_list[i])
                f.write('opt_x: %s\t opt_y: %s\n f_val:%s\t sub_val: %s\n' % (self.opt_x_list[i], self.opt_y_list[i],
                                                                            self.f_val_list[i], self.sub_val_list[i]))



# 建立下层函数的遗传算法
class SubGenetic(object):
    def __init__(self, sub_pop_size, pc, pm, sub_gen, a, alpha1, alpha2):
        super(SubGenetic, self).__init__()
        self.sub_pop_size = sub_pop_size
        self.pc = pc
        self.pm = pm
        self.sub_gen = sub_gen
        self.a = a
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def process(self, x, random_value, M):
        f_val = float('-inf')

        # 初始化
        q = np.zeros(self.sub_pop_size)
        q[0] = self.a
        for i in range(1, self.sub_pop_size):
            q[i] = q[i - 1] + self.a * math.pow((1 - self.a), i)
        upper_y = 100
        y = np.random.random(self.sub_pop_size)*upper_y
        for j in range(self.sub_pop_size):
            while random_value.apply(lambda series:2*x + y[j] <= 30 + series[0], axis=1).sum()/M\
                    < self.alpha1 or \
                    random_value.apply(lambda series:x + 2*y[j] <= 22 + series[2], axis=1).sum()/M\
                    < self.alpha2:
                y[j] = random.random()*upper_y

        for i in range(1, self.sub_gen):

            # 评估
            # print('sub evaluate')
            object_value = np.zeros(self.sub_pop_size)
            for j in range(self.sub_pop_size):
                object_value[j] = random_value.apply(self.sub_func, axis=1, args=(x, y[j])).mean()

            opt_value = object_value.max()
            max_index = np.argmax(object_value)
            if opt_value > f_val:
                f_val = opt_value
                opt_y = y[max_index]

            # 选择
            # print('sub selection')
            selection = np.array([y, object_value]).T
            selection = selection[np.argsort(-selection[:, -1])]
            for j in range(0, self.sub_pop_size):
                sr = random.random()*q[-1]
                y[j] = selection[np.sum(q < sr), 0]

            # 交叉
            # print('sub crossover')
            is_pc = np.random.random(self.sub_pop_size) < self.pc
            crossover = np.array([y, is_pc]).T
            crossover = crossover[np.argsort(-crossover[:, -1])]
            for j in range(0, self.sub_pop_size, 2):
                if crossover[j, 1] is True:
                    r = random.random()
                    y[j] = crossover[j, 0]*r + crossover[j+1, 0]*(1-r)
                    while random_value.apply(lambda series: 2 * x + y[j] <= 30 + series[
                        0], axis=1).sum() / M < self.alpha1 or random_value.apply(
                            lambda series: x + 2 * y[j] <= 22 + series[2], axis=1).sum() / M < self.alpha2:
                        r = random.random()
                        y[j] = crossover[j, 0] * r + crossover[j + 1, 0] * (1 - r)

                    y[j+1] = crossover[j+1, 0]*r + crossover[j, 0]*(1-r)
                    while random_value.apply(lambda series: 2 * x + y[j+1] <= 30 + series[
                        0], axis=1).sum() / M < self.alpha1 or random_value.apply(
                        lambda series: x + 2 * y[j+1] <= 22 + series[2], axis=1).sum() / M < self.alpha2:
                        r = random.random()
                        y[j + 1] = crossover[j + 1, 0] * r + crossover[j, 0] * (1 - r)

            # 突变
            # print('sub mutation')
            is_pm = np.random.random(self.sub_pop_size) < self.pm
            mutation = np.array([y, is_pm]).T
            for j in range(self.sub_pop_size):
                if mutation[j, 1] is True:
                    y[j] = random.random()*upper_y
                    while random_value.apply(lambda series: 2 * x + y[j] <= 30 + series[
                        0], axis=1).sum() / M < self.alpha1 or random_value.apply(
                            lambda series: x + 2 * y[j] <= 22 + series[2], axis=1).sum() / M < self.alpha2:
                        y[j] = random.random() * upper_y
        return opt_y, f_val

    def sub_func(self, series, x, y):
        x3 = series[2]
        x4 = series[3]
        value = (22 + x3 - x - 2*y)*y - math.pow((y - 8 - x4), 2)
        return value


def main():
    genetic = Genetic(pop_size=50, pc=0.2, pm=0.2, gen=50,
                      sub_gen=30, sub_pop_size=30, a=0.1,
                      M=5000, alpha1=0.95, alpha2=0.9)
    genetic.process()
    genetic.plot_gen()
    genetic.save_result(result_save_path='./result.txt')
    print('result as below:\n opt_x: %s\t opt_y: %s\n f_val:%s\t sub_val: %s' % (genetic.opt_x, genetic.opt_y, genetic.f_val, genetic.sub_val))


if __name__ == '__main__':
    main()
