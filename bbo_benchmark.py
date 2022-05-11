# %% [markdown]
# #  ブラックボックス最適化(BBO)ベンチマーク
# 
# ## 手法
# ### 遺伝的アルゴリズム
# 
# ### 進化戦略
# https://www.jstage.jst.go.jp/article/sicejl/54/8/54_567/_pdf
# http://www.matsumoto.nuem.nagoya-u.ac.jp/jascome/denshi-journal/20/No.08-201219.pdf
# https://arxiv.org/abs/1604.00772
# https://horomary.hatenablog.com/entry/2021/01/23/013508
# https://math-note.com/multivariate-normal-distribution/

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.facecolor"] = "white"

import os
import time
import datetime
from tqdm import tqdm

from lib.bbo_algo import *
from lib.benchmark_function import *

# %%
class BenchMarker():
    def __init__(self,max_iter=100):
        self._bbo_table = {
            'CMA-ES':CMA_ES(),
            'TPE':TPE(),
            'RandomSearch':RandomSearch(),
            #'GridSearch':GridSearch(),
            'GA(optuna)':NSGA(),
            'Nelder-Mead':NelderMead(),
            'GA(self made)':GeneticAlgorithm(),            
        }
        
        self._max_iter = max_iter
        self._result_s = dict()

    def benchmark(self,model,is_plot=False):
        

        result_table = {
            'algo':[],
            'calc time':[],
            'fval':[]      
        }

        result = dict()
        for key,value in tqdm(self._bbo_table.items(),leave=False):
            print("{} start!".format(key))
            start_time = time.perf_counter()
            fval,params = value.optimization(model,self._max_iter)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            result.update({key:(fval,params)})
            fval_s,params_s = value.get_history()
            self._result_s.update({key:(fval_s,params_s)})

            result_table['algo'].append(key)
            result_table['calc time'].append(elapsed_time)
            result_table['fval'].append(fval)
            
            del value


        for key,value in result.items():
            print("{} achieved {}.".format(key,value[0]))

        if is_plot:
            self.plot_history()

        return pd.DataFrame(result_table) 

    def plot_history(self,
                    fig_name=None,
                    is_log = False,
                    is_save=False,
                    is_show=False
                    ):
        plt.figure()

        trial = [i for i in range(self._max_iter)]
        for key,value in self._result_s.items():
            plt.plot(trial,value[0],label=key)
        plt.legend()
        plt.xlabel('Trial')
        plt.ylabel('Evaluation value')

        if is_log:
            plt.yscale('log')
        if is_save:
            os.makedirs('fig',exist_ok=True)
            if fig_name:
                d = fig_name
            else:
                now = datetime.datetime.now()
                d = now.strftime('%Y%m%d%H%M%S')
            plt.savefig('fig/'+d+'.png' ,format="png" ,dpi=300)
        if is_show:
            plt.show()      

# %%
df = pd.DataFrame()
dim_s = [100]
max_iter =100

for dim in dim_s:
    model_s = {
                'QuadraticFunction':QuadraticFunction(dim=dim),
                'StyblinskiTangFunction':StyblinskiTangFunction(dim=dim),
                'GriewankFunction':GriewankFunction(dim=dim),
                'AckleyFunction':AckleyFunction(dim=dim),
                'RosenbrockFunction':RosenbrockFunction(dim=dim),
                'SchwefelFunction':SchwefelFunction(dim=dim),
                'XinSheYangFunction':XinSheYangFunction(dim=dim),
            }

    for name,model in tqdm(model_s.items()):
        dict_additional = {
            'dimention':[],
            'model name':[]
        }

        bm = BenchMarker(max_iter=max_iter)
        df_temp = bm.benchmark(model = model)
        bm.plot_history(fig_name=str(dim)+'_'+name,is_log=True,is_save=True)

        
        dict_additional['dimention']=[dim for i in range(len(model_s))]
        dict_additional['model name']=[name for i in range(len(model_s))]
        df_additional = pd.DataFrame(dict_additional)

        df_temp = pd.concat([df_temp, df_additional], axis=1)
                
        df = pd.concat([df,df_temp])

# %%
df


