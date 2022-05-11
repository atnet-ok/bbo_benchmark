import numpy as np
import pandas as pd
import optuna
import math
from scipy.optimize import minimize
from scipy.optimize import Bounds 
from abc import ABCMeta,abstractmethod
from typing import List, Dict,Tuple

class Model(metaclass=ABCMeta):
    @abstractmethod
    def evaluation(self,params:List[any])->float:
        pass

    @abstractmethod
    def get_params_range(self)->List[Tuple[float]]:
        pass

class BBO(metaclass=ABCMeta):
    @abstractmethod
    def optimization(self,
                    model:Model,
                    max_iter:int
                    )->Tuple[float,List[any]]:
        """optimize model parameters

        Args:
            model (Model): [description]

        Returns:
            Tuple[float,List[any]]: [description]
        """

    @abstractmethod
    def get_history(self)->Tuple[List[any],List[List]]:
        """retrun optimization history

        Returns:
            Tuple[List[any],List[List]]: history of best value and parameters
        """

    # @abstractmethod
    # def get_calc_time(self)->float:
    #     pass

    def show_log(self,step,fval,params):
        print('trial:{}, fval:{}, params:{}'.format(step,
                                                fval,
                                                params
                                                )
        )




class GeneticAlgorithm(BBO):
    def __init__(self,
                    chrom_num = 200,
                    parent_rate = 0.5,
                    crossover_rate = 0.6,
                    crossover_method = 'two-point',
                    mutation_rate = 0.1,
                    mutation_prob = 0.11
                    ):

        self._chrom_num = chrom_num
        self._eval_history =[]
        self._params_history =[]

        self._max_generation = []
        self._parent_rate = parent_rate

        self._crossover_rate = crossover_rate 
        self._crossover_method = crossover_method

        self._mutation_rate = mutation_rate 
        self._mutation_prob = mutation_prob

    def optimization(self,model,max_iter):
        self._max_generation = max_iter-1

        #初期値をランダムに生成
        chrom_s = self._initial_step(model)
        #print(chrom_s)


        #最良の個体を選別
        chrom_best = self._select_chrom(chrom_s,
                                        selection_method='elite',
                                        select_num=1
                                        )
        self._eval_history.append(chrom_best['eval'])
        self._params_history.append(chrom_best['params'])
        #print(chrom_best)

        for i in range(self._max_generation):
            


            #親となる個体数を選別
            chrom_parent = self._select_chrom(chrom_s,
                                            selection_method='tournament',
                                            select_num=int(self._chrom_num*self._parent_rate)
                                            )
            #print(chrom_parent)

            #子を交叉により生成
            chrom_child_crossover=self._crossover(chrom_parent,
                                        crossover_method='two-point',
                                        crossover_num=int(self._chrom_num*(self._crossover_rate))
                                        )
            #print(chrom_child)

            #子を突然変異により生成
            chrom_child_mutation=self._mutation(chrom_parent,
                                            model=model,
                                            mutation_prob=self._mutation_prob ,
                                            mutation_num=int(self._chrom_num*(self._mutation_rate))
                                            )

            #print(chrom_child)

            #子の評価値を取得
            chrom_child = pd.concat([chrom_child_crossover,chrom_child_mutation])
            chrom_child = self._evaliation(chrom_child,
                                            model=model
                                            )

            #次世代を決定
            chrom_s = pd.concat([chrom_best,chrom_parent,chrom_child])
            chrom_s = self._select_chrom(chrom_s,
                                            selection_method='elite',
                                            select_num=self._chrom_num
                                            )

            #最良の個体を選別
            chrom_best = self._select_chrom(chrom_s,
                                            selection_method='elite',
                                            select_num=1
                                            )
            self._eval_history.append(chrom_best['eval'].values[0])
            self._params_history.append(chrom_best['params'].values[0])
            #print(chrom_best)

            self.show_log(i,
                          chrom_best['eval'].values[0],
                          np.mean(chrom_best['params'].values[0])
                        )
                

        return chrom_best['eval'].values[0], chrom_best['params'].values[0]

    def _evaliation(self,chrom_s,model):
        
        for index, row in chrom_s.iterrows():
            e_val = model.evaluation(row['params'])
            chrom_s.loc[index,'eval']  = e_val

        return chrom_s.sort_values('eval').reset_index(drop=True)

    def _initial_step(self,model):
        range_s = model.get_params_range()

        chrom_s = pd.DataFrame()
        
        for i in range(self._chrom_num):
            chrom = pd.DataFrame()

            #パラメータの初期値を一様分布で決定
            param_s =[]
            for range_ in range_s:
                param = (range_[1] - range_[0]) * np.random.rand() + range_[0]
                param_s.append(param)

            #評価値を取得
            val = model.evaluation(params=param_s)
            
            chrom['eval'] = [val]
            chrom['params'] = [param_s]
            chrom_s = pd.concat([chrom_s,chrom])

        return chrom_s.sort_values('eval').reset_index(drop=True)

    def _select_chrom(self,chrom_s,selection_method,select_num):
        
        if selection_method=='elite':
            chrom_s = chrom_s.sort_values('eval').reset_index(drop=True)
            return chrom_s.iloc[0:select_num,:]

        if selection_method=='tournament':
            chrom_s_selected = pd.DataFrame()
            tournament_num = 4
            
            for i in range(select_num):
                chrom_s_part = chrom_s.sample(tournament_num)
                min_index = chrom_s_part['eval'].idxmin()
                winner = chrom_s.iloc[[min_index]]
                chrom_s_selected = pd.concat([chrom_s_selected,winner])

            return chrom_s_selected.sort_values('eval').reset_index(drop=True)


    def _crossover(self,chrom_s,crossover_method,crossover_num):
        if crossover_method == 'one-point':
            pass

        if crossover_method == 'two-point':
            chrom_crossovered = pd.DataFrame()
            for i in range(round(crossover_num/2)):
                df = pd.DataFrame()
                chrom_temp = chrom_s.sample(2)
                param_a = list(chrom_temp.iloc[0]['params'])
                param_b = list(chrom_temp.iloc[1]['params'])

                index = np.random.randint(0, len(param_a), 2)
                while len(set(index))<2:
                    index = np.random.randint(0, len(param_a), 2)

                left_index = min(index)
                right_index = max(index)

                new_param_a = param_a[:left_index]+param_b[left_index:right_index]+param_a[right_index:]
                new_param_b = param_b[:left_index]+param_a[left_index:right_index]+param_b[right_index:]
                df['params'] = [new_param_a,new_param_b]
                df['eval'] = [np.nan,np.nan]

                chrom_crossovered = pd.concat([chrom_crossovered,df])
        
        return chrom_crossovered.reset_index(drop=True)

    def _mutation(self,chrom,model,mutation_prob,mutation_num):

        for i in range(mutation_num):
            index = np.random.randint(0, chrom.shape[0])

            params_temp = chrom.loc[index,'params'].copy()
            index_mutation_s = np.random.randint(0, len(params_temp),int(len(params_temp)*mutation_prob))

            for index_mutation in index_mutation_s:
                range_ = model.get_params_range()[index_mutation]             
                params_temp[index_mutation] = (range_[1] - range_[0]) * np.random.rand() + range_[0]

            col_idx = chrom.columns.get_loc('params')
            chrom.iat[index,col_idx] = params_temp #listをDataFrameの要素に代入するときは、iatじゃないとエラーがでる

        return chrom


    def get_history(self):
        return self._eval_history,self._params_history

class GridSearch(BBO):
    def __init__(self):
        self._study = []

    def optimization(self,model,max_iter):
        range_s = model.get_params_range()
        search_space = dict()
        grid_num = math.ceil(pow(max_iter,1/len(range_s)))
        for i,range_ in enumerate(range_s):
            grid = list(np.linspace(range_[0],range_[1],grid_num))
            search_space.update({"x"+str(i):grid})
        sampler = optuna.samplers.GridSampler(search_space)
        study = optuna.create_study(sampler=sampler)
        self._study = study

        def objective(trial):
            range_s = model.get_params_range()
            params = [trial.suggest_uniform('x'+str(i), range_[0], range_[1]) for i,range_ in enumerate(range_s)]
            score = model.evaluation(params)
            return score

        self._study.optimize(objective, n_trials=max_iter)
        return self._study.best_value,self._study.best_params


    def get_history(self):
        fval_s = [trial.value for trial in self._study.get_trials()]
        params_s = [list(trial.params.values()) for trial in self._study.get_trials()]
            
        return fval_s,params_s

class OptunaAlgo(BBO):
    def __init__(self,sampler):
        self._eval_history =[]
        study = optuna.create_study(sampler=sampler)
        self._study = study

    def optimization(self,model:Model,max_iter):

        def objective(trial):
            range_s = model.get_params_range()
            params = [trial.suggest_uniform('x'+str(i), range_[0], range_[1]) for i,range_ in enumerate(range_s)]
            score = model.evaluation(params)
            return score

        self._study.optimize(objective, n_trials=max_iter)
        return self._study.best_value, self._study.best_params


    def get_history(self):
        fval_s = [trial.value for trial in self._study.get_trials()]
        params_s = [list(trial.params.values()) for trial in self._study.get_trials()]
            
        return fval_s,params_s
        
class TPE(OptunaAlgo):
    def __init__(self):
        sampler = optuna.samplers.TPESampler()
        super().__init__(sampler)

class NSGA(OptunaAlgo):
    def __init__(self):
        sampler = optuna.samplers.NSGAIISampler()
        super().__init__(sampler)

class CMA_ES(OptunaAlgo):
    def __init__(self):
        sampler = optuna.samplers.CmaEsSampler()
        super().__init__(sampler)

class RandomSearch(OptunaAlgo):
    def __init__(self):
        sampler = optuna.samplers.RandomSampler()
        super().__init__(sampler)

class NelderMead(BBO):
    def __init__(self):
        self._result = []
        self._fval_history = []
        self._param_history = []

    def optimization(self,
                    model:Model,
                    max_iter=100
                    )->Tuple[float,List[any]]:
        def fun(x):
            params =list(x)
            fval = model.evaluation(params)
            return fval

        def callbackF(xx):
            fval =fun(xx)
            self._fval_history.append(fval)
            self._param_history.append(xx)

        range_s = model.get_params_range()
        param_s =[]
        for range_ in range_s:
            param = (range_[1] - range_[0]) * np.random.rand() + range_[0]
            param_s.append(param)
        x0 = np.array(param_s)

        lb = []
        ub = []
        for range_ in range_s:
            lb.append(range_[0])
            ub.append(range_[1])
        bounds = Bounds(lb,ub)

        self._result =  minimize(
                            fun, 
                            x0,  
                            callback=callbackF, 
                            method='Nelder-Mead',  
                            bounds=bounds ,
                            options={'maxiter':max_iter+1}
                            )
        
        return self._result.fun, self._result.x
        
    def get_history(self)->Tuple[List[any],List[List]]:
        return self._fval_history, self._param_history