import sys
sys.path.append('./')
import numpy as np
from uc_class import UC
from gurobipy import GRB
import os
import json
from multiprocessing import Pool
from rich.progress import Progress, BarColumn, SpinnerColumn, TimeRemainingColumn, TimeElapsedColumn
import argparse


def mycallback(model, where):
    if where == GRB.Callback.MIPSOL:
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        sol = model.cbGetSolution(model._u)
        model._objs.append(obj)
        model._sols.append(sol)

def get_positive(m,num_positive):
    m.setParam('OutputFlag', 0)
    m.setParam('Threads', 1)
    m.setParam('MIPGap', 1e-7)
    m.setParam('PoolSearchMode', 2)
    m.setParam('PoolSolutions', num_positive)
    m.setParam('TimeLimit', 60)
    mvars = m.getVars()
    u=[mvars[i] for i in range(len(mvars)) if 'u' in mvars[i].VarName]
    m.optimize(mycallback)
    objs = []
    if m.SolCount > 0:
        positives = []
        for sn in range(m.SolCount):
            m.setParam('SolutionNumber', sn)
            sol = np.round(m.getAttr("Xn", u)).astype(int)
            positives.append(sol)
            objs.append(m.PoolObjVal)
        positives = np.array(positives)
        return positives,np.mean(objs)
    return np.array([]),-1

def get_negative(m,num_negative):
    m.setParam('OutputFlag',0)
    m.setParam('MIPGap', 1e-2)
    m.setParam('Threads', 1)
    m.setParam('PoolSearchMode', 2)
    m.setParam('PoolSolutions', num_negative)
    m.setParam('TimeLimit', 4)
    m.optimize()
    if m.SolCount > 0:
        negatives = []
        for sn in range(m.SolCount):
            m.setParam('SolutionNumber',sn)
            mvars=m.getVars()
            u=[mvars[i] for i in range(len(mvars)) if 'u' in mvars[i].VarName]
            sol = np.round(m.getAttr("Xn", u)).astype(int)
            negatives.append(sol)
        negatives=np.array(negatives)
        return negatives
    return np.array([])

def solve(m,sol):
    m1 = m.copy()
    u = m1.getVars()[:len(sol)]
    for i in range(len(sol)):
        m1.addConstr(u[i]==sol[i],name=f'fix_u[{i}]')
    m1.optimize()
    return m1.objVal


def process(G):
    cur,mip_num,dt,u0,p0,onoff,sys_low,sys_up,uc,ins = G
    
    if cur%10==3:
        path = 'valid'
    elif cur%10==7:
        path = 'test'
    else:
        path = 'train'
    cnt = 0
    while cnt<mip_num:
        # print(f'{cur}_{cnt}')
        Dt=np.empty(len(dt))
        for i in range(len(dt)):
            Dt[i]=max(sys_low,min(sys_up,np.random.uniform(dt[i]*0.95,dt[i]*1.05)))
        Dt*=np.random.uniform(0.95,1.05)
        Spin = Dt*0.1
        Dt = Dt.tolist()
        Spin = Spin.tolist()

        data = {'Dt':Dt,'Spin':Spin,'u0':u0.tolist(),'p0':p0.tolist(),'on_off':onoff.tolist()}
        m=uc.get_3bin_model(data)
        mvars = m.getVars()
        u=[mvars[i] for i in range(len(mvars)) if 'u' in mvars[i].VarName]
        m._u = u
        m._objs=[]
        m._sols=[]
        positive_datasets,mean_obj = get_positive(m,10)
        if mean_obj < 0:
            continue
        negative_datasets = get_negative(m,10*9)
        sol_list = []
        obj_list = []
        for obj,sol in zip(m._objs,m._sols):
            if (obj - mean_obj)/obj > 1e-5:
                sol_list.append(sol)
                obj_list.append(obj)
        
        data['sol_list'] = sol_list
        data['obj_list'] = obj_list
        data['positive_datasets'] = positive_datasets.tolist()
        data['negative_datasets'] = negative_datasets.tolist()
        with open(os.path.join(f'datasets/{ins}/json/{path}/{cur}_{cnt}.json', ),'w') as f:
            json.dump(data,f)
        cnt+=1

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--instances', type=str, default='8_std')#set to 8 for quick start
    parser.add_argument('--core_num', type=int, default='1')
    parser.add_argument('--mip_num', type=int, default='1')#set to 1 for quick start, 10 or more for full dataset

    args = parser.parse_args()

    #set the number of cores and system size of dataset
    ins = args.instances
    CORE_NUM = args.core_num
    mip_num = args.mip_num


    Dt_origin=np.loadtxt('deman/pd.csv',delimiter=',',encoding='utf-8-sig')

    
    
    u0_save=np.load(f'instances/fixed/{ins}/u0_save.npy')
    p0_save=np.load(f'instances/fixed/{ins}/p0_save.npy')
    onoff_save=np.load(f'instances/fixed/{ins}/onoff_save.npy')

    os.makedirs(f'datasets/{ins}/json/train',exist_ok=True)
    os.makedirs(f'datasets/{ins}/json/valid',exist_ok=True)
    os.makedirs(f'datasets/{ins}/json/test',exist_ok=True)

    uc_path = f'UC_AF/{ins}.mod'
    uc = UC(uc_path)
    sys_up=sum(uc.ThPimax)*0.9
    sys_low=sys_up*0.4
    pd=(Dt_origin-Dt_origin.min())/(Dt_origin.max()-Dt_origin.min())*0.95*(sys_up-sys_low)+sys_low

    


    move = [(i, mip_num, pd[i], u0_save[i], p0_save[i], onoff_save[i],sys_low,sys_up,uc,ins) for i in range(365)]
    
    with Progress(
        "[progress.description]{task.description}({task.completed}/{task.total})",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.2f}%",
        TimeElapsedColumn(),
        '<',
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task(f"[cyan]Processing {ins} {dir} files...", total=len(move))
        with Pool(processes=CORE_NUM) as pool:
            for _ in pool.imap_unordered(process, move):

                progress.update(task_id, advance=1)


if __name__ == "__main__":
    main()