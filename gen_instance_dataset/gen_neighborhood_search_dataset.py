import sys
sys.path.append('./')
import numpy as np
from uc_class import UC
import torch
import os
import json
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from rich import print
from multiprocessing import Pool
import argparse
from utilities import get_Tripartite_graph_lp_with_sol



def local_search(m,sol,relaxed):
    m1 = m.copy()
    u = m1.getVars()[:len(sol)]
    for i in range(len(sol)):
        if relaxed[i] == 0:
            m1.addConstr(u[i]==sol[i],name=f'fix_u[{i}]')
    m1.setParam('TimeLimit', 360)
    m1.setParam('Threads', 1)
    m1.setParam('MIPGap', 1e-3)
    m1.optimize()
    return m1.objVal

def reduce_redundant(neighbor,rate=0.2):
    ones_indices = np.where(neighbor == 1)[0]
    num_to_change = len(ones_indices) - int(len(neighbor)*rate)
    if num_to_change > 0:
        indices_to_change = np.random.choice(ones_indices, size=num_to_change, replace=False)
        neighbor[indices_to_change] = 0
    return neighbor

def solve(m,sol):
    m1 = m.copy()
    u = m1.getVars()[:len(sol)]
    for i in range(len(sol)):
        m1.addConstr(u[i]==sol[i],name=f'fix_u[{i}]')
    m1.optimize()
    if m1.status != 2:
        m1.computeIIS()
        for c in m1.getConstrs():
            if c.IISConstr:
                print(c.constrName)
    return m1.objVal

def flip_neighbor(neighbor,change_rate=0.025):
    num_to_change = int(len(neighbor)*change_rate)
    indices_to_change = np.random.choice(len(neighbor), size=num_to_change, replace=False)
    for idx in indices_to_change:
        if np.random.rand() < 0.5:  # 50% flip
            neighbor[idx] = np.bitwise_xor(neighbor[idx], 1)
    return neighbor

def process(G):
    file,json_path,data_path,uc = G
    with open(os.path.join(json_path, file),'r') as f:
        data = json.load(f)
    m3 = uc.get_3bin_model(data)
    m1 = uc.get_1bin_model(data)
    negative_datasets = data['negative_datasets']
    positive_datasets = data['positive_datasets']
    negative_datasets.extend(positive_datasets)

    sol_list = data['sol_list']
    opt_sol = positive_datasets[0]
    def gen_graph_by_sol(sol,graph1,graph3):
        best_obj = solve(m3,opt_sol)
        cur_obj = solve(m3,sol)
        positive_neighbors=[]
        for positive in positive_datasets:
            neighbor = np.bitwise_xor(sol,positive)
            if np.count_nonzero(neighbor)>len(neighbor)//5:
                neighbor = reduce_redundant(neighbor,0.2)
            
            new_obj = local_search(m3,sol,neighbor)
            if cur_obj-new_obj>0.6*(cur_obj-best_obj):
                positive_neighbors.append(neighbor.copy())

        negative_num = len(positive_neighbors)*9
        negative_neighbors = []
        
        max_iter=20

        while max_iter>0 and len(negative_neighbors)<negative_num:
            change_rate=0
            for negative in negative_datasets:
                neighbor = np.bitwise_xor(sol,negative)
                neighbor = flip_neighbor(neighbor,change_rate)
                new_obj = local_search(m3,sol,neighbor)
                if cur_obj-new_obj<0.05*(cur_obj-best_obj):
                    negative_neighbors.append(neighbor.copy())
            change_rate+=0.05
            change_rate=min(1,change_rate)

            max_iter-=1
        negative_num = len(negative_neighbors)
        if negative_num<1:
            return
        positive_num = negative_num//9
        positive_neighbors = positive_neighbors[:positive_num]
        torch.save({
            'graph': graph1,
            'positive': positive_neighbors,
            'negative': negative_neighbors,
        },os.path.join(data_path,'1bin',file[:-5]+f'_{idx}.pth'))
        torch.save({
            'graph': graph3,
            'positive': positive_neighbors,
            'negative': negative_neighbors,
        },os.path.join(data_path,'3bin',file[:-5]+f'_{idx}.pth'))



    for idx,sol in enumerate(sol_list):
        sol = np.round(sol).astype(int)

        graph1 = get_Tripartite_graph_lp_with_sol(m1,sol)
        graph3 = get_Tripartite_graph_lp_with_sol(m3,sol)
        gen_graph_by_sol(sol,graph1,graph3)


#主函数
def main():
    dirs=['train','valid','test']

    parser = argparse.ArgumentParser()

    parser.add_argument('--instances', type=str, default='8_std')#set to 8 for quick start
    parser.add_argument('--core_num', type=int, default='1')
    args = parser.parse_args()

    CORE_NUM = args.core_num
    ins = args.instances

    uc_path = f'UC_AF/{ins}.mod'
    uc = UC(uc_path)
    
    for dir in dirs:
        json_path = os.path.join(f'datasets/{ins}/json',dir)

        data_path = os.path.join(f'datasets/{ins}/neighborhood',dir)
        
        os.makedirs(data_path+'/1bin',exist_ok=True)
        os.makedirs(data_path+'/3bin',exist_ok=True)

        json_files = os.listdir(json_path)
        move = [(file,json_path,data_path,uc) for file in json_files if file.endswith('.json')]

        with Progress(
            "[progress.description]{task.description}({task.completed}/{task.total})",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.2f}%",
            TimeElapsedColumn(),
            '<',
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task(f"[cyan]Processing {dir} files...", total=len(move))

            with Pool(processes=CORE_NUM) as pool:
                for _ in pool.imap_unordered(process, move):

                    progress.update(task_id, advance=1)


if __name__ == "__main__":
    main()