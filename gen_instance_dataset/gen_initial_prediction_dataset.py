import sys
sys.path.append('./')
from uc_class import UC
import torch
import collections
import os
import json
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from rich import print
from multiprocessing import Pool
import argparse

from utilities import get_Tripartite_graph_lp



def process(G):
    file,json_path,data_path,uc = G
    with open(os.path.join(json_path, file),'r') as f:
        data = json.load(f)

    m1 = uc.get_1bin_model(data)
    m3 = uc.get_3bin_model(data)


    negative_datasets = data['negative_datasets']
    positive_datasets = data['positive_datasets']

    tri1 = get_Tripartite_graph_lp(m1)
    tri3 = get_Tripartite_graph_lp(m3)
    torch.save({
        'graph': tri1,
        'positive': positive_datasets,
        'negative': negative_datasets,
    },os.path.join(data_path,'1bin',file[:-5]+f'.pt'))
    torch.save({
        'graph': tri3,
        'positive': positive_datasets,
        'negative': negative_datasets,
    },os.path.join(data_path,'3bin',file[:-5]+f'.pt'))


#主函数
def main():

    dirs=['train','valid','test']

    parser = argparse.ArgumentParser()

    parser.add_argument('--instances', type=str, default='8_std')#set to 8 for quick start
    parser.add_argument('--core_num', type=int, default='1')
    args = parser.parse_args()

    ins = args.instances
    CORE_NUM = args.core_num
    uc_path = f'UC_AF/{ins}.mod'
    uc = UC(uc_path)
    
    for dir in dirs:
        json_path = os.path.join(f'datasets/{ins}/json',dir)
        data_path = os.path.join(f'datasets/{ins}/initsol',dir)
        
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