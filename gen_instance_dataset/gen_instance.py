import sys
sys.path.append('./')
import numpy as np
import os
from uc_class import UC

import argparse

from rich import print
from rich.progress import Progress, BarColumn, SpinnerColumn, TimeRemainingColumn, TimeElapsedColumn


def cal_upo(m):
    mvars = m.getVars()
    u=[]
    p=[]
    for i in range(len(mvars)):
        if 'u'in mvars[i].Varname:
            u.append(mvars[i])
        if 'p['in mvars[i].Varname:
            p.append(mvars[i])

    px = np.round(np.array(m.getAttr("X", p), dtype=float), decimals=3).reshape(-1, 24)
    ux = np.around(m.getAttr("X", u)).astype(int).reshape(-1,24)
    n=px.shape[0]
    
    p0=[float(px[i][-1]) for i in range(n)]
    u0=[int(ux[i][-1]) for i in range(n)]

    onoff=[]
    for i in range(n):
        k = 1 if ux[i][-1]==1 else -1
        num=1
        for j in range(2,25):
            if ux[i][-j]==ux[i][1-j]:
                num+=1
            else:
                break
        onoff.append(k*num)
    return u0,p0,onoff

parser = argparse.ArgumentParser()
parser.add_argument('--instance', type=str, default='8_std')
args = parser.parse_args()
name = args.instance


file = 'UC_AF/'+name+'.mod'
Dt=np.loadtxt('deman/pd.csv',delimiter=',',encoding='utf-8-sig')
uc=UC(file)
N = uc.NumThermal
T = uc.HorizonLen
sys_up=sum(uc.ThPimax)*0.8
sys_low=sys_up*0.4
pd=(Dt-Dt.min())/(Dt.max()-Dt.min())*0.85*(sys_up-sys_low)+sys_low*1.05
TLE=60
CORE_NUM=1
n=365

cnt=0
u0_save=[]
p0_save=[]
onoff_save=[]
pmin = np.array(uc.ThPimin)

with Progress(
        "[progress.description]{task.description}({task.completed}/{task.total})",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.2f}%",
        TimeElapsedColumn(),
        '<',
        TimeRemainingColumn(),
    ) as progress:
    description = "generating instance"
    task = progress.add_task(description, total=n)
    for i in range(n):

        progress.update(task, completed=i, description=description)
        Dt=pd[i]
        Spin = Dt*0.1
        Dt = Dt.tolist()
        Spin = Spin.tolist()

        p0=uc.Pi0
        u0=uc.Ui0
        onoff=uc.ThTime_on_off_init

        data = {'Dt':Dt,'Spin':Spin,'u0':u0,'p0':p0,'on_off':onoff,'instance':name}

        m=uc.get_3bin_model(data)
        
        m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit",TLE)
        m.optimize()
        u0_save.append(u0)
        p0_save.append(p0)
        onoff_save.append(onoff)
        
        if m.status == 2 or m.status == 9:
            u0,p0,onoff = cal_upo(m)
            uc.change_state(p0=p0,onoff=onoff)
            cnt+=1
        progress.update(task, completed=cnt, description='finish instance making')


u0_save=np.array(u0_save)
p0_save=np.array(p0_save)
onoff_save=np.array(onoff_save)

os.makedirs(f'instances/fixed/{name}',exist_ok=True)
np.save(f'instances/fixed/{name}/u0_save',u0_save)
np.save(f'instances/fixed/{name}/p0_save',p0_save)
np.save(f'instances/fixed/{name}/onoff_save',onoff_save)

print(f'finish {name}')