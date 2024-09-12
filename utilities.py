import numpy as np
import torch
from gurobipy import GRB
from torch_geometric.data import HeteroData
import collections
import gurobipy as gp
import time

def cmp_sol(model, where):
    if where == gp.GRB.Callback.MIPSOL:
        t = time.time() - model._start_time        
        obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
        if len(model._objs)==0 or obj<model._objs[-1]:
            model._objs.append(obj)
            model._times.append(t)
            sol = np.round(model.cbGetSolution(model._u))
            model._sols.append(sol.copy())

def local_search(m,sol,relaxed,begin,TLE=10,out=True):
    m1 = m.copy()
    m1.setParam('TimeLimit', TLE)
    # m1.setParam('OutputFlag', out)
    m1.setParam('Threads', 1)
    # m1.setParam('MIPGap', 1e-5)
    u = m1.getVars()[:len(sol)]
    for i in range(len(sol)):
        if relaxed[i] == 0:
            m1.addConstr(u[i]==sol[i],name=f'fix_u[{i}]')
        else:
            u[i].setAttr('Start', sol[i])
    m1._start_time=begin
    m1._objs=[]
    m1._times=[]
    m1._sols = []
    m1._u=u
    # m1.optimize()
    if out:
        m1.optimize(cmp_sol)
    else:
        m1.optimize()
    if m1.status == 3:
        return -1,-1,-1
    sol = np.round(np.array(m1.getAttr('X', u)))
    return sol,m1._objs,m1._times

def solve(m,sol):
    sol=sol.reshape(-1)
    m1 = m.copy()
    u = m1.getVars()[:len(sol)]
    for i in range(len(sol)):
        m1.addConstr(u[i]==sol[i],name=f'fix_u[{i}]')
    m1.optimize()
    if m1.status==2 or m1.status==9:
        return m1.objVal
    else:
        print(m1.status)
    return 1e19+7

def get_norm(ori_tensor):
    ori_tensor = ori_tensor.reshape(-1,1)
    norm = torch.norm(ori_tensor, keepdim=True)
    norm_tensor = (ori_tensor/(norm+1e-6))
    return torch.cat((ori_tensor, norm_tensor), dim=1)

def get_Tripartite_graph_lp(m):
    r=m.relax()
    r.setParam('OutputFlag', 0)
    r.optimize()
    rvars=r.getVars()
    variables=[]
    constraints=[]
    mvars = m.getVars()
    constrs = m.getConstrs()
    for v in mvars:
        vname = v.VarName.split('[')[0]
        if vname not in variables:
            variables.append(vname)
    for c in constrs:
        cname = c.ConstrName.split('[')[0]
        cname = cname.rstrip('0123456789')
        if cname not in constraints:
            constraints.append(cname)
    A = m.getA()
    rows, cols = A.shape
    mvars = m.getVars()
    constrs = m.getConstrs()
    b_vars = torch.tensor([1 if 'u'in v.Varname else 0 for v in mvars])
    coo = A.tocoo()
    v0 = torch.tensor(m.getAttr("Obj", mvars)).reshape(-1,1)
    v1 = torch.tensor(coo.getnnz(axis=0)).reshape(-1,1)
    v2 = torch.tensor(coo.sum(axis=0)).reshape(-1,1)/v1
    v3 = torch.tensor(coo.max(axis=0).toarray()).reshape(-1,1)
    v4 = torch.tensor(coo.min(axis=0).toarray()).reshape(-1,1)
    v5 = b_vars.clone().reshape(-1,1)
    v6 = torch.tensor([v.x for v in rvars]).reshape(-1,1)
    v_nodes = torch.cat((v0,v1,v2,v3,v4,v5,v6),1).float()
    c0 = torch.tensor(coo.getnnz(axis=1)).reshape(-1,1)
    c1 = torch.tensor(coo.sum(axis=1)).reshape(-1,1)/c0
    c2 = torch.tensor(m.getAttr("RHS", constrs)).reshape(-1,1)
    c3 = torch.tensor([2 if i=='<' else 1 for i in m.getAttr("Sense", constrs)]).reshape(-1,1)
    c_nodes = torch.cat((c0,c1,c2,c3),1).float()

    tot=int(cols/len(variables))
    col_slides=collections.defaultdict(int)
    for v in variables:
        col_slides[v]=int(tot)
    row_slides=[]
    graph = HeteroData()

    row_slides = collections.defaultdict(int)
    cons = m.getConstrs()
    t=0
    coc=0
    for c in cons:
        name=constraints[t]
        while name not in c.ConstrName:
            t+=1
            name=constraints[t]
        if t>=len(constraints):
            break
        row_slides[name]+=1
    lastv=0
    for v in variables:
        graph[v].x=v_nodes[lastv:lastv+col_slides[v]]
        lastv+=col_slides[v]
    lastc=0
    for c in constraints:
        graph[c].x=c_nodes[lastc:lastc+row_slides[c]]
        lastc+=row_slides[c]
    lastr=0
    for c in constraints:
        lastc=0
        for v in variables:
            mtx = A[lastr:lastr+row_slides[c],lastc:lastc+col_slides[v]].tocoo()
            lastc+=col_slides[v]
            values = mtx.data
            if len(values)==0:
                continue
            row_indices = mtx.row
            col_indices = mtx.col
            indices = np.vstack((row_indices, col_indices))
            graph[c,'c2v',v].edge_index = torch.from_numpy(indices)
            graph[c,'c2v',v].edge_attr = torch.from_numpy(values).reshape(-1,1)
        lastr+=row_slides[c]

    
    #obj
    graph['obj'].x = torch.tensor([r.objVal]).reshape(1,-1)
    con_indice=[]
    con_attr=[]
    idx=0
    t=0
    for c in r.getConstrs():
        name=constraints[t]
        while name not in c.ConstrName:
            if 'edge_attr' in graph['obj','o2c',name]:
                graph['obj','o2c',name].edge_index = torch.tensor(con_indice).t()
                graph[name,'c2o','obj'].edge_index = torch.tensor(con_indice).t()[[1,0]]
                graph['obj','o2c',name].edge_attr = get_norm(torch.tensor(con_attr).reshape(-1,1))
            t+=1
            name=constraints[t]
            idx=0
            con_indice=[]
            con_attr=[]
        if t>=len(constraints):
            break
        
        if c.Slack < 1e-6:
            con_indice.append((0,idx))
            con_attr.append(c.RHS)
        idx+=1
    graph['obj','o2c',name].edge_index = torch.tensor(con_indice).t()
    graph[name,'c2o','obj'].edge_index = torch.tensor(con_indice).t()[[1,0]]
    graph['obj','o2c',name].edge_attr = get_norm(torch.tensor(con_attr).reshape(-1,1))
    #ov-edge
    for i,v in enumerate(['u','p','s']):
        graph['obj','o2v',v].edge_index = torch.cat((torch.zeros(1,tot, dtype=torch.int64), torch.arange(0,tot).unsqueeze(0)), 0)
        graph[v,'v2o','obj'].edge_index = torch.cat((torch.arange(0, tot).unsqueeze(0), torch.zeros(1, tot, dtype=torch.int64)), 0)
        graph['obj','o2v',v].edge_attr = get_norm(v0[tot*i:tot*(i+1)])

    return graph


def get_Tripartite_graph_lp_with_sol(m_o,sol):
    sol=sol.reshape(-1)
    m=m_o.copy()
    r=m.relax()
    r.setParam('OutputFlag', 0)
    rvars=r.getVars()
    mvars = m.getVars()
    constrs = m.getConstrs()
    rconstrs = r.getConstrs()
    A = m.getA()
    coo = A.tocoo()
    
    m_u=mvars[:len(sol)]
    m.addConstrs((m_u[i] == sol[i] for i in range(len(sol))), name='fixed')
    m.update()
    m.optimize()
    r.addConstrs((rvars[i] == sol[i] for i in range(len(sol))), name='fixed')
    r.update()
    r.optimize()
    constraints=[]
    variables=[]
    for v in mvars:
        vname = v.VarName.split('[')[0]
        if vname not in variables:
            variables.append(vname)
    for c in constrs:
        cname = c.ConstrName.split('[')[0]
        cname = cname.rstrip('0123456789')
        if cname not in constraints and cname!='fixed':
            constraints.append(cname)
    v0 = torch.tensor([v.Obj for v in mvars]).reshape(-1,1)
    v1 = torch.tensor(coo.getnnz(axis=0)).reshape(-1,1)
    v2 = torch.tensor(coo.sum(axis=0)).reshape(-1,1)/v1
    v3 = torch.tensor(coo.max(axis=0).toarray()).reshape(-1,1)
    v4 = torch.tensor(coo.min(axis=0).toarray()).reshape(-1,1)
    v5 = torch.tensor([1 if 'u'in v.Varname else 0 for v in mvars]).reshape(-1,1)#最关键变量
    v6 = torch.tensor([1 if v.vType == 'B' else 0 for v in mvars]).reshape(-1,1)
    v7 = torch.tensor([v.x for v in mvars]).reshape(-1,1)
    v8 = torch.tensor([v.lb for v in mvars]).reshape(-1,1)
    v9 = torch.tensor([v.ub for v in mvars]).reshape(-1,1)
    v10 = torch.tensor([1 if v.lb==v.x else 0 for v in mvars]).reshape(-1,1)
    v11 = torch.tensor([1 if v.ub==v.x else 0 for v in mvars]).reshape(-1,1)
    v12 = torch.tensor([min(v.ub-v.x,v.x-v.lb) for v in mvars]).reshape(-1,1)
    v13 = torch.tensor([v.x*v.Obj for v in mvars]).reshape(-1,1)
    
    v_nodes = torch.cat((v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13),1).float()

    c0 = torch.tensor(coo.getnnz(axis=1)).reshape(-1,1)
    c1 = torch.tensor(coo.sum(axis=1)).reshape(-1,1)/c0
    c2 = torch.tensor(m.getAttr("RHS", constrs)).reshape(-1,1)
    c3 = torch.tensor([2 if i=='<' else 1 for i in m.getAttr("Sense", constrs)]).reshape(-1,1)
    c4 = torch.tensor([c.Slack for c in constrs]).reshape(-1,1)
    c5 = torch.tensor([c.Pi for c in rconstrs]).reshape(-1,1)
    c_nodes = torch.cat((c0,c1,c2,c3,c4,c5),1).float()

    tot=len(sol)
    col_slides=collections.defaultdict(int)
    for v in variables:
        col_slides[v]=int(tot)
    row_slides=[]
    graph = HeteroData()

    row_slides = collections.defaultdict(int)
    t=0
    for c in constrs:
        name=constraints[t]
        while name not in c.ConstrName:
            t+=1
            name=constraints[t]
        if t>=len(constraints):
            break
        row_slides[name]+=1
    lastv=0
    for v in variables:
        graph[v].x=v_nodes[lastv:lastv+col_slides[v]]
        lastv+=col_slides[v]
    lastc=0
    for c in constraints:
        graph[c].x=c_nodes[lastc:lastc+row_slides[c]]
        lastc+=row_slides[c]
    lastr=0
    for c in constraints:
        lastc=0
        for v in variables:
            mtx = A[lastr:lastr+row_slides[c],lastc:lastc+col_slides[v]].tocoo()
            lastc+=col_slides[v]
            values = mtx.data
            if len(values)==0:
                continue
            row_indices = mtx.row
            col_indices = mtx.col
            indices = np.vstack((row_indices, col_indices))
            graph[c,'c2v',v].edge_index = torch.from_numpy(indices)
            graph[c,'c2v',v].edge_attr = torch.from_numpy(values).reshape(-1,1)
        lastr+=row_slides[c]

    #obj    
    graph['obj'].x = torch.tensor([r.objVal,m.objVal]).reshape(1,-1)
    con_indice=[]
    con_attr=[]
    idx=0
    t=0
    for c in rconstrs:
        name=constraints[t]
        while name not in c.ConstrName:
            if 'edge_attr' in graph['obj','o2c',name]:
                graph['obj','o2c',name].edge_index = torch.tensor(con_indice).t()
                graph[name,'c2o','obj'].edge_index = torch.tensor(con_indice).t()[[1,0]]
                graph['obj','o2c',name].edge_attr = get_norm(torch.tensor(con_attr).reshape(-1,1))
            t+=1
            name=constraints[t]
            idx=0
            con_indice=[]
            con_attr=[]
        if t>=len(constraints):
            break
        
        if c.Slack < 1e-6:
            con_indice.append((0,idx))
            con_attr.append(c.RHS)
        idx+=1
    graph['obj','o2c',name].edge_index = torch.tensor(con_indice).t()
    graph[name,'c2o','obj'].edge_index = torch.tensor(con_indice).t()[[1,0]]
    graph['obj','o2c',name].edge_attr = get_norm(torch.tensor(con_attr).reshape(-1,1))
    #ov-edge
    for i,v in enumerate(['u','p','sc']):
        graph['obj','o2v',v].edge_index = torch.cat((torch.zeros(1,tot, dtype=torch.int64), torch.arange(0,tot).unsqueeze(0)), 0)
        graph[v,'v2o','obj'].edge_index = torch.cat((torch.arange(0, tot).unsqueeze(0), torch.zeros(1, tot, dtype=torch.int64)), 0)
        graph['obj','o2v',v].edge_attr = get_norm(v0[tot*i:tot*(i+1)])


    sum=1
    for v in ['u','p','s','d','sc']:
        if 'x' in graph[v]:
            sum+=graph[v].num_nodes
        else:
            graph[v].num_nodes = 0
    for c in ['Minimum_up/down_time_constraints','Unit_generation_limits','Power_balance_constrains','System_spinning_reserve_requirement','Ramp_rate_limits','Initial_status_of_units','startup_cost','state_variable']:
        if 'x' in graph[c]:
            sum+=graph[c].num_nodes
        else:
            graph[c].num_nodes = 0
    graph.num_nodes = sum
    return graph


def restore_initial(ans,p0,rd,sd,on_off,on,off):
    N=ans.shape[0]
    T=ans.shape[1]
    for i in range(N):
        if on_off[i]<0:
            for t in range(off[i]+on_off[i]):
                ans[i][t]=-1
        elif on_off[i]>0:
            for t in range(on[i]-on_off[i]):
                ans[i][t]=1
        if p0[i]>sd[i]:
            for t in range(int((p0[i]-sd[i]+rd[i]-1)//rd[i])):
                ans[i][t]=1
        if on_off[i]<0 and -on_off[i]<off[i]:
            res = off[i]+on_off[i]
            for t in range(res):
                ans[i][t]=-1
        if on_off[i]>0 and on_off[i]<on[i]:
            res = on[i]-on_off[i]
            for t in range(res):
                ans[i][t]=1
    return ans

def restore_spin(ans_o,spin,dt,pmax):
    ans=ans_o.copy()
    N=ans.shape[0]
    T=ans.shape[1]
    for t in range(T):
        cur_max = 0
        for i in range(N):
            if ans[i][t]>1-1e-5:
                cur_max+=pmax[i]
        if cur_max<spin[t]+dt[t]:
            p_list=[]
            for i in range(N):
                if ans[i][t]>=0 and ans[i][t]<=1-1e-5:
                    p_list.append((ans[i,t],i))
            p_list.sort(key=lambda x:x[0],reverse=True)
            for p,i in p_list:
                cur_max+=pmax[i]
                ans[i][t]=1
                if cur_max>=spin[t]+dt[t]:
                    break
    return ans

def restore_on_off(ans,ans_o,on_off,on,off):
    N=ans.shape[0]
    T=ans.shape[1]
    for i in range(N):
        t=0
        while t<T:
            cnt1=0
            l=0
            while t+l<T and ans[i,t+l]==1:
                l+=1
                cnt1+=1
            if t+l>=T:
                break
            if cnt1>0 and t==0:
                cnt1+=max(0,on_off[i])
            if cnt1>0 and cnt1<on[i]:
                res = on[i]-cnt1
                dealed=0
                for ll in range(1,res+1):
                    idx = t-ll
                    if idx<0 or ans_o[i,idx]==-1 or ans[i,idx]==1:
                        break
                    dealed+=1
                    ans[i,idx]=1
                    ans_o[i,idx]=ans_o[i,t]
                res-=dealed
                for ll in range(1,res+1):
                    idx = t+l+ll
                    if idx>=T or ans[i,idx]==1:
                        break
                    ans[i,idx]=1
                    ans_o[i,idx]=ans_o[i,t]
            t+=l
            while t<T and ans[i,t]!=1:
                t+=1
        t=0
        while t<T:
            cnt0=0
            l=0
            while t+l<T and ans[i,t+l]<1:
                l+=1
                cnt0+=1
            if t+l>=T:
                break
            if cnt0>0 and t==0:
                cnt0+=max(0,-on_off[i])

            if cnt0>0 and cnt0<off[i]:
                for ll in range(l):
                    ans[i,t+ll]=1
                    ans_o[i,t+ll]=ans_o[i,t]
            t+=l
            while t<T and ans[i,t]>=1:
                t+=1
    return ans


def early_startup(ans,sol_o,res,t,l2r,su):
    N=ans.shape[0]
    T=ans.shape[1]
    sum=0
    for i in range(N):
        if l2r[i,t]<0:
            res-=su[i]
            sum+=su[i]
            ans[i,t]=1
            if t+1<T:
                sol_o[i,t]=sol_o[i,t+1]
        if res<=0:
            break
    return ans,sum

def delay_shutdown(ans,sol_o,res,t,r2l,sd):
    N=ans.shape[0]
    sum=0
    for i in range(N):
        if r2l[i,t]<0:
            res-=sd[i]
            sum+=sd[i]
            ans[i,t]=1
            if t-1>=0:
                sol_o[i,t]=sol_o[i,t-1]
        if res<=0:
            break
    return ans,sum

def com_cont_on_off(ans,ans_o,on_off,off):
    N=ans.shape[0]
    T=ans.shape[1]
    r2l_off = np.zeros_like(ans)
    l2r_off = np.zeros_like(ans)
    for i in range(N):
        if on_off[i]<0:
            l2r_off[i,0]=off[i]+on_off[i]-1
        elif ans[i,0]<=0:
            l2r_off[i,0]=off[i]-1
        for t in range(1,T):
            if ans[i,t-1]<1 and ans[i,t]<1:
                l2r_off[i,t]=l2r_off[i,t-1]-1
            elif ans[i,t-1]==1 and ans[i,t]<1:
                l2r_off[i,t]=off[i]-1
    for i in range(N):
        if ans[i,T-1]<1:
            r2l_off[i,T-1]=-1
        for t in range(T-2,-1,-1):
            if ans[i,t+1]<1 and ans[i,t]<1 and ans_o[i,t]>=0:
                r2l_off[i,t]=r2l_off[i,t+1]-1
            elif ans[i,t+1]==1 and ans[i,t]<1:
                r2l_off[i,t]=off[i]-1
    return r2l_off,l2r_off

def restore_ramp(u_close,sol_o,Dt,Ui0,Pi0,Pidown,Piup,ThPimin,ThPimax,Pistartup,Pishutdown,on_off,off):
    u = u_close.copy()
    u[u < 1] = 0
    N=u.shape[0]
    T=u.shape[1]
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        m = gp.Model(env=env)
    p = m.addVars(N, T, vtype=GRB.CONTINUOUS, name='p')
    m.addConstrs((u[i,t]*ThPimin[i] - p[i,t] <= 0  for i in range(N) for t in range (T)), name="Unit_generation_limits1" )
    m.addConstrs((p[i,t] <= u[i,t]*ThPimax[i]  for i in range(N) for t in range (T)), name="Unit_generation_limits2" )
    m.addConstrs((p[i,t]-(p[i,t-1]if t>0 else Pi0[i]) <= (u[i,t-1] if t>0 else Ui0[i])*Piup[i] + (u[i,t]-(u[i,t-1] if t>0 else Ui0[i]))*Pistartup[i] +(1-u[i,t])*ThPimax[i]  for i in range(N) for t in range(T)),name="Ramp_rate_limits1" )
    m.addConstrs(((p[i,t-1]if t>0 else Pi0[i])-p[i,t] <= u[i,t]*Pidown[i] + ((u[i,t-1] if t>0 else Ui0[i])-u[i,t])*Pishutdown[i] +(1-(u[i,t-1] if t>0 else Ui0[i]))*ThPimax[i]  for i in range(N) for t in range(T)),name="Ramp_rate_limits2")
    m.addConstrs((gp.quicksum(p[i, t] for i in range(N)) - Dt[t]  <= 0 for t in range(T)),name="Power_balance_constrains")
    m.setObjective(gp.quicksum(Dt[t] - gp.quicksum(p[i,t] for i in range(N)) for t in range(T)), GRB.MINIMIZE)
    m.update()        
    m.optimize()
    if m.status!=2:
        return u_close
    p_sum=[0]*T
    for i in range(N):
        for t in range(T):
            p_sum[t]+=p[i,t].x
    r2l,l2r = com_cont_on_off(u_close,sol_o,on_off,off)
    for t in range(T):
        if p_sum[t]+1e-5<Dt[t]:#ans,res,t,r2l,sd
            u_close,res = delay_shutdown(u_close,sol_o,Dt[t]-p_sum[t],t,r2l,Pishutdown)
            p_sum[t]+=res
    for t in range(T-1,-1,-1):
        if p_sum[t]+1e-5<Dt[t]:
            u_close,res = early_startup(u_close,sol_o,Dt[t]-p_sum[t],t,l2r,Pistartup)
            p_sum[t]+=res
    for t in range(T):
        if p_sum[t]<Dt[t]:
            for i in range(N):
                if u_close[i,t]==1:
                    if t>0 and u_close[i,t-1]<1 and sol_o[i,t-1]>=0:
                        u_close[i,t-1]=1
                        p_sum[t]+=Piup[i]
                        if p_sum[t]>=Dt[t]:
                            break
                    elif t<T-1 and u_close[i,t+1]<1:
                        u_close[i,t+1]=1
                        p_sum[t]+=Pidown[i]
                        if p_sum[t]>=Dt[t]:
                            break
        if p_sum[t]<Dt[t]:
            p_list=[]
            for i in range(N):
                if sol_o[i,t]>=0 and u_close[i,t]<1:
                    p_list.append((sol_o[i,t],i))
            p_list.sort(key=lambda x:x[0],reverse=True)
            for p,i in p_list:
                u_close[i,t]=1
                p_sum[t]+=Pistartup[i]
                if p_sum[t]>=Dt[t]:
                    break
                if t>0 and u_close[i,t-1]<1 and sol_o[i,t-1]>=0:
                    u_close[i,t-1]=1
                    p_sum[t]+=Piup[i]
                elif t<T-1 and u_close[i,t+1]<1:
                    u_close[i,t+1]=1
                    p_sum[t]+=Pidown[i]
                if p_sum[t]>=Dt[t]:
                    break
        if p_sum[t]<Dt[t]:
            print('fail meet ramp constraint')
    return u_close

def feasibility_pump(mo,sol):
    ans = sol.copy()
    ans = ans.reshape(-1)
    m=mo.copy()
    v = m.getVars()[:len(ans)]
    ans = np.round(ans)
    alpha = []
    for i in range(len(ans)):
        if ans[i]==0 or ans[i]==1:
            v[i].setAttr('Start', ans[i])
            tmp_var = m.addVar(vtype=GRB.CONTINUOUS,name=f'alpha{v[i].VarName}')
            alpha.append(tmp_var) 
            m.addConstr(tmp_var >= v[i] - ans[i], name=f'alpha_up_{v[i].VarName}')
            m.addConstr(tmp_var >= ans[i] - v[i], name=f'alpha_dowm_{v[i].VarName}')
            # m.addConstr(instance_variabels[i] == ans[i], name=f'alpha_{instance_variabels[i].VarName}')
    m.setObjective(sum(alpha), GRB.MINIMIZE)
    m.setParam('MIPFocus', 1)
    m.setParam('SolutionLimit', 1)
    # m.setParam('TimeLimit', 1000)
    m.update()
    m.optimize()
    if m.status==3 or m.status==4:
        return None
    return np.round(np.array(m.getAttr('X', v))).reshape(-1).astype(int)


def restore(sol_o, uc, data, m3):
    sol = sol_o.copy().reshape(-1,24)
    Dt = data['Dt']
    Spin = data['Spin']
    u0 = data['u0']
    p0 = data['p0']
    on_off = data['on_off']
    shut_down = uc.Pishutdown
    startup = uc.Pistartup
    on = uc.ThTime_on_min
    off = uc.ThTime_off_min
    pmax = uc.ThPimax
    pmin = uc.ThPimin
    ramp_up = uc.Piup
    ramp_down = uc.Pidown
    sol = restore_initial(sol,p0,ramp_down,shut_down,on_off,on,off)
    sol_s = restore_spin(sol,Spin,Dt,pmax)
    sol_s[sol_s >= 1] = 1
    sol_s[sol_s < 1] = 0
    sol[sol >= 1] = 1
    sol_s = restore_on_off(sol_s,sol,on_off,on,off)
    sol_s = restore_ramp(sol_s,sol,Dt,u0,p0,ramp_down,ramp_up,pmin,pmax,startup,shut_down,on_off,off)
    sol_s = restore_on_off(sol_s,sol,on_off,on,off)
    last = solve(m3,sol_s)
    if last>1e19:
        sol_s = feasibility_pump(m3,sol_o)
        last = solve(m3,sol_s)
    return sol.reshape(-1), sol_s.reshape(-1).astype(int), last



from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import DotProductSimilarity

def train_cl(predict, data_loader, DEVICE, optimizer=None):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    infoNCE_loss_function = losses.NTXentLoss(temperature=0.07,distance=DotProductSimilarity()).to(DEVICE)

    if optimizer:
        predict.train()
    else:
        predict.eval()
    mean_loss = 0
    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for step, batch in enumerate(data_loader):

            batch = batch.to(DEVICE)
            embeddings = predict(batch).reshape(len(batch), -1)
            anchor_positive = torch.tensor([i for i in range(len(batch)) for j in range(len(batch.positive[i]))])
            anchor_negative = torch.tensor([i for i in range(len(batch)) for j in range(len(batch.negative[i]))])
            positive_idx = torch.arange(len(batch),len(batch)+len(anchor_positive))
            negative_idx = torch.arange(len(batch)+len(anchor_positive),len(batch)+len(anchor_positive)+len(anchor_negative))
            positive = torch.from_numpy(np.concatenate(batch.positive)).to(DEVICE)
            negative = torch.from_numpy(np.concatenate(batch.negative)).to(DEVICE)
            embeddings = torch.cat((embeddings, positive, negative),0)
            triplets = (anchor_positive.to(DEVICE), positive_idx.to(DEVICE), anchor_negative.to(DEVICE), negative_idx.to(DEVICE))
            loss = infoNCE_loss_function(embeddings, indices_tuple = triplets)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            mean_loss += loss.item()
            n_samples_processed += 1
    mean_loss /= n_samples_processed
    return mean_loss

def get_edge(sol,on):
    sol = sol.reshape(-1,24)
    N = sol.shape[0]
    T = sol.shape[1]
    edge = np.zeros_like(sol)
    for i in range(N):
        for j in range(T-1):
            if sol[i][j]!=sol[i][j+1]:
                shift = on[i]//2
                length = on[i]//2
                start = max(0,j-shift)
                end = min(T,j+length)
                edge[i][start:end]=1
    return edge.reshape(-1)

def local_descend(global_descend, local, old_sol, new_sol):
    last_round_descend = np.ones_like(local).astype(float)
    indices = np.where(local == 1)[0]
    selected_indices = np.random.choice(indices, size=int(len(indices) * 1), replace=False)
    global_descend[selected_indices] *= 0.9
    last_round_descend[selected_indices] *= 0.5
    indices = np.where(old_sol != new_sol)[0]
    selected_indices = np.random.choice(indices, size=int(len(indices) * 1), replace=False)
    global_descend[selected_indices] *= 0.8
    last_round_descend[selected_indices] *= 0.01
    last_round_descend*=global_descend
    return global_descend,last_round_descend



def local_search_repeat(m, search_model, sol, begin, on, opt=0,out=10,TLE=600,search_time=30,log=False,no_gain_exit=False):
    
    percentage=80
    last_round_descend = np.ones_like(sol).astype(float)
    global_descend = np.ones_like(sol).astype(float)
    cur_time = time.time()
    time_left = TLE-(cur_time-begin)
    time_round=[]
    obj_round=[]
    last=np.inf
    for cnt in range(out):
        if time_left<=1:
            break
        st = time.time()
        Tri_graph = get_Tripartite_graph_lp_with_sol(m,sol)
        t1 = time.time()
        structTime = t1-st
        begin+=structTime
        neighbor = search_model(Tri_graph).detach().cpu().numpy().reshape(-1)
        neighbor *= last_round_descend
        threshold_num = np.percentile(neighbor, percentage)
        local = np.zeros_like(neighbor,dtype=int)
        res = int((1-percentage/100)*len(neighbor))
        for i in range(len(neighbor)):
            if neighbor[i]>=threshold_num and res>0:
                local[i]=1
                res-=1
        
        edge = get_edge(sol,on)
        indices = np.where(edge == 1)
        local[indices] = 1
        new_sol,new_objs,new_times = local_search(m, sol, local, begin, TLE=min(search_time,max(5,time_left)))
        new = new_objs[-1]
        cur_time = time.time()
        global_descend,last_round_descend = local_descend(global_descend, local, sol, new_sol)
        sol = new_sol
        time_round.extend(new_times)
        obj_round.extend(new_objs)
        time_left = max(0,TLE-(cur_time-begin))
        if no_gain_exit and (last-new)/last<=1e-6:
            break
        if (last-new)/last<=1e-6:
            percentage*=0.9
            percentage=max(percentage,70)
        else:
            percentage*=1.1
            percentage=min(percentage,90)

        last=new
    return new,cnt,time_round,obj_round,sol