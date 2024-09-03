import gurobipy as gp
import numpy as np
from gurobipy import GRB

class UC:
    def __init__(self, file):
        with open(file, "r") as f:
            lines = f.readlines()
        self.file = file
        self.HorizonLen = int(lines[1].split()[1])  # 时段
        self.NumThermal = int(lines[2].split()[1])  # 火电机组数量
        self.Dt = np.array([float(x) for x in lines[10].split()])  # 负荷
        self.Spin = np.array([float(x) for x in lines[12].split()])  # 备用

        self.MaxThermalNum = self.NumThermal * 1.1
        self.MinThermalNum = self.NumThermal * 0.9

        self.units = [[float(x) for x in line.split()] for line in lines[14:14 + self.NumThermal * 2:2]]  # 读取机组信息

        self.RampConstraints = [[x for x in line.split()] for line in lines[15:15 + self.NumThermal * 2:2]]  # 读取爬坡信息

        for i in range(len(self.units)):
            self.units[i] += [float(self.RampConstraints[i][1]), float(self.RampConstraints[i][2])]

        # 去重
        self.unique_units = {tuple(sub_array) for sub_array in self.units}
        self.unique_units = [list(t) for t in self.unique_units]
        #计算出现次数
        self.nums=[1]
        for i in range(1,len(self.units)):
            if self.units[i][0]==self.units[i-1][0]:
                self.nums[-1]+=1
            else:
                self.nums.append(1)
        
        
        for unit in self.unique_units:
            unit[6] = -unit[8]
            unit[15] = 0

        self.refresh()

    def refresh(self):
        self.NumThermal = len(self.units)
        self.gamma = np.array([row[1] for row in self.units])  # 机组参数
        self.beta = np.array([row[2] for row in self.units])  # 机组参数
        self.alpha = np.array([row[3] for row in self.units])  # 机组参数
        self.ThPimin = np.array([row[4] for row in self.units])  # 发电下界
        self.ThPimax = np.array([row[5] for row in self.units])  # 发电上界
        self.ThTime_on_off_init = np.array([int(row[6]) for row in self.units])  # 初始状态前以及开机/停机的时间
        self.ThTime_on_min = np.array([int(row[7]) for row in self.units])  # 最小开机时间
        self.ThTime_off_min = np.array([int(row[8]) for row in self.units])  # 最小关机时间
        self.fixedCost4startup = np.array([row[13] for row in self.units])  # 启动费用
        self.Pi0 = np.array([row[15] for row in self.units])  # 初始发电功率
        self.Ui0 = np.array([1 if init > 0 else 0 for init in self.Pi0])  # 初始运行状态
        if 'std' in self.file:
            self.Tcoldi = np.array([int(row[16]) for row in self.units])  # 冷却时间
            self.Piup = np.array([float(row[17]) for row in self.units])  # 上坡功率
            self.Pidown = np.array([float(row[18]) for row in self.units])  # 下坡功率
        else:
            self.Tcoldi = np.array([1 for row in self.units])
            self.Piup = np.array([float(row[16]) for row in self.units])  # 上坡功率
            self.Pidown = np.array([float(row[17]) for row in self.units])  # 下坡功率
        self.Pistartup = self.ThPimin
        self.Pishutdown = self.ThPimin

        self.hoti = self.fixedCost4startup  # 热启动价格
        self.coldi = self.fixedCost4startup*2  # 冷启动价格

    def change_state(self, p0, onoff):
        for i in range(len(self.units)):
            self.units[i][15] = p0[i]
            self.units[i][6] = onoff[i]
        self.refresh()

    def get_state(self):
        return self.Pi0, self.Ui0, self.ThTime_on_off_init


    def get_3bin_model(self, data):
        self.Dt = data['Dt']
        self.Spin = data['Spin']
        self.Ui0 = data['u0']
        self.Pi0 = data['p0']
        self.ThTime_on_off_init = data['on_off']

        Ui = np.maximum(0,np.minimum(np.ones((self.NumThermal)) * self.HorizonLen,self.Ui0 * (self.ThTime_on_min - self.ThTime_on_off_init))).astype(int)                    #--N*1矩阵 
        Li = np.maximum(0,np.minimum(np.ones((self.NumThermal)) * self.HorizonLen,(np.ones((self.NumThermal)) - self.Ui0) * (self.ThTime_off_min + self.ThTime_on_off_init))).astype(int)    #--N*1矩阵

        Ndi = [self.ThTime_off_min[i] + self.Tcoldi[i] + 1 for i in range(self.NumThermal)]
        # m = gp.Model("3-bin UC formulation")
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            m = gp.Model(env=env)
            u = m.addVars(self.NumThermal, self.HorizonLen, vtype=GRB.BINARY, name="u")  # N行T列
            s = m.addVars(self.NumThermal, self.HorizonLen, vtype=GRB.BINARY, name="s")  # N行T列
            d = m.addVars(self.NumThermal, self.HorizonLen, vtype=GRB.BINARY, name="d")  # N行T列
            p = m.addVars(self.NumThermal, self.HorizonLen, ub=[self.ThPimax[i] for i in range(self.NumThermal)for t in range(self.HorizonLen)], vtype=GRB.CONTINUOUS, name="p")  # N行T列
            sc = m.addVars(self.NumThermal, self.HorizonLen, ub=[self.coldi[i] for i in range(self.NumThermal)for t in range(self.HorizonLen)], vtype=GRB.CONTINUOUS, name="sc")  # N行T列
            # 12
            m.addConstrs((s[i, t] - d[i, t] == u[i, t] - (u[i, t - 1] if t > 0 else self.Ui0[i]) for i in range(self.NumThermal) for t in range(self.HorizonLen)), name="state_variable")

            #Unit generation capacity limits constrains	3
            m.addConstrs((u[i,t]*self.ThPimin[i] <= p[i,t]  for i in range(self.NumThermal) for t in range (self.HorizonLen)), name="Unit_generation_limits1" )
            m.addConstrs((p[i,t] <= u[i,t]*self.ThPimax[i]  for i in range(self.NumThermal) for t in range (self.HorizonLen)), name="Unit_generation_limits2" )

            # Power balance constrains 4
            m.addConstrs((gp.quicksum(p[i, t] for i in range(self.NumThermal)) == self.Dt[t] for t in range(self.HorizonLen)), name="Power_balance_constrains")

            # System spinning reserve requirement 5
            m.addConstrs((gp.quicksum(u[i, t] * -self.ThPimax[i] for i in range(self.NumThermal)) <= -self.Dt[t] - self.Spin[t] for t in range(self.HorizonLen)), name="System_spinning_reserve_requirement")

            # Ramp rate limits 17 18
            m.addConstrs((p[i,t]-(p[i,t-1] if t > 0 else self.Pi0[i])<=u[i,t]*(self.Piup[i]+self.ThPimin[i])-(u[i,t-1] if t > 0 else self.Ui0[i]) * self.ThPimin[i] + s[i,t] * (self.Pistartup[i] - self.Piup[i] - self.ThPimin[i]) for i in range(self.NumThermal) for t in range(self.HorizonLen)), name="Ramp_rate_limits1")
            m.addConstrs(((p[i, t - 1] if t > 0 else self.Pi0[i]) - p[i, t] <= (u[i, t - 1] if t > 0 else self.Ui0[i])* (self.Pidown[i] + self.ThPimin[i]) - u[i, t] * self.ThPimin[i] + d[i, t] * (self.Pishutdown[i] - self.Pidown[i] - self.ThPimin[i]) for i in range(self.NumThermal)for t in range(self.HorizonLen)), name="Ramp_rate_limits2")

            # Minimum up/down time constraints 13 14
            m.addConstrs((gp.quicksum(s[i, w] for w in range(max(0, t + 1 - self.ThTime_on_min[i]), t + 1)) <= u[i, t] for i in range(self.NumThermal) for t in range(Ui[i], self.HorizonLen)), name="Minimum_up/down_time_constraints1")
            m.addConstrs((gp.quicksum(d[i, w] for w in range(max(0, t + 1 - self.ThTime_off_min[i]), t + 1)) <= 1 - u[i, t] for i in range(self.NumThermal) for t in range(Li[i], self.HorizonLen)), name="Minimum_up/down_time_constraints2")

            # Initial status of units 10
            m.addConstrs((u[i, t] == self.Ui0[i] for i in range(self.NumThermal) for t in range(Ui[i] + Li[i])),name="Initial_status_of_units")

            # startup cost 19 20
            m.addConstrs((-sc[i, t] <= -self.hoti[i] * s[i, t] for i in range(self.NumThermal) for t in range(self.HorizonLen)),name="startup_cost1")
            m.addConstrs((-sc[i,t] <= -self.coldi[i]*(s[i,t]-gp.quicksum(d[i,w] for w in range(max(0,t-self.ThTime_off_min[i]-self.Tcoldi[i]),t)) - (1 if t+1-self.ThTime_off_min[i]-self.Tcoldi[i] <= 0 and max(0,-self.ThTime_on_off_init[i])<abs(t-self.ThTime_off_min[i]-self.Tcoldi[i])+1  else 0)) for i in range(self.NumThermal) for t in range (self.HorizonLen)),name="startup_cost2")
            
            m.setObjective(gp.quicksum(self.alpha[i] * u[i, t] + self.beta[i] * p[i, t] + sc[i, t] for i in range(self.NumThermal) for t in range(self.HorizonLen)), GRB.MINIMIZE)
            # m.setObjective(gp.quicksum(self.alpha[i]*u[i,t] + self.beta[i]*p[i,t] + self.gamma[i]*p[i,t]*p[i,t] + sc[i,t] for i in range(self.NumThermal) for t in range(self.HorizonLen)),GRB.MINIMIZE)

            m.update()
            return m
    def get_3bin_model_1bin_startupcost(self, Dt=None, Spin=None, ThTime_on_off_init=None, Ui0=None, Pi0=None):
        if Dt is not None:
            self.Dt = Dt
        if Spin is not None:
            self.Spin = Spin
        if ThTime_on_off_init is not None:
            self.ThTime_on_off_init = ThTime_on_off_init
        if Ui0 is not None:
            self.Ui0 = Ui0
        if Pi0 is not None:
            self.Pi0=Pi0

        Ui = np.maximum(0,np.minimum(np.ones((self.NumThermal)) * self.HorizonLen,self.Ui0 * (self.ThTime_on_min - self.ThTime_on_off_init))).astype(int)                    #--N*1矩阵 
        Li = np.maximum(0,np.minimum(np.ones((self.NumThermal)) * self.HorizonLen,(np.ones((self.NumThermal)) - self.Ui0) * (self.ThTime_off_min + self.ThTime_on_off_init))).astype(int)    #--N*1矩阵

        Ndi = [self.ThTime_off_min[i] + self.Tcoldi[i] + 1 for i in range(self.NumThermal)]
        # m = gp.Model("3-bin UC formulation")
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            m = gp.Model(env=env)
            u = m.addVars(self.NumThermal, self.HorizonLen, vtype=GRB.BINARY, name="u")  # N行T列
            s = m.addVars(self.NumThermal, self.HorizonLen, vtype=GRB.BINARY, name="s")  # N行T列
            d = m.addVars(self.NumThermal, self.HorizonLen, vtype=GRB.BINARY, name="d")  # N行T列
            p = m.addVars(self.NumThermal, self.HorizonLen, ub=[self.ThPimax[i] for i in range(self.NumThermal)for t in range(self.HorizonLen)], vtype=GRB.CONTINUOUS, name="p")  # N行T列
            sc = m.addVars(self.NumThermal, self.HorizonLen, ub=[self.coldi[i] for i in range(self.NumThermal)for t in range(self.HorizonLen)], vtype=GRB.CONTINUOUS, name="sc")  # N行T列
           
            # 12
            m.addConstrs((s[i, t] - d[i, t] == u[i, t] - (u[i, t - 1] if t > 0 else self.Ui0[i]) for i in range(self.NumThermal) for t in range(self.HorizonLen)), name="state_variable")

            #Unit generation capacity limits constrains	3
            m.addConstrs((u[i,t]*self.ThPimin[i] <= p[i,t]  for i in range(self.NumThermal) for t in range (self.HorizonLen)), name="Unit_generation_limits1" )
            m.addConstrs((p[i,t] <= u[i,t]*self.ThPimax[i]  for i in range(self.NumThermal) for t in range (self.HorizonLen)), name="Unit_generation_limits2" )

            # Power balance constrains 4
            m.addConstrs((gp.quicksum(p[i, t] for i in range(self.NumThermal)) == self.Dt[t] for t in range(self.HorizonLen)), name="Power_balance_constrains")

            # System spinning reserve requirement 5
            m.addConstrs((gp.quicksum(u[i, t] * -self.ThPimax[i] for i in range(self.NumThermal)) <= -self.Dt[t] - self.Spin[t] for t in range(self.HorizonLen)), name="System_spinning_reserve_requirement")

            # Ramp rate limits 17 18
            m.addConstrs((p[i,t]-(p[i,t-1] if t > 0 else self.Pi0[i])<=u[i,t]*(self.Piup[i]+self.ThPimin[i])-(u[i,t-1] if t > 0 else self.Ui0[i]) * self.ThPimin[i] + s[i,t] * (self.Pistartup[i] - self.Piup[i] - self.ThPimin[i]) for i in range(self.NumThermal) for t in range(self.HorizonLen)), name="Ramp_rate_limits1")
            m.addConstrs(((p[i, t - 1] if t > 0 else self.Pi0[i]) - p[i, t] <= (u[i, t - 1] if t > 0 else self.Ui0[i])* (self.Pidown[i] + self.ThPimin[i]) - u[i, t] * self.ThPimin[i] + d[i, t] * (self.Pishutdown[i] - self.Pidown[i] - self.ThPimin[i]) for i in range(self.NumThermal)for t in range(self.HorizonLen)), name="Ramp_rate_limits2")

            # Minimum up/down time constraints 13 14
            m.addConstrs((gp.quicksum(s[i, w] for w in range(max(0, t + 1 - self.ThTime_on_min[i]), t + 1)) <= u[i, t] for i in range(self.NumThermal) for t in range(Ui[i], self.HorizonLen)), name="Minimum_up/down_time_constraints1")
            m.addConstrs((gp.quicksum(d[i, w] for w in range(max(0, t + 1 - self.ThTime_off_min[i]), t + 1)) <= 1 - u[i, t] for i in range(self.NumThermal) for t in range(Li[i], self.HorizonLen)), name="Minimum_up/down_time_constraints2")

            # Initial status of units 10
            m.addConstrs((u[i, t] == self.Ui0[i] for i in range(self.NumThermal) for t in range(Ui[i] + Li[i])),name="Initial_status_of_units")

            # startup cost 19 20
            # m.addConstrs((-sc[i, t] <= -self.hoti[i] * s[i, t] for i in range(self.NumThermal) for t in range(self.HorizonLen)),name="startup_cost1")
            # m.addConstrs((-sc[i,t] <= -self.coldi[i]*(s[i,t]-gp.quicksum(d[i,w] for w in range(max(0,t-self.ThTime_off_min[i]-self.Tcoldi[i]),t)) - (1 if t+1-self.ThTime_off_min[i]-self.Tcoldi[i] <= 0 and max(0,-self.ThTime_on_off_init[i])<abs(t-self.ThTime_off_min[i]-self.Tcoldi[i])+1  else 0)) for i in range(self.NumThermal) for t in range (self.HorizonLen)),name="startup_cost2")
            #from 1bin
            m.addConstrs((-sc[i,t] <= -1*(self.hoti[i] if l<=self.ThTime_off_min[i]+self.Tcoldi[i] else self.coldi[i])* (u[i,t]-gp.quicksum((u[i,t-j]) if t>=j else(1 if j-t<=self.ThTime_on_off_init[i] or (self.ThTime_on_off_init[i]<0 and j-t>-self.ThTime_on_off_init[i]) else 0) for j in range(1,l+1))) for i in range(self.NumThermal) for t in range(self.HorizonLen) for l in [1,Ndi[i]]),name="startup_cost")

            m.setObjective(gp.quicksum(self.alpha[i] * u[i, t] + self.beta[i] * p[i, t] + sc[i, t] for i in range(self.NumThermal) for t in range(self.HorizonLen)), GRB.MINIMIZE)
            # m.setObjective(gp.quicksum(self.alpha[i]*u[i,t] + self.beta[i]*p[i,t] + self.gamma[i]*p[i,t]*p[i,t] + sc[i,t] for i in range(self.NumThermal) for t in range(self.HorizonLen)),GRB.MINIMIZE)

            m.update()
            return m
    def get_3bin_model_1bin_ramp(self, Dt=None, Spin=None, ThTime_on_off_init=None, Ui0=None, Pi0=None):
        if Dt is not None:
            self.Dt = Dt
        if Spin is not None:
            self.Spin = Spin
        if ThTime_on_off_init is not None:
            self.ThTime_on_off_init = ThTime_on_off_init
        if Ui0 is not None:
            self.Ui0 = Ui0
        if Pi0 is not None:
            self.Pi0=Pi0

        Ui = np.maximum(0,np.minimum(np.ones((self.NumThermal)) * self.HorizonLen,self.Ui0 * (self.ThTime_on_min - self.ThTime_on_off_init))).astype(int)                    #--N*1矩阵 
        Li = np.maximum(0,np.minimum(np.ones((self.NumThermal)) * self.HorizonLen,(np.ones((self.NumThermal)) - self.Ui0) * (self.ThTime_off_min + self.ThTime_on_off_init))).astype(int)    #--N*1矩阵

        Ndi = [self.ThTime_off_min[i] + self.Tcoldi[i] + 1 for i in range(self.NumThermal)]
        # m = gp.Model("3-bin UC formulation")
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            m = gp.Model(env=env)
            u = m.addVars(self.NumThermal, self.HorizonLen, vtype=GRB.BINARY, name="u")  # N行T列
            s = m.addVars(self.NumThermal, self.HorizonLen, vtype=GRB.BINARY, name="s")  # N行T列
            d = m.addVars(self.NumThermal, self.HorizonLen, vtype=GRB.BINARY, name="d")  # N行T列
            p = m.addVars(self.NumThermal, self.HorizonLen, ub=[self.ThPimax[i] for i in range(self.NumThermal)for t in range(self.HorizonLen)], vtype=GRB.CONTINUOUS, name="p")  # N行T列
            sc = m.addVars(self.NumThermal, self.HorizonLen, ub=[self.coldi[i] for i in range(self.NumThermal)for t in range(self.HorizonLen)], vtype=GRB.CONTINUOUS, name="sc")  # N行T列
           
            # 12
            m.addConstrs((s[i, t] - d[i, t] == u[i, t] - (u[i, t - 1] if t > 0 else self.Ui0[i]) for i in range(self.NumThermal) for t in range(self.HorizonLen)), name="state_variable")

            #Unit generation capacity limits constrains	3
            m.addConstrs((u[i,t]*self.ThPimin[i] <= p[i,t]  for i in range(self.NumThermal) for t in range (self.HorizonLen)), name="Unit_generation_limits1" )
            m.addConstrs((p[i,t] <= u[i,t]*self.ThPimax[i]  for i in range(self.NumThermal) for t in range (self.HorizonLen)), name="Unit_generation_limits2" )

            # Power balance constrains 4
            m.addConstrs((gp.quicksum(p[i, t] for i in range(self.NumThermal)) == self.Dt[t] for t in range(self.HorizonLen)), name="Power_balance_constrains")

            # System spinning reserve requirement 5
            m.addConstrs((gp.quicksum(u[i, t] * -self.ThPimax[i] for i in range(self.NumThermal)) <= -self.Dt[t] - self.Spin[t] for t in range(self.HorizonLen)), name="System_spinning_reserve_requirement")

            # Ramp rate limits 17 18
            # m.addConstrs((p[i,t]-(p[i,t-1] if t > 0 else self.Pi0[i])<=u[i,t]*(self.Piup[i]+self.ThPimin[i])-(u[i,t-1] if t > 0 else self.Ui0[i]) * self.ThPimin[i] + s[i,t] * (self.Pistartup[i] - self.Piup[i] - self.ThPimin[i]) for i in range(self.NumThermal) for t in range(self.HorizonLen)), name="Ramp_rate_limits1")
            # m.addConstrs(((p[i, t - 1] if t > 0 else self.Pi0[i]) - p[i, t] <= (u[i, t - 1] if t > 0 else self.Ui0[i])* (self.Pidown[i] + self.ThPimin[i]) - u[i, t] * self.ThPimin[i] + d[i, t] * (self.Pishutdown[i] - self.Pidown[i] - self.ThPimin[i]) for i in range(self.NumThermal)for t in range(self.HorizonLen)), name="Ramp_rate_limits2")
            #from 1bin        
            m.addConstrs((p[i,t]-(p[i,t-1]if t>0 else self.Pi0[i]) <= (u[i,t-1] if t>0 else self.Ui0[i])*self.Piup[i] + (u[i,t]-(u[i,t-1] if t>0 else self.Ui0[i]))*self.Pistartup[i] +(1-u[i,t])*self.ThPimax[i]  for i in range(self.NumThermal) for t in range(self.HorizonLen)),name="Ramp_rate_limits1" )
            m.addConstrs(((p[i,t-1]if t>0 else self.Pi0[i])-p[i,t] <= u[i,t]*self.Pidown[i] + ((u[i,t-1] if t>0 else self.Ui0[i])-u[i,t])*self.Pishutdown[i] +(1-(u[i,t-1] if t>0 else self.Ui0[i]))*self.ThPimax[i]  for i in range(self.NumThermal) for t in range(self.HorizonLen)),name="Ramp_rate_limits2" )


            # Minimum up/down time constraints 13 14
            m.addConstrs((gp.quicksum(s[i, w] for w in range(max(0, t + 1 - self.ThTime_on_min[i]), t + 1)) <= u[i, t] for i in range(self.NumThermal) for t in range(Ui[i], self.HorizonLen)), name="Minimum_up/down_time_constraints1")
            m.addConstrs((gp.quicksum(d[i, w] for w in range(max(0, t + 1 - self.ThTime_off_min[i]), t + 1)) <= 1 - u[i, t] for i in range(self.NumThermal) for t in range(Li[i], self.HorizonLen)), name="Minimum_up/down_time_constraints2")

            # Initial status of units 10
            m.addConstrs((u[i, t] == self.Ui0[i] for i in range(self.NumThermal) for t in range(Ui[i] + Li[i])),name="Initial_status_of_units")

            # startup cost 19 20
            m.addConstrs((-sc[i, t] <= -self.hoti[i] * s[i, t] for i in range(self.NumThermal) for t in range(self.HorizonLen)),name="startup_cost1")
            m.addConstrs((-sc[i,t] <= -self.coldi[i]*(s[i,t]-gp.quicksum(d[i,w] for w in range(max(0,t-self.ThTime_off_min[i]-self.Tcoldi[i]),t)) - (1 if t+1-self.ThTime_off_min[i]-self.Tcoldi[i] <= 0 and max(0,-self.ThTime_on_off_init[i])<abs(t-self.ThTime_off_min[i]-self.Tcoldi[i])+1  else 0)) for i in range(self.NumThermal) for t in range (self.HorizonLen)),name="startup_cost2")
            #from 1bin
            # m.addConstrs((-sc[i,t] <= -1*(self.hoti[i] if l<=self.ThTime_off_min[i]+self.Tcoldi[i] else self.coldi[i])* (u[i,t]-gp.quicksum((u[i,t-j]) if t>=j else(1 if j-t<=self.ThTime_on_off_init[i] or (self.ThTime_on_off_init[i]<0 and j-t>-self.ThTime_on_off_init[i]) else 0) for j in range(1,l+1))) for i in range(self.NumThermal) for t in range(self.HorizonLen) for l in [1,Ndi[i]]),name="startup_cost")

            m.setObjective(gp.quicksum(self.alpha[i] * u[i, t] + self.beta[i] * p[i, t] + sc[i, t] for i in range(self.NumThermal) for t in range(self.HorizonLen)), GRB.MINIMIZE)
            # m.setObjective(gp.quicksum(self.alpha[i]*u[i,t] + self.beta[i]*p[i,t] + self.gamma[i]*p[i,t]*p[i,t] + sc[i,t] for i in range(self.NumThermal) for t in range(self.HorizonLen)),GRB.MINIMIZE)

            m.update()

            return m
    def get_1bin_model(self, data):
        self.Dt = data['Dt']
        self.Spin = data['Spin']
        self.Ui0 = data['u0']
        self.Pi0 = data['p0']
        self.ThTime_on_off_init = data['on_off']

        Ui = np.maximum(0,np.minimum(np.ones((self.NumThermal)) * self.HorizonLen,self.Ui0 * (self.ThTime_on_min - self.ThTime_on_off_init))).astype(int)                    #--N*1矩阵 
        Li = np.maximum(0,np.minimum(np.ones((self.NumThermal)) * self.HorizonLen,(np.ones((self.NumThermal)) - self.Ui0) * (self.ThTime_off_min + self.ThTime_on_off_init))).astype(int)    #--N*1矩阵
      

        Ndi = [self.ThTime_off_min[i] + self.Tcoldi[i] + 1 for i in range(self.NumThermal)]
        # m = gp.Model("1-bin UC formulation")
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            m = gp.Model(env=env)
        
            u = m.addVars(self.NumThermal, self.HorizonLen, vtype=GRB.BINARY, name="u")  # N行T列
            p = m.addVars(self.NumThermal, self.HorizonLen, lb = [0 for i in range(self.NumThermal)for t in range(self.HorizonLen)], ub=[self.ThPimax[i] for i in range(self.NumThermal)for t in range(self.HorizonLen)], vtype=GRB.CONTINUOUS, name="p")  # N行T列
            sc = m.addVars(self.NumThermal, self.HorizonLen, lb=0 , ub=[self.coldi[i] for i in range(self.NumThermal)for t in range(self.HorizonLen)], vtype=GRB.CONTINUOUS, name="sc")  # N行T列
           
            #Minimum up/down time constraints
            m.addConstrs((u[i,t]-(u[i,t-1] if t>0 else self.Ui0[i])<=u[i,l] for t in range(self.HorizonLen) for i in range(self.NumThermal) for l in range(t+1,min(t+self.ThTime_on_min[i],self.HorizonLen))),name="Minimum_up/down_time_constraints1")
            m.addConstrs(((u[i,t-1] if t>0 else self.Ui0[i])-u[i,t]<=1-u[i,l] for t in range(self.HorizonLen) for i in range(self.NumThermal) for l in range(t+1,min(t+self.ThTime_off_min[i],self.HorizonLen))),name="Minimum_up/down_time_constraints2")
         
            #Unit generation capacity limits constrains	
            m.addConstrs((u[i,t]*self.ThPimin[i] <= p[i,t]  for i in range(self.NumThermal) for t in range (self.HorizonLen)), name="Unit_generation_limits1" )
            m.addConstrs((p[i,t] <= u[i,t]*self.ThPimax[i]  for i in range(self.NumThermal) for t in range (self.HorizonLen)), name="Unit_generation_limits2" )

            #Power balance constrains
            m.addConstrs((gp.quicksum(p[i,t] for i in range(self.NumThermal)) == self.Dt[t] for t in range (self.HorizonLen)),name="Power_balance_constrains" )

            #System spinning reserve requirement
            m.addConstrs((-gp.quicksum(u[i,t]*self.ThPimax[i] for i in range(self.NumThermal)) <= -self.Dt[t]-self.Spin[t] for t in range (self.HorizonLen)),name="System_spinning_reserve_requirement" )

            #Ramp rate limits
            m.addConstrs((p[i,t]-(p[i,t-1]if t>0 else self.Pi0[i]) <= (u[i,t-1] if t>0 else self.Ui0[i])*self.Piup[i] + (u[i,t]-(u[i,t-1] if t>0 else self.Ui0[i]))*self.Pistartup[i] +(1-u[i,t])*self.ThPimax[i]  for i in range(self.NumThermal) for t in range(self.HorizonLen)),name="Ramp_rate_limits1" )
            m.addConstrs(((p[i,t-1]if t>0 else self.Pi0[i])-p[i,t] <= u[i,t]*self.Pidown[i] + ((u[i,t-1] if t>0 else self.Ui0[i])-u[i,t])*self.Pishutdown[i] +(1-(u[i,t-1] if t>0 else self.Ui0[i]))*self.ThPimax[i]  for i in range(self.NumThermal) for t in range(self.HorizonLen)),name="Ramp_rate_limits2" )

            #Initial status of units
            m.addConstrs((u[i,t]==self.Ui0[i] for i in range(self.NumThermal) for t in range(Ui[i]+Li[i])),name="Initial_status_of_units")

            #startup cost
            m.addConstrs((-sc[i,t] <= -1*(self.hoti[i] if l<=self.ThTime_off_min[i]+self.Tcoldi[i] else self.coldi[i])* (u[i,t]-gp.quicksum((u[i,t-j]) if t>=j else(1 if j-t<=self.ThTime_on_off_init[i] or (self.ThTime_on_off_init[i]<0 and j-t>-self.ThTime_on_off_init[i]) else 0) for j in range(1,l+1))) for i in range(self.NumThermal) for t in range(self.HorizonLen) for l in [1,Ndi[i]]),name="startup_cost")
            # m.addConstrs(( sc[i,t] >= (self.hoti[i] if l<=self.ThTime_off_min[i]+self.Tcoldi[i] else self.coldi[i])* (u[i,t]-gp.quicksum((u[i,t-j]) if t>=j else(1 if j-t<=self.ThTime_on_off_init[i] or (self.ThTime_on_off_init[i]<0 and j-t>-self.ThTime_on_off_init[i]) else 0) for j in range(1,l+1))) for i in range(self.NumThermal) for t in range(self.HorizonLen) for l in [1,Ndi[i]]),name="startup_cost")

            # m.setObjective(gp.quicksum(self.beta[i]*p[i,t] + self.gamma[i]*p[i,t]*p[i,t] for i in range(self.NumThermal) for t in range(self.HorizonLen)),GRB.MINIMIZE)
            m.setObjective(gp.quicksum(self.alpha[i]*u[i,t] + self.beta[i]*p[i,t] + sc[i,t] for i in range(self.NumThermal) for t in range(self.HorizonLen)),GRB.MINIMIZE)
            # m.setObjective(gp.quicksum(self.alpha[i]*u[i,t] + self.beta[i]*p[i,t] + self.gamma[i]*p[i,t]*p[i,t] + sc[i,t] for i in range(self.NumThermal) for t in range(self.HorizonLen)),GRB.MINIMIZE)

            m.update()
            return m
