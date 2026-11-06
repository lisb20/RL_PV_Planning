"""Historical IEEE-300 planning environment with portable data paths."""
import numpy as np
from pathlib import Path
from pypower.api import ppoption, runpf, loadcase
from scipy.io import loadmat
import torch
import math


def sin_position_embding(x, dim=32):
    return torch.sin(x / 10 ** (torch.arange(0, dim // 2) / (dim // 2)))


def calculate_grid_cost(P_gen):
    max_gen = np.max(P_gen, axis=1)
    large_index = np.where(max_gen > 750)[0]
    mid_index = np.where((max_gen <= 750) & (max_gen > 300))[0]
    small_index = np.where(max_gen <= 300)[0]

    ramping = P_gen[:, 4:] - P_gen[:, :-4]
    cost = 0
    for i in large_index:
        row = np.abs(ramping[i])
        cost += (
            np.sum(row[row < max_gen[i] * 0.30]) * 1.96
            + np.sum(row[row >= max_gen[i] * 0.30]) * 1.96 * 6
        )
    for i in mid_index:
        row = np.abs(ramping[i])
        cost += (
            np.sum(row[row < max_gen[i] * 0.35]) * 2.45
            + np.sum(row[row >= max_gen[i] * 0.35]) * 2.45 * 6
        )
    for i in small_index:
        row = np.abs(ramping[i])
        cost += (
            np.sum(row[row < max_gen[i] * 0.32]) * 3.34
            + np.sum(row[row >= max_gen[i] * 0.32]) * 3.34 * 6
        )
    return cost


class myenv:
    def __init__(
        self,
        num_agents=300,
        max_step=10,
        lambda1=0,
        lambda2=0,
        case=None,
        data_dir=None,
    ):
        data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "data"
        case = str(case or data_dir / "case300.mat")
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.N_agent = num_agents

        self.solar_template = np.load(data_dir / "solarbej.npy")

        self.solar_template = self.solar_template / np.sum(self.solar_template)
        # interpolate 24 - > 96
        self.solar_template = np.interp(
            np.arange(0, 96), np.arange(0, 24) * 4, self.solar_template
        )
        self.solar_template = self.solar_template / np.sum(self.solar_template)

        self.case = loadcase(case)
        self.P = np.zeros((num_agents, 96))
        self.Q = np.zeros((num_agents, 96))
        tmp = loadmat(data_dir / "power_2class.mat")
        self.template_resi = tmp["power_resi"].reshape(-1)
        self.template_resi = self.template_resi / np.mean(self.template_resi)
        self.template_resi = np.interp(
            np.arange(0, 48, 0.5), np.arange(0, 48), self.template_resi
        )
        self.template_other = tmp["power_indu"].reshape(-1)
        self.template_other = self.template_other / np.mean(self.template_other)
        self.template_other = np.interp(
            np.arange(0, 48, 0.5), np.arange(0, 48), self.template_other
        )
        if "case39" in case:
            resi = [x[0] for x in self.case["bus"] if x[6] == 1]
            for i in range(1, 40):
                if i in resi:
                    self.P[i - 1] = self.template_resi * self.case["bus"][i - 1, 2]
                    self.Q[i - 1] = self.template_resi * self.case["bus"][i - 1, 3]
                else:
                    self.P[i - 1] = self.template_other * self.case["bus"][i - 1, 2]
                    self.Q[i - 1] = self.template_other * self.case["bus"][i - 1, 3]
            self.case["bus"][29:40, 1] = 3
            self.case["gen"][:, 8] = 9999
            self.case["gen"][:, 9] = -9999
            self.case["gen"][:, 3] = 9999
            self.case["gen"][:, 4] = -9999
            self.N_gen = 10
        elif "case300" in case:
            none_zero_p = 0
            for i in range(1, 301):
                if i < 150:
                    self.P[i - 1] = self.template_resi * self.case["bus"][i - 1, 2]
                    self.Q[i - 1] = self.template_resi * self.case["bus"][i - 1, 3]
                else:
                    self.P[i - 1] = self.template_other * self.case["bus"][i - 1, 2]
                    self.Q[i - 1] = self.template_other * self.case["bus"][i - 1, 3]
                if self.P[i - 1, 48] != 0:
                    none_zero_p += 1

            self.case["gen"][:, 8] = 9999
            self.case["gen"][:, 9] = -9999
            self.case["gen"][:, 3] = 9999
            self.case["gen"][:, 4] = -9999
            self.N_gen = self.case["gen"].shape[0]
            gen_bus_id = self.case["gen"][:, 0].astype(np.int64)
            gen_bus_id = set(gen_bus_id)
            for i in range(300):
                if self.case["bus"][i, 0] in gen_bus_id:
                    self.case["bus"][i, 1] = 3

            self.BASE = 5e4
        else:
            raise NotImplementedError

        self.area_vec = self.P[:, 48] * 2 / 0.15 * 1e3  ## maximun 0.15kw/m^2
        tot_area = np.sum(self.area_vec)
        idx_high = [i for i in range(self.N_agent) if self.case["bus"][i, 9] > 110]
        idx_low = [i for i in range(self.N_agent) if self.case["bus"][i, 9] <= 110]
        p_high = np.sum(self.P[idx_high, :])
        p_low = np.sum(self.P[idx_low, :])
        self.area_vec[idx_high] = tot_area * 0.9 / len(idx_high)
        self.area_vec[idx_low] = tot_area * 0.1 / len(idx_low)

        self.solar_percentage = np.random.uniform(0, 1, (self.N_agent,))
        self.max_step = max_step

        self.mean = np.mean(self.P)
        self.std = np.std(self.P)

    def get_graph(self):
        edge_mapping = {}
        for i,row in enumerate(self.case["bus"]):
            edge_mapping[row[0]] = i

        edge_index = []
        for x in self.case["branch"]:
            edge_index.append([edge_mapping[x[0]], edge_mapping[x[1]]])
            edge_index.append([edge_mapping[x[1]], edge_mapping[x[0]]])
        edge_index = np.array(edge_index).T.astype(np.int64)
        # edge_index -= 1
        return edge_index

    def reset(self):
        self.solar = np.zeros((self.N_agent,))
        self.solar_percentage = np.random.uniform(0, 1, (self.N_agent,))
        max_percentage = 1 - self.solar_percentage
        min_percentage = -1 * self.solar_percentage

        area = self.area_vec.copy()
        area = area / np.max(area)

        self.step_num = 0
        step_emb = sin_position_embding(self.step_num, 32)
        step_emb = torch.stack([step_emb] * self.N_agent, dim=0)

        solar = np.zeros((self.N_agent, 96))
        action_mat = (self.solar_percentage * self.area_vec).reshape(self.N_agent, 1)

        solar = 4 / 1e3 * action_mat @ self.solar_template.reshape(1, 96)
        solar_Q = solar * math.sqrt((1 / 0.99) ** 2 - 1)

        total_solar = np.sum(solar)
        power_demand_now = self.P - solar
        power_demand_now = np.clip(power_demand_now, 0, None)

        Q_now = self.Q - solar_Q
        Q_now = np.minimum(Q_now, np.abs(self.Q))
        Q_now = np.maximum(Q_now,  -1 * np.abs(self.Q))

        gen_power = np.zeros((self.N_gen, 96)) + 1
        res = []

        for t in range(96):
            P_t = power_demand_now[:, t]
            Q_t = Q_now[:, t]
            self.case["bus"][:, 2] = P_t
            self.case["bus"][:, 3] = Q_t
            ppc = {
                "version": 2,
                "baseMVA": 100,
                "bus": self.case["bus"],
                "branch": self.case["branch"],
                "gen": self.case["gen"],
            }
            ppopt = ppoption(OUT_ALL=0, VERBOSE=0, PF_ALG=1)
            results, succ = runpf(ppc, ppopt)
            res.append(results)
            if not succ:
                raise Exception("Flow error")
            gen_power[:, t] = res[-1]["gen"][:, 1]

        # Ramping
        ramping = gen_power[:, 4:] - gen_power[:, :-4]  # N_gen, 95
        ramping = np.abs(ramping)
        ramping_rate = ramping / np.max(gen_power, axis=1).reshape(-1, 1)  # N_gen, 95
        max_ramping = np.max(ramping_rate, axis=1)
        ramping_cost = calculate_grid_cost(gen_power) / self.BASE

        power_negative = 0
        for i in range(gen_power.shape[0]):
            row = gen_power[i].copy()
            maxgen = np.max(row)
            min_lim = maxgen * 0.4
            valley = row - min_lim
            power_negative += np.sum(valley[valley < 0])

        # Reward
        self.last_reward = (
            total_solar / self.BASE
            - (ramping_cost * self.lambda2)
        )

        state = power_demand_now
        state = (state - self.mean) / self.std
        solar_reset = (solar - self.mean) / self.std

        state = np.concatenate(
            [
                state,
                solar_reset,
                self.solar_percentage.reshape(-1, 1),
                max_percentage.reshape(-1, 1),
                min_percentage.reshape(-1, 1),
                area.reshape(-1, 1),
                step_emb,
            ],
            axis=1,
        )

        return state

    def step(self, action):
        # action: [N]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        action = np.clip(action, -1, 1).squeeze()
        self.solar_percentage = self.solar_percentage + action

        bad_percentage = np.sum(self.solar_percentage[self.solar_percentage > 1] - 1) - np.sum(self.solar_percentage[self.solar_percentage < 0])


        self.solar_percentage = np.clip(self.solar_percentage, 0, 1)
        max_percentage = 1 - self.solar_percentage
        min_percentage = -1 * self.solar_percentage

        area = self.area_vec.copy()
        area = area / np.max(area)

        self.step_num += 1

        step_emb = sin_position_embding(self.step_num, 32)
        step_emb = torch.stack([step_emb] * self.N_agent, dim=0)
        solar = np.zeros((self.N_agent, 96))

        action_mat = (self.solar_percentage * self.area_vec).reshape(self.N_agent, 1)
        solar = 4 / 1e3 * action_mat @ self.solar_template.reshape(1, 96)
        solar_Q = solar * math.sqrt((1 / 0.99) ** 2 - 1)
        total_solar = np.sum(solar)

        power_demand_now = self.P - solar
        power_demand_now = np.clip(power_demand_now, 0, None)

        Q_now = self.Q - solar_Q
        Q_now = np.minimum(Q_now, np.abs(self.Q))
        Q_now = np.maximum(Q_now,  -1 * np.abs(self.Q))

        next_state = power_demand_now
        next_state = (next_state - self.mean) / self.std
        solar_now = (solar - self.mean) / self.std

        next_state = np.concatenate(
            [
                next_state,
                solar_now,
                self.solar_percentage.reshape(-1, 1),
                max_percentage.reshape(-1, 1),
                min_percentage.reshape(-1, 1),
                area.reshape(-1, 1),
                step_emb,
            ],
            axis=1,
        )

        done = False if self.step_num < self.max_step else True

        res = []
        gen_power = np.zeros((self.N_gen, 96)) + 1
        # t0 = time.time()
        for t in range(96):
            # 运行flow
            P_t = power_demand_now[:,t]
            Q_t = Q_now[:,t]
            self.case["bus"][:, 2] = P_t
            self.case["bus"][:, 3] = Q_t
            # self.case["bus"][:, 3] = 0
            ppc = {
                "version": 2,
                "baseMVA": 100,
                "bus": self.case["bus"],
                "branch": self.case["branch"],
                "gen": self.case["gen"],
            }
            ppopt = ppoption(OUT_ALL=0, VERBOSE=0, PF_ALG=1)
            results, succ = runpf(ppc, ppopt)
            res.append(results)
            if not succ:
                raise Exception("Flow error")
            gen_power[:, t] = res[-1]["gen"][:, 1]

        ramping = gen_power[:, 4:] - gen_power[:, :-4]  # N_gen, 95
        ramping_cost = calculate_grid_cost(gen_power) / self.BASE

        ramping = np.abs(ramping)
        # ramping_rate
        ramping_rate = ramping / np.max(gen_power, axis=1).reshape(-1, 1)  # N_gen, 95
        max_ramping = np.max(ramping_rate, axis=1)

        power_negative = 0
        for i in range(gen_power.shape[0]):
            row = gen_power[i].copy()
            maxgen = np.max(row)
            min_lim = maxgen * 0.4
            valley = row - min_lim
            power_negative += np.sum(valley[valley < 0])

        # 计算reward
        reward = (
            total_solar / self.BASE
            - (ramping_cost * self.lambda2)
            - bad_percentage * 5 * self.lambda1
        )
        info = [
            np.sum(solar),
            np.sum(power_negative),
            np.mean(max_ramping),
            np.sum(ramping_cost),
            bad_percentage,
            self.solar_percentage,
        ]

        ret_now = reward - self.last_reward
        self.last_reward = reward

        return next_state, ret_now, done, info
