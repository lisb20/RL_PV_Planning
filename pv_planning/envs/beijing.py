"""Historical Beijing planning environment with bundled data."""
import numpy as np
import json
from pathlib import Path
import os
from .flow import Flow
import yaml
import torch
from .utils import facilityid_2_graphid


def sin_position_embding(x, dim=32):
    return torch.sin(x / 10 ** (torch.arange(0, dim // 2) / (dim // 2)))


def calculate_grid_cost(P_gen):
    max_gen = np.max(P_gen, axis=1)
    large_index = np.where(max_gen > 500)[0]
    mid_index = np.where((max_gen <= 500) & (max_gen > 300))[0]
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
        num_agents=803,
        max_step=10,
        lambda1=0,
        lambda2=0,
        flow_config=None,
        input_fp=None,
        power_ref=None,
        solar_ref=None,
        trans_2_area=None,
    ):
        data_dir = Path(__file__).resolve().parents[1] / "data" / "beijing"
        if flow_config is None:
            with (data_dir / "config.yaml").open() as f:
                flow_config = yaml.safe_load(f)
        if input_fp is None:
            import gzip
            with gzip.open(data_dir / "grid.json.gz", "rt") as f:
                input_fp = json.load(f)
        self.flow = Flow(config=flow_config, input_fp=input_fp)

        self.elec_json = self.flow.elec_geojson
        ## power demand: N*96
        if power_ref is None:
            self.power_demand_ref = np.load(
                data_dir / "power_map0221.npy"
            )
        else:
            self.power_demand_ref = power_ref
        self.mean, self.std = np.mean(self.power_demand_ref), np.std(
            self.power_demand_ref
        )

        if solar_ref is None:
            self.solar_template = np.load(
                data_dir.parent / "solarbej.npy"
            )
        else:
            self.solar_template = solar_ref

        self.solar_template = self.solar_template / np.sum(self.solar_template)
        # interpolate 24 - > 96
        self.solar_template = np.interp(
            np.arange(0, 96), np.arange(0, 24) * 4, self.solar_template
        )
        self.solar_template = self.solar_template / np.sum(self.solar_template)
        self.solar_template_ref = self.solar_template.reshape(1, 96)

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        if trans_2_area is None:
            with (data_dir / "roof_areas.json").open() as f:
                self.trans110_2_area = {int(k): v for k, v in json.load(f).items()}
        else:
            self.trans110_2_area = trans_2_area
        # self.trans110_2_area = {int(k - 4e8): v for k, v in self.trans110_2_area.items()}
        # self.area_vec = sorted
        self.N_agent = num_agents
        self.solar_percentage = np.ones((self.N_agent,)) * 0.5
        self.max_step = max_step
        self.area_vec = np.zeros((self.N_agent,))
        for k, v in self.trans110_2_area.items():
            idx = int(k - 4e8)
            self.area_vec[idx] = v

    def get_graph(self):
        edge_index = []
        for f in self.elec_json:
            if "relation" in f["properties"]:
                src, dst = (
                    f["properties"]["relation"][0],
                    f["properties"]["relation"][1],
                )
                if src // 1e8 < 5 and dst // 1e8 < 5:
                    src = facilityid_2_graphid(src)
                    dst = facilityid_2_graphid(dst)
                    edge_index.append((src, dst))
                    edge_index.append((dst, src))

        edge_index = list(set(edge_index))
        edge_index = np.array(edge_index).T.astype(np.int64)
        return edge_index

    def reset(self):
        # reset the environment
        self.flow.reset()
        self.power_demand = self.power_demand_ref.copy()
        self.solar_template = self.solar_template_ref.copy()

        self.solar = np.zeros((self.N_agent,))
        # self.solar_percentage = np.ones((self.N_agent, )) * 0.5
        # max_percentage = np.ones((self.N_agent, )) * 0.5
        # min_percentage = -1 * np.ones((self.N_agent, )) * 0.5
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
        solar = 4 / 1e3 * action_mat @ self.solar_template
        total_solar = np.sum(solar)
        power_demand_now = self.power_demand - solar
        # power_negative = np.sum(power_demand_now[power_demand_now < 0])
        # power_demand_now = np.clip(power_demand_now, 0, None)

        gen_power = np.zeros((25, 96)) + 1
        # t0 = time.time()
        res = []
        for t in range(96):
            # 运行flow
            result = self.flow.run_fast(power_demand_now[:, t])
            if result is None:
                raise RuntimeError(f"AC power flow failed at quarter-hour {t}")
            res.append(result)
            # 计算发电量
            gen_power[:, t] = res[-1]["gen"][:, 1]

        power_negative = 0
        for i in range(gen_power.shape[0]):
            row = gen_power[i].copy()
            maxgen = np.max(row)
            min_lim = maxgen * 0.4
            valley = row - min_lim
            power_negative += np.sum(valley[valley < 0])

        # 计算ramping
        ramping = gen_power[:, 4:] - gen_power[:, :-4]  # N_gen, 95
        ramping = np.abs(ramping)
        ramping_rate = ramping / np.max(gen_power, axis=1).reshape(-1, 1)  # N_gen, 95
        max_ramping = np.max(ramping_rate, axis=1)
        # 计算reward
        ramping_cost = calculate_grid_cost(gen_power) / 5e4 - 4
        self.last_reward = (
            total_solar / 5e4
            - (ramping_cost * self.lambda2)
        )

        state = power_demand_now
        state = (state - self.mean) / self.std
        solar_reset = (solar - self.mean) / self.std
        # state = np.log10(state + 1)
        # solar_reset = np.log10(solar + 1)
        # state = np.concatenate([state, solar_reset], axis=1)
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

        # 将action限制在0和1之间，并去掉维度
        action = np.clip(action, -1, 1).squeeze()
        self.solar_percentage = self.solar_percentage + action
        # assert np.all(self.solar_percentage >= 0) and np.all(self.solar_percentage <= 1)
        bad_percentage = np.sum(
            self.solar_percentage[self.solar_percentage > 1] - 1
        ) - np.sum(self.solar_percentage[self.solar_percentage < 0])

        area = self.area_vec.copy()
        area = area / np.max(area)

        self.solar_percentage = np.clip(self.solar_percentage, 0, 1)
        max_percentage = 1 - self.solar_percentage
        min_percentage = -1 * self.solar_percentage

        self.step_num += 1
        step_emb = sin_position_embding(self.step_num, 32)
        # print("step_emb", step_emb)
        step_emb = torch.stack([step_emb] * self.N_agent, dim=0)
        solar = np.zeros((self.N_agent, 96))

        # 将action乘以area_vec，并重新调整维度
        action_mat = (self.solar_percentage * self.area_vec).reshape(self.N_agent, 1)
        # 计算solar
        solar = 4 / 1e3 * action_mat @ self.solar_template
        # 计算总太阳能
        total_solar = np.sum(solar)

        # 计算当前时刻的电力需求
        power_demand_now = self.power_demand - solar

        # 计算电力负值
        # power_negative = np.sum(power_demand_now[power_demand_now < 0])
        # power_demand_now = np.clip(power_demand_now, 0, None)

        # total_solar += power_negative * 1  # 惩罚过多的太阳能发电
        next_state = power_demand_now

        next_state = (next_state - self.mean) / self.std
        solar_now = (solar - self.mean) / self.std
        # next_state = np.log10(next_state + 1)
        # solar_now = np.log10(solar + 1)

        # 将next_state和solar_now拼接
        # next_state = np.concatenate([next_state, solar_now], axis=1)
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

        # 判断是否结束
        done = False if self.step_num < self.max_step else True
        # done = False
        # return next_state, fake_reward, done, None
        res = []
        gen_power = np.zeros((25, 96)) + 1
        # t0 = time.time()
        for t in range(96):
            # 运行flow
            result = self.flow.run_fast(power_demand_now[:, t])
            if result is None:
                raise RuntimeError(f"AC power flow failed at quarter-hour {t}")
            res.append(result)
            # 计算发电量
            gen_power[:, t] = res[-1]["gen"][:, 1]

        # 计算ramping
        ramping = gen_power[:, 4:] - gen_power[:, :-4]  # N_gen, 95
        ramping_cost = calculate_grid_cost(gen_power) / 5e4 - 4

        ramping = np.abs(ramping)
        # 计算ramping_rate
        ramping_rate = ramping / np.max(gen_power, axis=1).reshape(-1, 1)  # N_gen, 95
        # print(ramping_rate[0])
        # 计算最大ramping_rate
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
            total_solar / 5e4
            - (ramping_cost * self.lambda2)
            - self.lambda1 * bad_percentage * 0.05
        )
        info = [
            np.sum(solar),
            np.sum(power_negative),
            np.mean(max_ramping),
            np.sum(ramping_cost),
            bad_percentage,
        ]

        ret_now = reward - self.last_reward
        self.last_reward = reward

        return next_state, ret_now, done, info
