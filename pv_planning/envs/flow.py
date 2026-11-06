"""AC power-flow construction for the synthetic Beijing network."""
import numpy as np
import json
import math
import yaml
import os
import pickle
import pyproj
from collections import defaultdict
from shapely.geometry import LineString
from pypower.api import ppoption, runpf
from . import facility_types as input_pb
import copy
np.set_printoptions(suppress=True)
# projector = pyproj.Proj(
#     "+proj=utm +zone=50 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
# )
# projector at beijing
projector = pyproj.Proj(
    "+lon_0=116.391 +lat_0=39.904 +proj=utm +zone=50 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
)
BASE = 100000000
UPPER_220 = 1.4
UPPER_POWER = 1.5


class Flow(object):
    def __init__(self, config, input_fp=""):
        self.basic_config = config["basic_config"]
        self.index_config = config["index_config"]
        if isinstance(input_fp, str):
            with open(input_fp) as f:
                input_data = json.load(f)
                self.input_data = input_data
        else:
            self.input_data = input_fp

        self.projector = pyproj.Proj(
            "+proj=utm +zone=50 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        )
        self.elec_geojson = self.get_elec_json()
        self.init_config()
        self.init_relation()


    def get_elec_json(self):
        """
        从数据库中获取电力需求数据,以及电网拓扑的geojson数据用来计算电阻
        """
        require_list = []
        elec_geojson = []
        for data in self.input_data:
            if data["type"] == "requirement":
                require_list.append(data["data"])
            else:
                elec_geojson.append(data)

        return elec_geojson

    def get_edge_and_node_dict(self):
        """
        从geojson中获取边的信息,用来构建母线bus
        并生成节点信息字典
        """
        edge_dict = {}
        bus_dict = {}
        node_dict = {
            input_pb.FACILITY_TYPE_POWER_STATION: {},
            input_pb.FACILITY_TYPE_TRANSFORMER_500: {},
            input_pb.FACILITY_TYPE_TRANSFORMER_220: {},
            input_pb.FACILITY_TYPE_TRANSFORMER_110: {},
        }
        for geo_data in self.elec_geojson:
            if geo_data["geometry"]["type"] == "LineString":
                relation = tuple(geo_data["properties"]["relation"])
                # 将经纬度转换到xy坐标
                # if input_pb   in   np.array(relation)//BASE
                if input_pb.FACILITY_TYPE_TRANSFORMER_10 in np.array(relation) // BASE:
                    # 去掉10kv的边连接
                    continue
                # line = geo_data["geometry"]["coordinates"]
                line = geo_data["geometry"]["coordinates"]
                proj_line = []
                for l in line:
                    proj_line.append(projector(l[0], l[1]))

                edge_dict[relation] = LineString(proj_line)
                assert edge_dict[relation].is_valid

            else:
                node_id = geo_data["properties"]["id"]
                node_type = node_id // BASE
                if node_type in node_dict:
                    node_dict[node_type][node_id] = {}
                    bus_dict[node_id] = {}
        return edge_dict, node_dict, bus_dict

    def init_relation(self):
        """
        读取电网拓扑关系，构建110kv-aoi的映射
        """
        branchs = [x for x in self.elec_geojson if x["geometry"]["type"] == "LineString"]
        trans110_2_trans10 = defaultdict(list)
        trans10_2_aoi = defaultdict(list)
        for b in branchs:
            if b["properties"]["relation"][0] // BASE == input_pb.FACILITY_TYPE_TRANSFORMER_110 \
                and b["properties"]["relation"][1] // BASE == input_pb.FACILITY_TYPE_TRANSFORMER_10:
                trans110_2_trans10[b["properties"]["relation"][0]].append(b["properties"]["relation"][1])

            if b["properties"]["relation"][0] // BASE == input_pb.FACILITY_TYPE_TRANSFORMER_10 \
                and b["properties"]["relation"][1] // BASE == input_pb.FACILITY_TYPE_AOI:
                trans10_2_aoi[b["properties"]["relation"][0]].append(b["properties"]["relation"][1])

        self.trans110_2_aoi = defaultdict(list)
        for k, v in trans110_2_trans10.items():
            for trans10 in v:
                v0 = trans10_2_aoi[trans10]
                v0 = [(5e8 + x // BASE) for x in v0]
                self.trans110_2_aoi[k].extend(v0)

    def init_config(self):
        self.edge_dict, self.node_dict, self.bus_dict = self.get_edge_and_node_dict()
        self.power_num = len(self.node_dict[input_pb.FACILITY_TYPE_POWER_STATION])
        self.trans500_num = len(self.node_dict[input_pb.FACILITY_TYPE_TRANSFORMER_500])
        self.trans220_num = len(self.node_dict[input_pb.FACILITY_TYPE_TRANSFORMER_220])
        self.trans110_num = len(self.node_dict[input_pb.FACILITY_TYPE_TRANSFORMER_110])
        self.power_require = np.zeros(self.trans110_num)
        # print("-----------config------------")
        # print("Power_number",self.power_num,
        #       "trans500_number",self.trans500_num,
        #       "trans220_number",self.trans220_num,
        #       "trans110_number",self.trans110_num)

        self.bus_matrix, self.facility_list = self.generate_bus()
        self.branch_matrix = self.generate_branch()
        self.power_matrix = self.get_power_data()

    def update_power(self, power_demand):
        self.power_require = np.zeros(self.trans110_num)
        for k,v in self.trans110_2_aoi.items():
            if k // BASE == input_pb.FACILITY_TYPE_TRANSFORMER_110:
                k = k % BASE
            for aoi in v:
                if aoi in power_demand:
                    self.power_require[k] += power_demand[aoi]

        # print("power_require", self.power_require)


    def get_vms(self, vm):
        """
        辅助函数，计算 vm相关的参数
        """
        return vm, vm * 1.5, vm / 1.5

    def generate_bus(self):
        """
        获取总线数据
        """
        bus_matrix = []
        bus_num = (
            2 * (self.trans500_num + self.power_num + self.trans220_num)
            + self.trans110_num
        )
        power_ids = sorted(
            set(self.node_dict[input_pb.FACILITY_TYPE_POWER_STATION].keys()),
            reverse=False,
        )
        trans500_ids = sorted(
            set(self.node_dict[input_pb.FACILITY_TYPE_TRANSFORMER_500].keys()),
            reverse=False,
        )
        trans220_ids = sorted(
            set(self.node_dict[input_pb.FACILITY_TYPE_TRANSFORMER_220].keys()),
            reverse=False,
        )
        trans110_ids = sorted(
            set(self.node_dict[input_pb.FACILITY_TYPE_TRANSFORMER_110].keys()),
            reverse=False,
        )
        for bus_id in range(bus_num):
            if bus_id < self.power_num:  #
                type_id = 3
                Vm, Vmax, Vmin = self.get_vms(vm=5 / 1.1)
                Pd = 0
                Qd = 0
                n = bus_id - 0
                self.bus_dict[power_ids[n]]["in"] = bus_id

            elif bus_id < self.power_num + self.trans500_num:
                type_id = 3
                Vm, Vmax, Vmin = self.get_vms(vm=5 / 1.1)
                Pd = 0
                Qd = 0
                n = bus_id - self.power_num
                self.bus_dict[trans500_ids[n]]["in"] = bus_id

            elif bus_id < 2 * self.power_num + self.trans500_num:
                type_id = 1
                Pd = 0
                Qd = 0
                Vm, Vmax, Vmin = self.get_vms(vm=2)
                n = bus_id - (self.power_num + self.trans500_num)
                self.bus_dict[power_ids[n]]["out"] = bus_id

            elif bus_id < 2 * (self.power_num + self.trans500_num):
                type_id = 1
                Pd = 0
                Qd = 0
                Vm, Vmax, Vmin = self.get_vms(vm=2)
                n = bus_id - (2 * self.power_num + self.trans500_num)
                self.bus_dict[trans500_ids[n]]["out"] = bus_id

            elif bus_id < 2 * (self.power_num + self.trans500_num) + self.trans220_num:
                type_id = 1
                Pd = 0
                Qd = 0
                Vm, Vmax, Vmin = self.get_vms(vm=2)
                n = bus_id - 2 * (self.power_num + self.trans500_num)
                self.bus_dict[trans220_ids[n]]["in"] = bus_id
            elif bus_id < 2 * (self.power_num + self.trans500_num + self.trans220_num):
                type_id = 1
                Pd = 0
                Qd = 0
                Vm, Vmax, Vmin = self.get_vms(vm=1)
                n = (
                    bus_id
                    - 2 * (self.power_num + self.trans500_num)
                    - self.trans220_num
                )
                self.bus_dict[trans220_ids[n]]["out"] = bus_id
            else:
                type_id = 1
                n = bus_id - 2 * (
                    self.power_num + self.trans500_num + self.trans220_num
                )
                Pd = self.power_require[n]
                # Pd = 1e-6
                # Qd = 0
                Qd = Pd * math.sqrt((1 / self.basic_config["cos_phi"]) ** 2 - 1)
                Vm, Vmax, Vmin = self.get_vms(vm=1)
                self.bus_dict[trans110_ids[n]]["in"] = bus_id
            bus_matrix.append(
                [
                    bus_id,
                    type_id,
                    Pd,
                    Qd,
                    self.basic_config["Gs"],
                    self.basic_config["Bs"],
                    self.basic_config["area"],
                    Vm,
                    self.basic_config["Va"],
                    self.basic_config["baseKV"],
                    self.basic_config["Zone"],
                    Vmax,
                    Vmin,
                ]
            )
        # bus_matrix[0][1] = 3
        facility_ids_list = []
        for list_data in [power_ids, trans500_ids, trans220_ids]:
            facility_ids_list.extend(list_data)
        return np.array(bus_matrix), facility_ids_list

    def generate_branch(self):
        """
        branch
        """
        branch_matrix = []
        index = 0
        for facility_id in self.facility_list:
            r = 0
            x = 1e-6
            b = 0
            # x = 1e-3
            # b = 1e-6
            fbus = self.bus_dict[facility_id]["in"]

            tbus = self.bus_dict[facility_id]["out"]
            if facility_id // BASE == input_pb.FACILITY_TYPE_TRANSFORMER_220:
                ratio = 2
            else:
                ratio = 5 / 2.2
            self.node_dict[facility_id // BASE][facility_id]["index"] = index
            index += 1

            branch_matrix.append(
                [
                    fbus,
                    tbus,
                    r,
                    x,
                    b,
                    self.basic_config["rateA"],
                    self.basic_config["rateB"],
                    self.basic_config["rateC"],
                    ratio,
                    self.basic_config["angle"],
                    self.basic_config["status"],
                    self.basic_config["angmin"],
                    self.basic_config["angmax"],
                ]
            )
        cnt = 0

        for edge, info in self.edge_dict.items():
            node_1, node_2 = edge  # 两个设施节点构成一条边，默认node_1 < node_2
            assert node_1 < node_2
            if node_2 // BASE == input_pb.FACILITY_TYPE_TRANSFORMER_110:
                assert node_1 // BASE == input_pb.FACILITY_TYPE_TRANSFORMER_220
                fbus = self.bus_dict[node_1]["out"]
                tbus = self.bus_dict[node_2]["in"]
            elif node_2 // BASE == input_pb.FACILITY_TYPE_TRANSFORMER_220:
                cnt += 1
                if node_1 // BASE == input_pb.FACILITY_TYPE_TRANSFORMER_220:
                    fbus = self.bus_dict[node_1]["in"]
                    tbus = self.bus_dict[node_2]["in"]


                elif node_1 // BASE == input_pb.FACILITY_TYPE_POWER_STATION:
                    fbus = self.bus_dict[node_1]["out"]
                    tbus = self.bus_dict[node_2]["in"]
                elif node_1 // BASE == input_pb.FACILITY_TYPE_TRANSFORMER_500:
                    fbus = self.bus_dict[node_1]["out"]
                    tbus = self.bus_dict[node_2]["in"]
            else:
                assert node_1 // BASE != input_pb.FACILITY_TYPE_TRANSFORMER_220
                fbus = self.bus_dict[node_1]["in"]
                tbus = self.bus_dict[node_2]["in"]
            distance = info.length

            if distance < 1000:
                distance = 1000
            r = (0.033/181.5) * distance * 1e-3
            # r = 1e-4
            # x = 0
            # b = 0
            x = (0.03/181.5) * distance * 1e-3
            # b = 1e-6
            b = 1e-6 / (distance * 1e-3)
            # b = 1e-6 / (distance * 1e-3)

            ratio = 0
            branch_matrix.append(
                [
                    fbus,
                    tbus,
                    r,
                    x,
                    b,
                    self.basic_config["rateA"],
                    self.basic_config["rateB"],
                    self.basic_config["rateC"],
                    ratio,
                    self.basic_config["angle"],
                    self.basic_config["status"],
                    self.basic_config["angmin"],
                    self.basic_config["angmax"],
                ]
            )

        return np.array(branch_matrix)

    def reset(self):
        pass


    def run(self, power_demand):
        # self.update_power(power_demand)
        self.power_require = power_demand
        self.bus_matrix, self.facility_list = self.generate_bus()
        self.branch_matrix = self.generate_branch()
        self.power_matrix = self.get_power_data()
        ppc = {
            "version": 2,
            "baseMVA": 100,
            "bus": self.bus_matrix,
            "branch": self.branch_matrix,
            "gen": self.power_matrix,
        }
        # if os.path.exists("log.txt"):
        #     os.remove("log.txt")
        ppopt = ppoption(OUT_GEN=1, VERBOSE=0, PF_ALG=1)
        results, succ = runpf(ppc, ppopt, fname="log.txt")
        if not succ:
            print("Failed to run the flow calculation")
            return None

        return results

    def run_fast(self, power_demand):
                # self.update_power(power_demand)
        self.power_require = power_demand
        P = power_demand
        Q = P * math.sqrt((1 / self.basic_config["cos_phi"]) ** 2 - 1)
        self.bus_matrix[-self.trans110_num:, 2] = P
        self.bus_matrix[-self.trans110_num:, 3] = Q
        ppc = {
            "version": 2,
            "baseMVA": 100,
            "bus": self.bus_matrix,
            "branch": self.branch_matrix,
            "gen": self.power_matrix,
        }
        # print(self.bus_matrix[:10])
        # pdb.set_trace()
        # if os.path.exists("log.txt"):
        #     os.remove("log.txt")
        ppopt = ppoption(OUT_ALL=0, VERBOSE=0, PF_ALG=1)
        ## no output
        results, succ = runpf(ppc, ppopt)
        if not succ:
            print("Failed to run the flow calculation")
            return None

        return results

    def get_power_data(self):
        """
        能源设施，包括发电站和500kv变压器
        """
        Pg = 0
        Qg = 0
        power_matrix = []
        for facility_id in self.facility_list:
            if facility_id // BASE == input_pb.FACILITY_TYPE_TRANSFORMER_220:
                continue
            bus = self.bus_dict[facility_id]["in"]
            power_matrix.append(
                [
                    bus,
                    Pg,
                    Qg,
                    self.basic_config["Qmax"],
                    self.basic_config["Qmin"],
                    self.basic_config["Vg"],
                    self.basic_config["mbase"],
                    self.basic_config["status"],
                    self.basic_config["Pmax"],
                    self.basic_config["Pmin"],
                    self.basic_config["Pc1"],
                    self.basic_config["Pc2"],
                    self.basic_config["Qc1min"],
                    self.basic_config["Qc1max"],
                    self.basic_config["Qc2min"],
                    self.basic_config["Qc2max"],
                    self.basic_config["ramp_agc"],
                    self.basic_config["ramp_10"],
                    self.basic_config["ramp_30"],
                    self.basic_config["ramp_q"],
                    self.basic_config["apf"],
                ]
            )
        return np.array(power_matrix)
