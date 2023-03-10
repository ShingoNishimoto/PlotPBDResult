import datetime
import os
import shutil
from asyncio import constants
from glob import glob

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.axis import YAxis
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots

from plot_set import fig_init, fig_init_horizon_bar, fig_output, x_axis_scale


def get_latest_modified_file_path(dirname):
    target = os.path.join(dirname, "*")
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    latest_modified_file_path = sorted(files, key=lambda files: files[1])[-1]
    return latest_modified_file_path[0]


copy_base_dir = "G:/マイドライブ/Documents/University/lab/Research/FFGNSS/plot_result/"

# ここを設定
SEPARATION = 0
ISS = 0
IAR = 0
CALIBRATION = 0
RECEIVER = 0
FILTER = 0
PCV_METHOD = 0
COST_ACCURACY = 1
PCO = 0
REL_POS_BIAS = 0
ORBITS = 0

log_base = copy_base_dir
separation_files = [
    "20230124_163016_100m",
    "20230124_165001_500m",
    "20230124_100149_1km",
    "20230124_161620_2km",
    "20230128_001743_3km",
    "20230124_170120_10km",  # 一つ抜けているのでなくしていいかも
    # "w_PCCerror/20230128_094542_100m_after_PCC",
    # "w_PCCerror/20230129_130455_500m_after_PCC",
    # "w_PCCerror/20230127_170520_WLS_PCC1cycle_1km",
    # "w_PCCerror/20230129_123418_2km_after_PCC",
    # "w_PCCerror/20230127_233258_3km_afterPCCestimation",
    # # "20230124_170120_10km",  # 一つ抜けているのでなくしていいかも
]
ISS_separation_files = [
    # "20230124_230233_WLS_postPCV_ISS_sep100m",
    # "20230124_232716_WLS_postPCV_ISS_sep500m",
    # "20230128_000212_ISS_900m_postPCO",
    # "20230130_134830_WLS_PCC1cycle1km_wo_PCO_horizon",
    "wo_PCC_estimation/20230129_142557_100m",
    "wo_PCC_estimation/20230129_144719_500m",
    "wo_PCC_estimation/20230129_140414_900m",
    # "wo_PCC_estimation/20230130_140933_wo_PCC_1km",
]
IAR_files = ["20230124_105907_wo_IAR_wo_PCCerror", "20230124_100149_w_IAR_wo_PCCerror"]
PCC_calibration_files = [
    "20230124_100149_w_IAR_wo_PCCerror",
    "20230126_160837_PCC1cycle_wo_PCC_error",
    # "20230123_110838_PCC_2cycle_1km",
]
receiver_files = [
    "20230123_081904_PCC1cycle_1km_Phoenix",
    "20230123_212716_PCC1cycle_OEM",
    "20230123_231352_PCC1cycle_SILVIA",
    "20230124_002738_PCC1cycle_TsinhuaSat",
]
filter_files = ["20230125_182712_EKF_w_IAR_w_PCCerror", "20230124_103810_AEKF_w_IAR_w_PCCerror"]
pcv_method_files = [
    "20230127_163104_wo_estimation",
    "20230127_170520_WLS_PCC1cycle",
    "20230127_171952_RES_PCC1cycle",
]
orbits_csv_files = ["dr_accuracies_all_SSO_wo2km.csv", "dr_accuracies_all_ISS.csv"]
cost_accuracy_files = [
    # "20230131_191315_WLS_10_PCC1cycle",
    # "20230130_145307_WLS_20_PCC1cycle",
    # "20230127_170520_WLS_40_PCC1cycle",
    # "20230130_150824_WLS_80_PCC1cycle",
    # "Residual/20230126_010656_RES_after2560",
    # "Residual/20230130_192307_RES_after4030",
    # "Residual/20230126_003114_RES_after_6970",
    "Residual/20230131_104331_RES_90deg_10800_after",
    "Residual/20230131_170713_RES_45deg_PCC1cycle",
    "Residual/20230131_174452_RES_30deg_PCC1cycle",
    "Residual/20230131_134355_RES_PCC1cycle_10800_15deg",
]
pco_files = [
    "20230127_163104_wo_estimation",
    "20230129_170121_only_PCO",
    "20230129_171707_only_PCO_rotate",
]
rel_pos_bias_files = [
    # "20230107_201550_after_PCV_WLS_w_rel_pos_bias_1cm_z",
    # "20230107_173154_after_PCV_WLS_w_rel_pos_bias_1cm_x",
    # "20230107_194024_after_PCV_WLS_w_rel_pos_bias_1cm_y",
    "20230107_204525_after_PCV_RES_w_rel_pos_bias_1cm_z",
    "20230107_174958_after_PCV_RES_w_rel_pos_bias_1cm_x",
    "20230107_195426_after_PCV_RES_w_rel_pos_bias_1cm_y",
]
rel_pos_bias_csv_files = ["accuracies_mean.csv", "accuracies_std.csv"]

if SEPARATION:
    log_base += "202301_thesis_separation/"
    files = separation_files
    legend_names = ["100m", "500m", "1km", "2km", "3km", "10km"]
elif ISS:
    log_base += "202301_thesis_ISS/"
    files = ISS_separation_files
    legend_names = ["100m", "500m", "900m"]
    # legend_names = ["w/o PCC estimation", "w/ PCC estimation"]
elif IAR:
    log_base += "202301_thesis_IAR/"
    files = IAR_files
    legend_names = ["w/o IAR", "w/ IAR"]
elif CALIBRATION:
    log_base += "202301_thesis_PCC_error_wo_PCCestimation/"
    files = PCC_calibration_files
    legend_names = ["w/o calibration", "1 cycle", "2 cycle"]
elif RECEIVER:
    log_base += "202301_thesis_receiver_performance/"
    files = receiver_files
    legend_names = ["Phoenix", "OEM", "NGPSR", "Tsinghua"]
elif FILTER:
    log_base += "202301_thesis_AKF/"
    files = filter_files
    legend_names = ["EKF", "AEKF"]
elif PCV_METHOD:
    log_base += "202301_thesis_1km_base_results/"
    files = pcv_method_files
    legend_names = ["w/o calibration", "Model", "Residual"]
elif ORBITS:
    log_base += "202301_thesis_orbits/"
    files = orbits_csv_files
    legend_names = [["100m", "500m", "1km", "3km"], ["100m", "500m", "900m"]]
    subplot_names = ["SSO", "ISS"]
elif COST_ACCURACY:
    log_base += "202301_thesis_computational_cost_accuracy/"
    files = cost_accuracy_files
    legend_names = ["900", "1800", "3600", "7200"]  # matrix size
    # legend_names = ["2560s", "4030s", "6970s"]  # residual time
    legend_names = ["90deg", "45deg", "30deg", "15deg"]  # residual bin size
elif PCO:
    log_base += "202301_thesis_PCO/"
    files = pco_files
    legend_names = ["w/o PCO estimation", "along-track: north", "along-track: east"]  # matrix size
elif REL_POS_BIAS:
    # log_base += "202301_thesis_relpos_performance/bias/"
    log_base += "20230130_220120_rel_pos_bias_RES/"
    files = rel_pos_bias_csv_files  # rel_pos_bias_files
    legend_names = ["radial", "tangential", "normal"]  # matrix size
    subplot_names = ["mean", "std"]
else:
    print("false input!")
    abort()

log_bases = [log_base + each_dir + "/" for each_dir in files]  # fileを設定

output_path = "figure/"
os.makedirs(output_path, exist_ok=True)
dt_now = datetime.datetime.now()
copy_dir = copy_base_dir + dt_now.strftime("%Y%m%d_%H%M%S") + "/"
os.makedirs(copy_dir)
# shutil.copytree(s2e_log_dir, copy_dir, ignore=shutil.ignore_patterns("figure"))
accuracy_file = copy_dir + "accuracies.csv"

# accuracy_log = pd.DataFrame(columns=["name", "axis", "unit", "value"])  # ここに計算した精度を入れていく．
accuracy_log = pd.DataFrame(
    columns=["legend", "name", "axis", "unit", "mean", "std", "rms"]
)  # ここに計算した精度を入れていく．

# この二つはどっかからとってきたいな．
REDUCE_DYNAMIC = 1
GNSS_CH_NUM = 15
# SVG_ENABLE = 1
# PCV = 0
data_offset = 1000  # s 6580(WLS)
x_axis_scale = 1.0 / 3600  # sec -> hour


# true information
col_true_info = [
    "x_m_t",
    "y_m_t",
    "z_m_t",
    "t_m_t",
    "vx_m_t",
    "vy_m_t",
    "vz_m_t",
    "x_t_t",
    "y_t_t",
    "z_t_t",
    "t_t_t",
    "vx_t_t",
    "vy_t_t",
    "vz_t_t",
]
col_est_info = [
    "x_m_e",
    "y_m_e",
    "z_m_e",
    "t_m_e",
    "vx_m_e",
    "vy_m_e",
    "vz_m_e",
    "ar_m_e",
    "at_m_e",
    "an_m_e",
    "x_t_e",
    "y_t_e",
    "z_t_e",
    "t_t_e",
    "vx_t_e",
    "vy_t_e",
    "vz_t_e",
    "ar_t_e",
    "at_t_e",
    "an_t_e",
]
col_residual = [
    "res_pos_R_m",
    "res_pos_T_m",
    "res_pos_N_m",
    "res_vel_R_m",
    "res_vel_T_m",
    "res_vel_N_m",
    "res_pos_R_t",
    "res_pos_T_t",
    "res_pos_N_t",
    "res_vel_R_t",
    "res_vel_T_t",
    "res_vel_N_t",
]
col_ambiguity_main = ["N" + str(i + 1) + "_m_t" for i in range(GNSS_CH_NUM)] + [
    "N" + str(i + 1) + "_m_e" for i in range(GNSS_CH_NUM)
]
col_ambiguity_target = ["N" + str(i + 1) + "_t_t" for i in range(GNSS_CH_NUM)] + [
    "N" + str(i + 1) + "_t_e" for i in range(GNSS_CH_NUM)
]
col_ambiguity = col_ambiguity_main + col_ambiguity_target

col_MN_main = ["MN" + str(i + 1) + "_m" for i in range(GNSS_CH_NUM)]
col_MN_target = ["MN" + str(i + 1) + "_t" for i in range(GNSS_CH_NUM)]

col_M = (
    ["Mx_m", "My_m", "Mz_m", "Mt_m", "Mvx_m", "Mvy_m", "Mvz_m", "Mar_m", "Mat_m", "Man_m"]
    + col_MN_main
    + ["Mx_t", "My_t", "Mz_t", "Mt_t", "Mvx_t", "Mvy_t", "Mvz_t", "Mar_t", "Mat_t", "Man_t"]
    + col_MN_target
    + [
        "Mrr_m",
        "Mrt_m",
        "Mrn_m",
        "Mvr_m",
        "Mvt_m",
        "Mvn_m",
        "Mrr_t",
        "Mrt_t",
        "Mrn_t",
        "Mvr_t",
        "Mvt_t",
        "Mvn_t",
    ]
)

col_gnss_dir_main = ["azi" + str(i + 1) + "_m" for i in range(GNSS_CH_NUM)] + [
    "ele" + str(i + 1) + "_m" for i in range(GNSS_CH_NUM)
]
col_gnss_dir_target = ["azi" + str(i + 1) + "_t" for i in range(GNSS_CH_NUM)] + [
    "ele" + str(i + 1) + "_t" for i in range(GNSS_CH_NUM)
]
col_gnss_dir = col_gnss_dir_main + col_gnss_dir_target

col_pco = ["pco_x_m", "pco_y_m", "pco_z_m", "pco_x_t", "pco_y_t", "pco_z_t"]
col_ambiguity_flag = ["N" + str(i + 1) + "_m" for i in range(GNSS_CH_NUM)] + [
    "N" + str(i + 1) + "_t" for i in range(GNSS_CH_NUM)
]

data_col = (
    col_true_info
    + col_est_info
    + col_residual
    + col_ambiguity
    + col_M
    + ["sat_num_main", "sat_num_target", "sat_num_common"]
    + ["id_ch" + str(i + 1) + "_m" for i in range(GNSS_CH_NUM)]
    + ["id_ch" + str(i + 1) + "_t" for i in range(GNSS_CH_NUM)]
    + ["Qx_m", "Qy_m", "Qz_m", "Qt_m", "Qvx_m", "Qvy_m", "Qvz_m", "Qar_m", "Qat_m", "Qan_m"]
    + ["QN" + str(i + 1) + "_m" for i in range(GNSS_CH_NUM)]
    + ["Qx_t", "Qy_t", "Qz_t", "Qt_t", "Qvx_t", "Qvy_t", "Qvz_t", "Qar_t", "Qat_t", "Qan_t"]
    + ["QN" + str(i + 1) + "_t" for i in range(GNSS_CH_NUM)]
    + ["Rgr" + str(i + 1) + "_m" for i in range(GNSS_CH_NUM)]
    + ["Rgr" + str(i + 1) + "_t" for i in range(GNSS_CH_NUM)]
    + ["Rcp" + str(i + 1) for i in range(GNSS_CH_NUM)]
    + [
        "ax_m",
        "ay_m",
        "az_m",
        "ax_t",
        "ay_t",
        "az_t",
        "ax_dist_m",
        "ay_dist_m",
        "az_dist_m",
        "ax_dist_t",
        "ay_dist_t",
        "az_dist_t",
        "ar_dist_m",
        "at_dist_m",
        "an_dist_m",
        "ar_dist_t",
        "at_dist_t",
        "an_dist_t",
    ]
    + col_gnss_dir
    + col_pco
    + col_ambiguity_flag
    + [""]
)
acc_col = [
    "Mar_m",
    "Mat_m",
    "Man_m",
    "Mar_t",
    "Mat_t",
    "Man_t",
    "Qar_m",
    "Qat_m",
    "Qan_m",
    "Qar_t",
    "Qat_t",
    "Qan_t",
]

s2e_data_col = [
    "sat_position_i(X)[m]",
    "sat_position_i(Y)[m]",
    "sat_position_i(Z)[m]",
    "sat_acc_i_i(X)[m/s^2]",
    "sat_acc_i_i(Y)[m/s^2]",
    "sat_acc_i_i(Z)[m/s^2]",
    "sat_acc_rtn_rtn(X)[m/s^2]",
    "sat_acc_rtn_rtn(Y)[m/s^2]",
    "sat_acc_rtn_rtn(Z)[m/s^2]",
    "gnss_position_eci(X)[m]",
    "gnss_position_eci(Y)[m]",
    "gnss_position_eci(Z)[m]",
    "gnss_arp_true_eci(X)[m]",
    "gnss_arp_true_eci(Y)[m]",
    "gnss_arp_true_eci(Z)[m]",
]
s2e_data_col += [col + ".1" for col in s2e_data_col]


def find_log_data(log_base: str) -> list:
    """
    log_base: plotスクリプトを実行した後の一つのログフォルダパス
    return:
        [data, data_s2e]: 独自csvとs2e logのPandasDataFrameをまとめたリスト
    """
    sim_config_path = log_base + "readme.txt"
    log_path = log_base + "result.csv"
    s2e_log_dir = log_base
    s2e_csv_log_path = (
        s2e_log_dir
        + [file for file in os.listdir(s2e_log_dir) if file.endswith(".csv")][0]  # バグりそう．
    )
    # pcc_log_path = log_base + "pcc_1.csv"
    # pcv_log_path = log_base + "pcv_1.csv"

    data = pd.read_csv(log_path, header=None)
    if len(data.columns) != len(data_col):
        REDUCE_DYNAMIC = 0
        for name in acc_col:
            data_col.remove(name)
    data = data.set_axis(data_col, axis=1)

    # ログの頻度を合わせて回すこと！
    data_s2e = pd.read_csv(s2e_csv_log_path)

    return [data, data_s2e]


# 3drmsを含めてしまいたいが
def add_accuracy(
    precision: pd.DataFrame(), legend_name: str, name: str, axis: str, unit: str
) -> None:
    global accuracy_log
    accuracy_log = accuracy_log.append(
        {
            "legend": legend_name,
            "name": name,
            "axis": axis,
            "unit": unit,
            "mean": float("{:.3f}".format(precision.mean())),
            "std": float("{:.3f}".format(precision.std())),
            "rms": float("{:.3f}".format(np.sqrt(np.mean(precision**2)))),
        },
        ignore_index=True,
    )


def add_3d_rms(
    precision: pd.DataFrame(), legend_name: str, name: str, axes: list, unit: str
) -> None:
    global accuracy_log
    _3d_ms = 0
    for axis in axes:
        _3d_ms += np.mean(precision[axis] ** 2)
    _3d_rms = np.sqrt(_3d_ms)
    accuracy_log = accuracy_log.append(
        {
            "legend": legend_name,
            "name": name,
            "axis": "3d",
            "unit": unit,
            "rms": float("{:.3f}".format(_3d_rms)),
        },
        ignore_index=True,
    )


def get_suffix(m_t: str) -> str:
    if m_t == "m":
        return "main"
    elif m_t == "t":
        return "target"
    else:
        # print("ERROR: input error!")
        return ""


def DCM_from_eci_to_lvlh(base_pos_vel):
    r_i, v_i = base_pos_vel
    z_lvlh_i = -r_i / np.linalg.norm(r_i)
    x_lvlh_i = v_i / np.linalg.norm(v_i)
    y_lvlh_i = np.cross(z_lvlh_i, x_lvlh_i)
    x_lvlh_i = np.cross(y_lvlh_i, z_lvlh_i)
    DCM_from_eci_to_lvlh = np.array([x_lvlh_i, y_lvlh_i, z_lvlh_i])
    return DCM_from_eci_to_lvlh


def DCM_from_eci_to_rtn(base_pos_vel):
    r_i, v_i = base_pos_vel
    r_rtn_i = r_i / np.linalg.norm(r_i)
    t_rtn_i = v_i / np.linalg.norm(v_i)
    n_rtn_i = np.cross(r_rtn_i, t_rtn_i)
    t_rtn_i = np.cross(n_rtn_i, r_rtn_i)
    DCM_from_eci_to_rtn = np.array([r_rtn_i, t_rtn_i, n_rtn_i])
    return DCM_from_eci_to_rtn


def calc_relinfo(r_v_a, t_e, data):
    """
    Args:
        r_v_a: position or velocity or acceleration
        t_e: true or estimation
        data: data(PandasDataframe)
    """
    if r_v_a == "r":
        base_names = ["x", "y", "z"]
    elif r_v_a == "v":
        base_names = ["vx", "vy", "vz"]
    elif r_v_a == "a":
        base_names = ["ar", "at", "an"]
    else:
        print("false input")
        return

    frame_names = ["R", "T", "N"]
    col_names = ["d" + r_v_a + "_" + frame for frame in frame_names]
    names_main = [base_name + "_m_" + t_e for base_name in base_names]
    names_target = [base_name + "_t_" + t_e for base_name in base_names]
    base = pd.DataFrame()
    for i in range(len(base_names)):
        base[col_names[i]] = data[names_target[i]] - data[names_main[i]]
    return base


# 加速度の精度
def calc_a_precision(
    data: pd.DataFrame(), data_s2e_log: pd.DataFrame(), m_t: str, frame: str
) -> pd.DataFrame():
    if frame == "eci":
        a_base_names = ["ax", "ay", "az"]
        a_names = [base_name + "_" + m_t for base_name in a_base_names]
        a_t_names = [
            "sat_acc_i_i(X)[m/s^2]",
            "sat_acc_i_i(Y)[m/s^2]",
            "sat_acc_i_i(Z)[m/s^2]",
        ]  # 一旦べた書き
    elif frame == "rtn":
        a_base_names = ["ar", "at", "an"]
        a_names = [base_name + "_" + m_t + "_e" for base_name in a_base_names]
        a_t_names = [
            "sat_acc_rtn_rtn(X)[m/s^2]",
            "sat_acc_rtn_rtn(Y)[m/s^2]",
            "sat_acc_rtn_rtn(Z)[m/s^2]",
        ]
    else:
        return
    a_dist_names = [base_name + "_dist_" + m_t for base_name in a_base_names]
    a_e = pd.DataFrame()
    for i in range(3):
        a_e[a_base_names[i]] = data[a_names[i]] * 1e-3 + data[a_dist_names[i]] * 1e6  # um/s2

    if m_t == "t":
        a_t_names = [name + ".1" for name in a_t_names]
    a_t = pd.DataFrame()
    for i in range(3):
        a_t[a_base_names[i]] = data_s2e_log[a_t_names[i]] * 1e6  # um/s2
    a_t = a_t[1:].reset_index()  # ここはs2e_logとのindexずれがあるので補正している．

    M_names = ["M" + name + "_" + m_t for name in a_base_names]

    precision = pd.DataFrame()
    for i in range(3):
        precision[a_names[i]] = a_e[a_base_names[i]] - a_t[a_base_names[i]]
    # Mのデータも追加．
    for i in range(3):
        precision[M_names[i]] = data[M_names[i]] * (1e-3) ** 2  # um/s2 Mは2乗なので
        data[M_names[i]] = precision[M_names[i]]  # dataの方にもコピー
    return precision


red_list = [(103, 0, 13), (165, 15, 21), (203, 24, 29), (239, 59, 44), (251, 106, 74)]  # Reds

color_list = [
    "darkred",
    "tomato",
    "goldenrod",
    "chartreuse",
    "darkcyan",
    "darkblue",
    "darkmagenta",
    # "darkred",
    # "chartreuse",
    # "darkblue",
    # "darkmagenta",
    # "tomato",
    # "goldenrod",
    # "darkcyan",
]


class PlotSettings:
    def __init__(self, r_v_a: str, m_t: str, data_type: str):
        self.r_v_a = r_v_a
        self.m_t = m_t
        self.suffix = get_suffix(m_t)

        if data_type == "abs_rtn":
            self.init_abs_rtn()
        elif data_type == "rel_rtn":
            self.init_rel_rtn()
        else:
            print("false input!")

    # 使わない．
    # def init_abs_eci(self):
    #     if self.r_v_a == "r":
    #         self.base_names = ["x", "y", "z"]
    #         self.axis_names = base_names
    #         self.unit = "m"
    #         self.scale_param = 1
    #     elif self.r_v_a == "v":
    #         self.base_names = ["vx", "vy", "vz"]
    #         self.axis_names = ["v_x", "v_y", "v_z"]
    #         self.unit = "mm/s"
    #         self.scale_param = 1000
    #     elif self.r_v_a == "a":
    #         self.base_names = ["ax", "ay", "az"]
    #         self.axis_names = ["a_x", "a_y", "a_z"]
    #         self.unit = "um/s^2"
    #         self.scale_param = 1e-3
    #     else:
    #         print("false input")

    def init_abs_rtn(self):
        if self.r_v_a == "r":
            col_base_name = "res_pos_"
            self.unit = "m"
            self.scale_param = 1
            self.output_name = "position_rtn"
            self.M_names_base = ["Mrr", "Mrt", "Mrn"]
            self.y_range = 2.5
        elif self.r_v_a == "v":
            col_base_name = "res_vel_"
            self.unit = "mm/s"
            self.scale_param = 1000
            self.output_name = "velocity_rtn"
            self.M_names_base = ["Mvr", "Mvt", "Mvn"]
            self.y_range = 5.0
        elif self.r_v_a == "a":
            self.col_names = ["a_R", "a_T", "a_N"]
            self.unit = "um/s^2"
            self.scale_param = 1
            self.output_name = "a_rtn"
            self.M_names_base = ["Mar", "Mat", "Man"]
            self.y_range = 20.0  # ホンマはautoにしたい．
        else:
            print("false input")
            return

        self.frame_names = ["R", "T", "N"]
        # ちょっと気持ち悪いので統一したい．
        if self.r_v_a != "a":
            self.col_names = [
                col_base_name + frame_name + "_" + self.m_t for frame_name in self.frame_names
            ]
        self.base_name = self.r_v_a
        self.names = [self.base_name + "_" + frame for frame in self.frame_names]

    def init_rel_rtn(self):
        if self.r_v_a == "r":
            self.unit = "cm"
            self.scale_param = 100
            self.output_name = "relative_position"
            self.M_names = ["Mrr", "Mrt", "Mrn"]
            self.y_range = 0.14 * self.scale_param  # 0.18
        elif self.r_v_a == "v":
            self.unit = "mm/s"
            self.scale_param = 1000
            self.output_name = "relative_velocity"
            self.M_names = ["Mvr", "Mvt", "Mvn"]
            self.y_range = 1.2  # 3.5
        elif self.r_v_a == "a":
            self.unit = "um/s^2"
            self.scale_param = 1
            self.output_name = "relative_a"
            self.M_names = ["Mar", "Mat", "Man"]
            self.y_range = 0.07
        else:
            print("input error!")
            return

        self.frame_names = ["R", "T", "N"]
        self.base_name = "d" + self.r_v_a
        self.col_names = [self.base_name + "_" + frame_name for frame_name in self.frame_names]
        self.names = self.col_names


def plot_multiple_precision(r_v_a: str, m_t: str, data_type: str, datas: list) -> None:
    plot_set = PlotSettings(r_v_a, m_t, data_type)
    fig = fig_init(datas[0], plot_set.names, plot_set.unit)

    for i, data in enumerate(datas):
        if data_type == "abs_rtn":
            plot_precision_rtn(fig, data, plot_set, color_list[i], legend_name=legend_names[i])
        elif data_type == "rel_rtn":
            plot_differential_precision(fig, data, plot_set, color_list[i], legend_names[i])
    fig_output(fig, output_path + plot_set.output_name + "_precision_" + plot_set.suffix)


# 座標系に対しても統合したい．
def plot_precision_rtn(
    fig, data: pd.DataFrame(), plot_set: PlotSettings, color_str: str, legend_name: str
) -> None:

    data_for_plot = data * plot_set.scale_param
    # M_names = [M_name_base + "_" + m_t for M_name_base in M_names_base]

    for i in range(len(plot_set.names)):
        add_accuracy(
            data_for_plot.loc[data_offset:, plot_set.col_names[i]],
            legend_name,
            plot_set.base_name + "_" + plot_set.m_t,
            plot_set.frame_names[i],
            plot_set.unit,
        )
        RMS = np.sqrt(np.mean(data_for_plot.loc[data_offset:, plot_set.col_names[i]] ** 2))
        show_legend = False
        if i == 0:
            show_legend = True
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=data_for_plot.index * x_axis_scale,
                y=data_for_plot.loc[:, plot_set.col_names[i]],
                name=legend_name,
                legendgroup=str(i + 1),
                marker=dict(size=2, color=color_str),
                # marker=dict(
                #     size=2,
                # ),
                showlegend=show_legend,
            ),
            row=(i + 1),
            col=1,
        )
        # # 1 sigmaを計算してプロットする．
        # fig.add_trace(
        #     go.Scatter(
        #         x=data.index * x_axis_scale,
        #         y=np.sqrt(data[M_names[i]]) * scale_param,
        #         legendgroup=str(i),
        #         line=dict(width=1, color="black"),
        #         showlegend=False,
        #     ),
        #     row=(i + 1),
        #     col=1,
        # )
        # fig.add_trace(
        #     go.Scatter(
        #         x=data.index * x_axis_scale,
        #         y=-np.sqrt(data[M_names[i]]) * scale_param,
        #         legendgroup=str(i),
        #         line=dict(width=1, color="black"),
        #         showlegend=False,
        #     ),
        #     row=(i + 1),
        #     col=1,
        # )
        fig.update_yaxes(
            range=(-plot_set.y_range, plot_set.y_range),
            row=(i + 1),
            col=1,
        )
    add_3d_rms(
        data_for_plot.loc[data_offset:, :],
        legend_name,
        plot_set.base_name + "_" + plot_set.m_t,
        plot_set.col_names,
        plot_set.unit,
    )
    # fig_output(fig, output_path + output_name + "_precision_" + suffix)


def trans_eci_to_rtn(DCMs, precision_data):
    for i in range(len(precision_data)):
        DCM = DCMs[i]
        precision_data.iloc[i, :] = np.dot(DCM, precision_data.iloc[i, :])
    return precision_data


def plot_differential_precision(
    fig, precision_data: pd.DataFrame(), plot_set: PlotSettings, color_str: str, legend_name: str
) -> None:
    precision_data *= plot_set.scale_param
    # # eci -> rtnに変換
    # if input_frame == "ECI":
    #     for i in range(len(precision_data)):
    #         DCM = DCM_from_eci_to_rtn(
    #             (data.loc[i, "x_m_t":"z_m_t"], data.loc[i, "vx_m_t":"vz_m_t"])
    #         )
    #         precision_data.iloc[i, :] = np.dot(DCM, precision_data.iloc[i, :])

    for i in range(len(plot_set.names)):
        add_accuracy(
            precision_data.iloc[data_offset:, i],
            legend_name,
            plot_set.base_name,
            plot_set.frame_names[i],
            plot_set.unit,
        )
        RMS = np.sqrt(np.mean(precision_data.iloc[data_offset:, i] ** 2))
        show_legend = False
        if i == 0:
            show_legend = True
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=precision_data.index * x_axis_scale,
                y=precision_data.iloc[:, i],
                name=legend_name,
                legendgroup=str(i + 1),
                marker=dict(size=2, color=color_str),
                # marker=dict(size=2),
                showlegend=show_legend,
            ),
            row=(i + 1),
            col=1,
        )
        # 分散のところは絶対的な状態量から出しているもので正確ではないのでplotしない．
        fig.update_yaxes(
            range=(-plot_set.y_range, plot_set.y_range),
            row=(i + 1),
            col=1,
        )
    add_3d_rms(
        precision_data.loc[data_offset:, :],
        legend_name,
        plot_set.base_name,
        plot_set.col_names,
        plot_set.unit,
    )
    # fig_output(fig, output_path + output_name + "_precision")


def plot_multiple_bar(
    accuracy_data: pd.DataFrame(), categories: list, r_v_a: str, m_t: str, data_type: str
) -> None:
    plot_set = PlotSettings(r_v_a, m_t, data_type)
    precision_title = "$" + plot_set.base_name + "[\\text{" + plot_set.unit + "}]$"
    fig = fig_init_horizon_bar([accuracy_data], legend_names, precision_title)

    for i, category in enumerate(categories):
        fig = plot_accuracy_bar(
            fig, accuracy_data[accuracy_data["legend"] == category], category, color_list[i], 1
        )

    fig.update_layout(barmode="group")
    fig_output(fig, output_path + plot_set.output_name + "_precision_bar")


def plot_accuracy_bar(
    fig: go.Figure(), accuracy_data: pd.DataFrame(), category: str, color_str: str, col_num: int
) -> go.Figure():
    fig.add_trace(
        go.Bar(
            name=category,
            x=accuracy_data["axis"],
            y=accuracy_data["rms"],
            # text=accuracy_data["std"],
            texttemplate="%{y:.3f}",
            textposition="outside",
            textfont=dict(size=25),
            marker_color=color_str,
            legendgroup=str(col_num),
            # showlegend=False,
        ),
        row=1,
        col=col_num,
    )
    # mean, stdもプロットしたいときはsubplotごとにする．
    return fig


def plot_accuracy_comparison_w_subplot(
    r_v_a: str, m_t: str, data_type: str, datas: list, categories: list, subplot_names: list
) -> None:
    plot_set = PlotSettings(r_v_a, m_t, data_type)
    fig = fig_init_horizon_bar(datas, plot_set.names, plot_set.unit)

    for j, data in enumerate(datas):
        for i, category in enumerate(categories[j]):  # categories[j]
            fig = plot_accuracy_bar(
                fig, data[data["legend"] == category], category, color_list[i], j + 1
            )
        fig.update_xaxes(
            col=j + 1,
            title=dict(
                text=subplot_names[j],
                # standoff=2,
            ),
        )
    fig.update_traces(
        showlegend=True,
        row=1,
        col=1,
    )

    fig_output(fig, output_path + plot_set.output_name + "_precision_bar_all")


# pd.options.display.float_format = "{:.2f}".format
# pd.options.display.precision = 3
data_type_abs = "abs_rtn"
data_type_rel = "rel_rtn"

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
datas_list = []
for log_path in log_bases:
    datas_list.append(find_log_data(log_path))
datas, s2e_datas = list(zip(*datas_list))

dr_precisions = []
dv_precisions = []
da_precisions = []
# am_precisions = []
# at_precisions = []

# この辺の計算が重いからやっぱり一回csvに残しておきたいな．
DCMs = []  # 主衛星の軌道は変わらないので使いまわせる．
data = datas[0]
for i in range(len(data)):
    DCMs.append(
        DCM_from_eci_to_rtn((data.loc[i, "x_m_t":"z_m_t"], data.loc[i, "vx_m_t":"vz_m_t"]))
    )
for data, s2e_data in zip(datas, s2e_datas):
    dr_precision = calc_relinfo("r", "e", data) - calc_relinfo("r", "t", data)
    dr_precisions.append(trans_eci_to_rtn(DCMs, dr_precision))

    dv_precision = calc_relinfo("v", "e", data) - calc_relinfo("v", "t", data)
    dv_precisions.append(trans_eci_to_rtn(DCMs, dv_precision))

    # a_m_precision = calc_a_precision(data, s2e_data, "m", "rtn")
    # a_t_precision = calc_a_precision(data, s2e_data, "t", "rtn")
    # # plot_precision_rtn("a", "m", a_m_precision)
    # # plot_precision_rtn("a", "t", a_t_precision)
    # da_precision = calc_relinfo("a", "e", pd.concat([a_m_precision, a_t_precision], axis=1))
    # da_precisions.append(da_precision)
    # # plot_differential_precision(
    # #     data, a_t_precision.iloc[:, 0:3] - a_m_precision.iloc[:, 0:3], "a", "RTN"
    # # )

plot_multiple_precision("r", "m", data_type_abs, datas)
# plot_multiple_precision("r", "t", data_type_abs, datas)
plot_multiple_precision("v", "m", data_type_abs, datas)
# plot_multiple_precision("v", "t", data_type_abs, datas)

plot_multiple_precision("r", "", data_type_rel, dr_precisions)
plot_multiple_precision("v", "", data_type_rel, dv_precisions)
# plot_multiple_precision("a", "", data_type_rel, da_precisions)


# bar plot
plot_multiple_bar(
    # accuracy_log.query('name.str().contains("dr")'),  # engine="python"),
    accuracy_log[accuracy_log["name"] == "dr"],
    legend_names,
    "r",
    "",
    data_type_rel,
)
plot_multiple_bar(
    # accuracy_log.query('name.str().contains("dv")'),  # engine="python"),
    accuracy_log[accuracy_log["name"] == "dv"],
    legend_names,
    "v",
    "",
    data_type_rel,
)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# accuracy_logs = [log_base + log for log in files]
# accuracy_datas = []
# for file in accuracy_logs:
#     accuracy_datas.append(pd.read_csv(file))

# plot_accuracy_comparison_w_subplot(
#     "r", "", data_type_rel, accuracy_datas, legend_names, subplot_names
# )

accuracy_log.to_csv(accuracy_file, sep=",", index=False)
# 最後に全グラフをまとめてコピー
# shutil.move(output_path, copy_dir + "figure/")
shutil.move(output_path, copy_dir)
