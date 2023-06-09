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

from plot_set import fig_init, fig_output, set_fig_params, x_axis_scale


def get_latest_modified_file_path(dirname):
    target = os.path.join(dirname, "*")
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    latest_modified_file_path = sorted(files, key=lambda files: files[1])[-1]
    return latest_modified_file_path[0]


s2e_dir = "../../s2e_pbd/"
s2e_log_path_base = s2e_dir + "data/logs"
s2e_debug = s2e_dir + "CMakeBuilds/Debug/"
s2e_log_dir = get_latest_modified_file_path(s2e_log_path_base) + "/"
s2e_csv_log_path = (
    s2e_log_dir + [file for file in os.listdir(s2e_log_dir) if file.endswith(".csv")][0]
)
# ここは適宜修正する．
# copy_base_dir = "/mnt/g/マイドライブ/Documents/University/lab/Research/FFGNSS/plot_result/"
copy_base_dir = "G:/マイドライブ/Documents/University/lab/Research/FFGNSS/plot_result/"
sim_config_path = s2e_log_dir + "readme.txt"
log_path = s2e_log_dir + "result.csv"
pcc_log_path = s2e_log_dir + "pcc_1.csv"
pcv_log_path = s2e_log_dir + "pcv_1.csv"
sdcp_residual_log_path = s2e_log_dir + "sdcp_residual.csv"

# # # コピーしたログファイルからグラフ描くとき===========
# log_base = copy_base_dir + "20230306_105605_1km/"  # ここを修正
# # log_base = "D:\Documents\Project\S2E\s2e_pbd\data\logs\logs_230201_182357/"
# sim_config_path = log_base + "readme.txt"
# log_path = log_base + "result.csv"
# s2e_log_dir = log_base
# s2e_csv_log_path = (
#     s2e_log_dir + [file for file in os.listdir(s2e_log_dir) if file.endswith(".csv")][0]  # バグりそう．
# )
# pcc_log_path = log_base + "pcc_1.csv"
# pcv_log_path = log_base + "pcv_1.csv"
# sdcp_residual_log_path = log_base + "sdcp_residual.csv"
# s2e_debug = log_base
# ================================================

# 実行時にメモ残すようにするといいかも

output_path = "figure/"
os.makedirs(output_path, exist_ok=True)
dt_now = datetime.datetime.now()
copy_dir = copy_base_dir + dt_now.strftime("%Y%m%d_%H%M%S") + "/"
# os.mkdir(copy_dir) # treeでcopyするので必要なし
shutil.copytree(s2e_log_dir, copy_dir, ignore=shutil.ignore_patterns("figure"))
accuracy_file = copy_dir + "accuracies.txt"
# 独自csvをすべてコピー．
# for file in glob(s2e_debug + "*.csv"):
#     shutil.copy(file, copy_dir)

data = pd.read_csv(log_path, header=None)
accuracy_log = pd.DataFrame(columns=["name", "axis", "unit", "value"])  # ここに計算した精度を入れていく．

# この二つはどっかからとってきたいな．
REDUCE_DYNAMIC = 1
GNSS_CH_NUM = 15
PCV = 0
# 精度評価には最後の1000sくらいのデータを使うようにする．
# data_offset = len(data) - 1000  # s
data_offset = 1000  # s 6970(WLS)
x_axis_scale = 1.0 / 3600  # sec -> hour
# constant（ここはorbit_initツールで設定した値に更新する！）
mu0 = 3.986 * 10**14  # [m^3/s^2]
a = 7078.136 * 10**3  # m

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


if len(data.columns) != len(data_col):
    REDUCE_DYNAMIC = 0
    for name in acc_col:
        data_col.remove(name)

data = data.set_axis(data_col, axis=1)

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
# data_s2e_csv = pd.read_csv(s2e_csv_log_path, usecols=s2e_data_col) # ログの頻度を合わせて回すこと！
data_s2e_csv = pd.read_csv(s2e_csv_log_path)


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
    names_main = [base_name + "_m_" + t_e for base_name in base_names]
    names_target = [base_name + "_t_" + t_e for base_name in base_names]
    base = pd.DataFrame()
    for i in range(len(base_names)):
        base[base_names[i]] = data[names_target[i]] - data[names_main[i]]
    return base


# base = calc_relinfo("r", "t", data)
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=data.index, y=base.iloc[:, 0], name=r"$b_x$"))
# fig.add_trace(go.Scatter(x=data.index, y=base.iloc[:, 1], name=r"$b_y$"))
# fig.add_trace(go.Scatter(x=data.index, y=base.iloc[:, 2], name=r"$b_z$"))
# fig.update_xaxes(title_text="$t[\\text{s}]$")
# fig.update_yaxes(title_text="$b[\\text{m}]$")
# fig_output(fig, output_path + "baseline")
# fig.show()

# lvlh
def plot_rel_state(r_v_a: str, rel_data: pd.DataFrame()) -> None:
    if r_v_a == "r":
        base_names = ["x", "y", "z"]
        axis_names = base_names
        unit = "m"
        scale_param = 1
    elif r_v_a == "v":
        base_names = ["vx", "vy", "vz"]
        axis_names = ["v_x", "v_y", "v_z"]
        unit = "mm/s"
        scale_param = 1000
    elif r_v_a == "a":
        base_names = ["ax", "ay", "az"]
        axis_names = ["a_x", "a_y", "a_z"]
        unit = "um/s^2"
        scale_param = 1e-3
    else:
        print("false input")
        return
    fig = fig_init(rel_data, axis_names, unit)
    colors = ["red", "blue", "green"]

    for i in range(len(base_names)):
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=rel_data.index * x_axis_scale,
                y=rel_data.iloc[:, i],
                name="$" + axis_names[i] + "$",
                legendgroup=str(i + 1),
                marker=dict(size=2, color=colors[i]),
            ),
            row=(i + 1),
            col=1,
        )
        # 分散のところは絶対的な状態量から出しているもので正確ではないのでplotしない．
        # fig.update_yaxes(
        #     range=(-y_range, y_range),
        #     row=(i + 1),
        #     col=1,
        # )

    fig_output(fig, output_path + "relative_" + r_v_a)


def plot_separation_eci(baseline: pd.DataFrame()) -> None:
    fig = fig_init(data, ["separation"], "m")
    # colors = ["red", "blue", "green"]
    separation = np.sqrt((baseline * baseline).sum(axis=1))
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=separation.index * x_axis_scale,
            y=separation,
            marker=dict(size=2),
        ),
    )
    fig_output(fig, output_path + "separation")


def trans_eci2rtn(position, velocity):
    DCM_eci_to_rtn = 1


# 後でちゃんと実装する．
def generate_fig_3axis_precision(
    base_names,
    data,
):
    for i in range(len(names)):
        RMS = np.sqrt(np.mean(data_for_plot.loc[data_offset:, col_names[i]] ** 2))
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=data_for_plot.index * x_axis_scale,
                y=data_for_plot.loc[:, col_names[i]],
                name="RMS:" + "{:.3f}".format(RMS) + "[" + unit + "]",
                legendgroup=str(i + 1),
                marker=dict(size=2, color=colors[i]),
            ),
            row=(i + 1),
            col=1,
        )
        # 1 sigmaを計算してプロットする．
        fig.add_trace(
            go.Scatter(
                x=data.index * x_axis_scale,
                y=np.sqrt(data[M_names[i]]) * scale_param,
                legendgroup=str(i),
                line=dict(width=1, color="black"),
                showlegend=False,
            ),
            row=(i + 1),
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.index * x_axis_scale,
                y=-np.sqrt(data[M_names[i]]) * scale_param,
                legendgroup=str(i),
                line=dict(width=1, color="black"),
                showlegend=False,
            ),
            row=(i + 1),
            col=1,
        )
    return fig


# 3drmsを含めてしまいたいが
def add_accuracy(precision: pd.DataFrame(), name: str, axis: str, unit: str) -> None:
    global accuracy_log
    accuracy_log = accuracy_log.append(
        {"name": name + "_" + "mean", "axis": axis, "unit": unit, "value": precision.mean()},
        ignore_index=True,
    )
    accuracy_log = accuracy_log.append(
        {"name": name + "_" + "std", "axis": axis, "unit": unit, "value": precision.std()},
        ignore_index=True,
    )
    accuracy_log = accuracy_log.append(
        {
            "name": name + "_" + "rms",
            "axis": axis,
            "unit": unit,
            "value": np.sqrt(np.mean(precision**2)),
        },
        ignore_index=True,
    )


def add_3d_rms(precision: pd.DataFrame(), name: str, axes: list, unit: str) -> None:
    global accuracy_log
    _3d_ms = 0
    for axis in axes:
        _3d_ms += np.mean(precision[axis] ** 2)
    _3d_rms = np.sqrt(_3d_ms)
    accuracy_log = accuracy_log.append(
        {"name": name + "_" + "3drms", "axis": "", "unit": unit, "value": _3d_rms},
        ignore_index=True,
    )


def plot_precision(r_v_a, m_t, data):
    if r_v_a == "r":
        base_names = ["x", "y", "z"]
        axis_names = base_names
        unit = "m"
        scale_param = 1
    elif r_v_a == "v":
        base_names = ["vx", "vy", "vz"]
        axis_names = ["v_x", "v_y", "v_z"]
        unit = "mm/s"
        scale_param = 1000
    elif r_v_a == "a":
        base_names = ["ax", "ay", "az"]
        axis_names = ["a_x", "a_y", "a_z"]
        unit = "um/s^2"
        scale_param = 1e-3
    else:
        print("false input")
        return

    suffix = get_suffix(m_t)
    precision = pd.DataFrame()
    fig = fig_init(data=data, names=axis_names, unit=unit)
    colors = ["red", "blue", "green"]
    for i, name in enumerate(base_names):
        est_name = name + "_" + m_t + "_e"
        true_name = name + "_" + m_t + "_t"
        precision[name] = (data[est_name] - data[true_name]) * scale_param
        add_accuracy(precision.loc[data_offset:, name], r_v_a + "_" + m_t, name, unit)
        RMS = np.sqrt(np.mean(precision.loc[data_offset:, name] ** 2))  # ここで再計算するのか？
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=data.index * x_axis_scale,
                y=precision[name],
                name="$" + name + ":" + "{:.3f}".format(RMS) + "(\\text{" + unit + "})$",
                legendgroup=str(i),
                marker=dict(size=2, color=colors[i]),
                showlegend=True,
            ),
            row=(i + 1),
            col=1,
        )
        M_name = "M" + name + "_" + m_t
        fig.add_trace(
            go.Scatter(
                x=data.index * x_axis_scale,
                y=np.sqrt(data[M_name]) * scale_param,
                name="1 sigma",
                legendgroup=str(i),
                line=dict(width=1, color="black"),
                showlegend=False,
            ),
            row=(i + 1),
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.index * x_axis_scale,
                y=-np.sqrt(data[M_name]) * scale_param,
                legendgroup=str(i),
                line=dict(width=1, color="black"),
                showlegend=False,
            ),
            row=(i + 1),
            col=1,
        )
        # fig.update_layout(yaxis=dict(title_text=r"$\delta$" + name +"[" + unit + "]"))
    add_3d_rms(precision.loc[data_offset:, :], r_v_a + "_" + m_t, base_names, unit)
    # fig.update_layout(plot_bgcolor="#f5f5f5", paper_bgcolor="white", legend_tracegroupgap = 180, font=dict(size=15)) # lightgray
    filename = r_v_a + "_eci_precision_" + suffix
    fig_output(fig, output_path + filename)


def get_suffix(m_t: str) -> str:
    if m_t == "m":
        return "main"
    elif m_t == "t":
        return "target"
    else:
        print("ERROR: input error!")
        return ""


# 座標系に対しても統合したい．
def plot_precision_rtn(r_v_a, m_t, data):
    if r_v_a == "r":
        col_base_name = "res_pos_"
        unit = "m"
        scale_param = 1
        output_name = "position_rtn"
        M_names_base = ["Mrr", "Mrt", "Mrn"]
        y_range = 2.5
    elif r_v_a == "v":
        col_base_name = "res_vel_"
        unit = "mm/s"
        scale_param = 1000
        output_name = "velocity_rtn"
        M_names_base = ["Mvr", "Mvt", "Mvn"]
        y_range = 5.0
    elif r_v_a == "a":
        col_names = ["ar", "at", "an"]
        unit = "um/s^2"
        scale_param = 1
        output_name = "a_rtn"
        M_names_base = ["Mar", "Mat", "Man"]
        y_range = 19.0  # ホンマはautoにしたい．
    else:
        print("false input")
        return

    frame_names = ["R", "T", "N"]
    if r_v_a != "a":
        col_names = [col_base_name + frame + "_" + m_t for frame in frame_names]

    suffix = get_suffix(m_t)

    data_for_plot = data * scale_param
    names = [r_v_a + "_" + frame for frame in frame_names]
    M_names = [M_name_base + "_" + m_t for M_name_base in M_names_base]
    fig = fig_init(data, names, unit)
    colors = ["red", "blue", "green"]

    for i in range(len(names)):
        add_accuracy(
            data_for_plot.loc[data_offset:, col_names[i]], r_v_a + "_" + m_t, col_names[i], unit
        )
        RMS = np.sqrt(np.mean(data_for_plot.loc[data_offset:, col_names[i]] ** 2))
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=data_for_plot.index * x_axis_scale,
                y=data_for_plot.loc[:, col_names[i]],
                name="$" + names[i] + ":" + "{:.3f}".format(RMS) + "(\\text{" + unit + "})$",
                legendgroup=str(i + 1),
                marker=dict(size=2, color=colors[i]),
            ),
            row=(i + 1),
            col=1,
        )
        # 1 sigmaを計算してプロットする．
        fig.add_trace(
            go.Scatter(
                x=data.index * x_axis_scale,
                y=np.sqrt(data[M_names[i]]) * scale_param,
                legendgroup=str(i),
                line=dict(width=1, color="black"),
                showlegend=False,
            ),
            row=(i + 1),
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.index * x_axis_scale,
                y=-np.sqrt(data[M_names[i]]) * scale_param,
                legendgroup=str(i),
                line=dict(width=1, color="black"),
                showlegend=False,
            ),
            row=(i + 1),
            col=1,
        )
        fig.update_yaxes(
            range=(-y_range, y_range),
            row=(i + 1),
            col=1,
        )
    add_3d_rms(data_for_plot.loc[data_offset:, :], r_v_a + "_" + m_t, col_names, unit)
    # fig.add_trace(
    #     go.Scatter(
    #         x=[],
    #         y=[],
    #         name="3D:" + "{:.3f}".format(_3drms) + "(RMS) [" + unit + "]",
    #         legendgroup=str(len(names) + 1),
    #     )
    # )
    fig_output(fig, output_path + output_name + "_precision_" + suffix)


def plot_differential_precision(
    data: pd.DataFrame(), precision_data: pd.DataFrame(), r_v_a: str, input_frame: str
) -> None:
    if r_v_a == "r":
        base_names = ["x", "y", "z"]
        axis_names = ["r_R", "r_T", "r_N"]
        unit = "cm"
        scale_param = 100
        output_name = "relative_position"
        M_names = ["Mrr", "Mrt", "Mrn"]
        y_range = 0.18 * scale_param
    elif r_v_a == "v":
        base_names = ["vx", "vy", "vz"]
        axis_names = ["v_R", "v_T", "v_N"]
        unit = "mm/s"
        scale_param = 1000
        output_name = "relative_velocity"
        M_names = ["Mvr", "Mvt", "Mvn"]
        y_range = 3.5
    elif r_v_a == "a":
        base_names = ["ar", "at", "an"]
        unit = "um/s^2"
        scale_param = 1
        axis_names = ["a_R", "a_T", "a_N"]
        output_name = "relative_a"
        M_names = ["Mar", "Mat", "Man"]
        y_range = 0.19  # 2.4
    else:
        print("input error!")
        return

    precision_data *= scale_param
    names = ["d" + axis_name for axis_name in axis_names]
    fig = fig_init(data, names, unit)
    colors = ["red", "blue", "green"]
    # eci -> rtnに変換
    if input_frame == "ECI":
        for i in range(len(precision_data)):
            DCM = DCM_from_eci_to_rtn(
                (data.loc[i, "x_m_t":"z_m_t"], data.loc[i, "vx_m_t":"vz_m_t"])
            )
            precision_data.iloc[i, :] = np.dot(DCM, precision_data.iloc[i, :])

    for i in range(len(base_names)):
        add_accuracy(precision_data.iloc[data_offset:, i], names[i], base_names[i], unit)
        RMS = np.sqrt(np.mean(precision_data.iloc[data_offset:, i] ** 2))
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=precision_data.index * x_axis_scale,
                y=precision_data.iloc[:, i],
                name="$" + names[i] + ":" + "{:.3f}".format(RMS) + "(\\text{" + unit + "})$",
                legendgroup=str(i + 1),
                marker=dict(size=2, color=colors[i]),
            ),
            row=(i + 1),
            col=1,
        )
        # 分散のところは絶対的な状態量から出しているもので正確ではないのでplotしない．
        fig.update_yaxes(
            range=(-y_range, y_range),
            row=(i + 1),
            col=1,
        )
    add_3d_rms(precision_data.loc[data_offset:, :], "d" + r_v_a, base_names, unit)
    # fig.add_trace(
    #     go.Scatter(
    #         x=[],
    #         y=[],
    #         name="3D:" + "{:.3f}".format(_3drms) + "(RMS) [" + unit + "]",
    #         legendgroup=str(len(base_names) + 1),
    #     )
    # )
    fig_output(fig, output_path + output_name + "_precision")


def plot_a(data, m_t):
    a_base_names = ["ar", "at", "an"]
    a_names = [base_name + "_" + m_t + "_e" for base_name in a_base_names]
    fig = make_subplots(rows=3, cols=1, subplot_titles=tuple(a_names))
    fig = fig_init(data, a_base_names, unit="mm/s2")
    scale_factor = 1e-6
    for i in range(len(a_base_names)):
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=data.index * x_axis_scale,
                y=data[a_names[i]] * scale_factor,
                name=a_names[i],
                marker=dict(size=2, color="red"),
            ),
            row=i + 1,
            col=1,
        )
    # fig.update_xaxes(title_text="t[s]")
    # fig.update_yaxes(title_text="acc[nm/s2]")
    fig_output(fig, output_path + "a_emp_est_" + m_t)


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
        precision[a_base_names[i]] = a_e[a_base_names[i]] - a_t[a_base_names[i]]
    # Mのデータも追加．
    for i in range(3):
        precision[M_names[i]] = data[M_names[i]] * (1e-3) ** 2  # um/s2 Mは2乗なので
        data[M_names[i]] = precision[M_names[i]]  # dataの方にもコピー
    return precision


def plot_a_eci(data, m_t):
    a_base_names = ["ax", "ay", "az"]
    a_names = [base_name + "_" + m_t for base_name in a_base_names]
    fig = make_subplots(rows=3, cols=1, subplot_titles=tuple(a_base_names))
    fig = fig_init(data, a_base_names, unit="mm/s2")
    scale_factor = 1e-6
    for i in range(3):
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=data.index * x_axis_scale,
                y=data[a_names[i]] * scale_factor,
                name=a_base_names[i],
                marker=dict(size=2, color="red"),
            ),
            row=i + 1,
            col=1,
        )
    fig_output(fig, output_path + "a_emp_eci_" + m_t)


def plot_a_dist(data, m_t, frame):
    if frame == "eci":
        a_base_names = ["ax", "ay", "az"]
    elif frame == "rtn":
        a_base_names = ["ar", "at", "an"]
    else:
        return
    a_names = [base_name + "_dist_" + m_t for base_name in a_base_names]
    fig = make_subplots(rows=3, cols=1, subplot_titles=tuple(a_base_names))
    fig = fig_init(data, a_base_names, unit="mm/s2")
    for i in range(3):
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=data.index * x_axis_scale,
                y=data[a_names[i]] * 1e3,
                name=a_base_names[i],
                marker=dict(size=2, color="red"),
            ),
            row=i + 1,
            col=1,
        )
    fig_output(fig, output_path + "a_dist_" + frame + "_" + m_t)


# これは外乱成分のみの加速度，2体問題のやつは除外されている．
def plot_a_eci_true(data_s2e_log, m_t):
    a_base_names = ["a_x", "a_y", "a_z"]
    # true
    a_t_names = [
        "sat_acc_i_i(X)[m/s^2]",
        "sat_acc_i_i(Y)[m/s^2]",
        "sat_acc_i_i(Z)[m/s^2]",
    ]  # 一旦べた書き
    a_t = pd.DataFrame()
    for i in range(3):
        a_t[a_base_names[i]] = data_s2e_log[a_t_names[i]] * 1e3
    fig = make_subplots(rows=3, cols=1, subplot_titles=tuple(a_base_names))
    fig = fig_init(data, a_base_names, unit="mm/s2")
    for i in range(3):
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=a_t.index * x_axis_scale,
                y=a_t[a_base_names[i]],
                name=a_base_names[i],
                marker=dict(size=2, color="red"),
            ),
            row=i + 1,
            col=1,
        )
    fig_output(fig, output_path + "a_true_eci_" + m_t)


def plot_a_rtn_true(data_s2e_log, m_t):
    a_base_names = ["a_R", "a_T", "a_N"]
    # true
    a_t_names = [
        "sat_acc_rtn_rtn(X)[m/s^2]",
        "sat_acc_rtn_rtn(Y)[m/s^2]",
        "sat_acc_rtn_rtn(Z)[m/s^2]",
    ]  # 一旦べた書き
    a_t = pd.DataFrame()
    for i in range(3):
        a_t[a_base_names[i]] = data_s2e_log[a_t_names[i]] * 1e3
    fig = make_subplots(rows=3, cols=1, subplot_titles=tuple(a_base_names))
    fig = fig_init(data, a_base_names, unit="mm/s2")
    for i in range(3):
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=a_t.index * x_axis_scale,
                y=a_t[a_base_names[i]],
                name=a_base_names[i],
                marker=dict(size=2, color="red"),
            ),
            row=i + 1,
            col=1,
        )
    fig_output(fig, output_path + "a_true_rtn_" + m_t)


# 今後3d用の初期化関数も用意する．ECIで書くときは地球くらい書きたい．
def plot_3d_orbit(data: pd.DataFrame()) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=data["x_m_t"],
            y=data["y_m_t"],
            z=data["z_m_t"],
            name="main",
            mode="lines",
            line=dict(width=2, color="red"),
        )
    )  # , showscale=False
    fig.add_trace(
        go.Scatter3d(
            x=data["x_t_t"],
            y=data["y_t_t"],
            z=data["z_t_t"],
            name="target",
            mode="lines",
            line=dict(width=2, color="blue"),
        )
    )  # , showscale=False
    # fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1), zaxis=dict(scaleanchor="x", scaleratio=1))
    fig.update_layout(scene=dict(aspectmode="data"))
    fig_output(fig, output_path + "orbit_3d")


# 入力データは座標変換後のものにする．
def plot_3d_relative_orbit(data: pd.DataFrame(), frame: str) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=data["x"],
            y=data["y"],
            z=data["z"],
            # name="orbit",
            mode="lines",
            line=dict(width=2, color="red"),
        )
    )  # , showscale=False
    # sc1
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            name="relative orbit",
            mode="markers",
            marker=dict(size=2, color="black"),
        )
    )
    # fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1), zaxis=dict(scaleanchor="x", scaleratio=1))
    fig.update_layout(scene=dict(aspectmode="data"))
    fig_output(fig, output_path + "relative_orbit_3d_" + frame)


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


def cdt_plot(data, output_name):
    names = ["cdt_main", "cdt_target"]
    suffix = ["m", "t"]
    fig = make_subplots(rows=2, cols=1, subplot_titles=(tuple(names)))
    fig = fig_init(data, names, unit="m")
    for i in range(len(names)):
        data_true_key = "t_" + suffix[i] + "_t"
        data_est_key = "t_" + suffix[i] + "_e"
        fig.add_trace(
            go.Scatter(
                x=data.index * x_axis_scale,
                y=data[data_true_key],
                name=names[i] + "_t",
                line=dict(width=2, color="black"),
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.index * x_axis_scale,
                y=data[data_est_key],
                name=names[i] + "_e",
                line=dict(width=2, color="red"),
            ),
            row=i + 1,
            col=1,
        )
    fig_output(fig, output_name)


def calc_cdt_sparse_precision(data):
    data_sparse = data[data.index % 10 == 9]
    t_names = ["cdt_main", "cdt_target"]
    suffix = ["m", "t"]
    for i in range(len(t_names)):
        true_name_key = "t_" + suffix[i] + "_t"
        est_name_key = "t_" + suffix[i] + "_e"
        data_sparse.loc[:, t_names[i]] = (
            data_sparse.loc[:, est_name_key] - data_sparse.loc[:, true_name_key]
        )
    return data_sparse


cdt_precision = calc_cdt_sparse_precision(data)
t_names = ["cdt_main", "cdt_target"]
axis_names = ["c\delta t_{main}", "c\delta t_{target}"]
suffix = ["m", "t"]
fig = fig_init(data, axis_names, unit="m")
for i in range(len(t_names)):
    fig.add_trace(
        go.Scatter(
            x=cdt_precision.index * x_axis_scale,
            y=(cdt_precision[t_names[i]]),
            name=t_names[i],
            mode="markers",
            marker=dict(size=2, color="red"),
        ),
        row=i + 1,
        col=1,
    )
    M_name = "Mt_" + suffix[i]
    fig.add_trace(
        go.Scatter(
            x=cdt_precision.index * x_axis_scale,
            y=np.sqrt(cdt_precision[M_name]),
            line=dict(width=1, color="black"),
            name="$1 \sigma$",
        ),
        row=i + 1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cdt_precision.index * x_axis_scale,
            y=-np.sqrt(cdt_precision[M_name]),
            line=dict(width=1, color="black"),
            showlegend=False,
        ),
        row=i + 1,
        col=1,
    )
fig_output(fig, output_path + "cdt_sparse_precision")

cdt_precision.loc[:, "dcdt"] = cdt_precision.loc[:, t_names[1]] - cdt_precision.loc[:, t_names[0]]
fig = fig_init(cdt_precision, ["\Delta c\delta t"], unit="m")
fig.add_trace(
    go.Scatter(
        x=cdt_precision.index * x_axis_scale,
        y=(cdt_precision["dcdt"]),
        mode="markers",
        marker=dict(size=2, color="red"),
    ),
)
fig_output(fig, output_path + "dcdt_sparse_precision")


def calc_N_precision(data, m_t):
    precision = pd.DataFrame()
    for i in range(GNSS_CH_NUM):
        t_col_name = "N" + str(i + 1) + "_" + m_t + "_t"
        e_col_name = "N" + str(i + 1) + "_" + m_t + "_e"
        precision[i + 1] = data[e_col_name] - data[t_col_name]
    return precision


def plot_N_precision(data, m_t):
    fig = fig_init(data, ["N"], "cycle")
    precision = calc_N_precision(data, m_t)
    for i in range(GNSS_CH_NUM):
        fig.add_trace(
            go.Scatter(x=data.index * x_axis_scale, y=precision[i + 1], name="N" + str(i + 1))
        )

    suffix = get_suffix(m_t)
    fig_output(fig, output_path + "N_precision_" + suffix)


def plot_dN_precision(data: pd.DataFrame):
    precision_m = calc_N_precision(data, "m")
    precision_t = calc_N_precision(data, "t")
    precision = pd.DataFrame(columns=[i + 1 for i in range(GNSS_CH_NUM)])
    for row in data.itertuples():
        # このやり方しかないんか？コスト高すぎしんどい．
        row_array = np.zeros(GNSS_CH_NUM)
        for i in range(GNSS_CH_NUM):
            id_m = row._asdict()["id_ch" + str(i + 1) + "_m"]
            for j in range(GNSS_CH_NUM):
                if row._asdict()["id_ch" + str(j + 1) + "_t"] == id_m:
                    row_array[i] = (
                        precision_m.at[row.Index, i + 1] - precision_t.at[row.Index, j + 1]
                    )
                    break
        precision.loc[row.Index] = row_array
    names = ["dN"]
    unit = "cycle"
    fig = fig_init(data, names, unit)
    for i in range(GNSS_CH_NUM):
        fig.add_trace(
            go.Scatter(x=data.index * x_axis_scale, y=precision[i + 1], name="dN" + str(i + 1))
        )

    fig_output(fig, output_path + "dN_precision")


def plot_N_fix_flag(data, m_t):
    precision = pd.DataFrame()
    names = ["is fixed"]
    unit = "True/False"
    fig = fig_init(data, names, unit)
    # fig = go.Figure()
    for i in range(GNSS_CH_NUM):
        col_name = "N" + str(i + 1) + "_" + m_t
        fig.add_trace(
            go.Scatter(x=data.index * x_axis_scale, y=data[col_name], name="N" + str(i + 1))
        )

    suffix = get_suffix(m_t)

    fig_output(fig, output_path + "N_is_fixed_" + suffix)


# main, targetまとめてよさそう？
def N_plot(m_t: str, t_e: str) -> None:
    fig = fig_init(data, ["N"], "cycle")
    for i in range(GNSS_CH_NUM):
        t_col_name = "N" + str(i + 1) + "_" + m_t + "_" + t_e
        fig.add_trace(
            go.Scatter(x=data.index * x_axis_scale, y=data[t_col_name], name="N" + str(i + 1))
        )

    if t_e == "t":
        suffix1 = "true"
    else:
        suffix1 = "est"
    suffix2 = get_suffix(m_t)

    fig_output(fig, output_path + "N_" + suffix1 + "_" + suffix2)


def plot_QM_N(data, Q_M, m_t):
    fig = fig_init(data, Q_M, cycle)
    out_fname_base = Q_M + "N"
    for i in range(GNSS_CH_NUM):
        col_name = out_fname_base + str(i + 1) + "_" + m_t
        fig.add_trace(
            go.Scatter(
                x=data.index * x_axis_scale,
                y=np.sqrt(data[col_name]),
                name="$" + Q_M + "_{N" + str(i + 1) + "}$",
            )
        )

    suffix = get_suffix(m_t)

    fig_output(fig, output_path + out_fname_base + "_" + suffix)


def plot_Ma(data, m_t):
    Ma_base_names = ["Mar", "Mat", "Man"]
    Ma_names = [base_name + "_" + m_t for base_name in Ma_base_names]
    # fig = make_subplots(rows=3, cols=1, subplot_titles=tuple(Ma_base_names))
    fig = fig_init(data, Ma_base_names, unit="nm/s2")
    for i in range(3):
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=data.index * x_axis_scale,
                y=np.sqrt(data[Ma_names[i]]),
                name=Ma_base_names[i],
                marker=dict(size=2, color="red"),
            ),
            row=i + 1,
            col=1,
        )
    fig_output(fig, output_path + "Ma_emp_" + get_suffix(m_t))


def plot_visible_gnss_sat(data: pd.DataFrame()):
    names = ["main", "target", "common"]
    fig = fig_init(data, names, unit="")
    for i in range(3):
        fig.add_trace(
            go.Scatter(
                mode="lines",
                x=data.index * x_axis_scale,
                y=data["sat_num_" + names[i]],
                name=names[i],
            ),
            row=i + 1,
            col=1,
        )
    fig_output(fig, output_path + "visible_gnss_sat")


def plot_gnss_id(data: pd.DataFrame(), m_t: str) -> None:
    fig = fig_init(data, ["gnss sat id"], "")
    for i in range(GNSS_CH_NUM):
        ch_col_name = "id_ch" + str(i + 1) + "_" + m_t
        fig.add_trace(
            go.Scatter(x=data.index * x_axis_scale, y=data[ch_col_name], name="ch" + str(i + 1))
        )
    suffix = get_suffix(m_t)
    fig_output(fig, output_path + "gnss_sat_id_" + suffix)


def plot_Q(data, rvat, m_t):
    if rvat == "r":
        base_names = ["Qx", "Qy", "Qz"]
        file_name_base = "Q_r"
        unit = "m"
        suffix = get_suffix(m_t)
    elif rvat == "v":
        base_names = ["Qvx", "Qvy", "Qvz"]
        file_name_base = "Q_v"
        unit = "m/s"
        suffix = get_suffix(m_t)
    elif rvat == "a":
        base_names = ["Qar", "Qat", "Qan"]
        file_name_base = "Q_acc"
        unit = "nm/s"
        suffix = get_suffix(m_t)
    elif rvat == "t":
        base_names = ["Qt_m", "Qt_t"]
        file_name_base = "Q_cdt"
        unit = "m"
        suffix = ""
        data_name = base_names

    fig = fig_init(data, base_names, unit=unit)

    if rvat != "t":
        data_name = [base + "_" + m_t for base in base_names]

    for i, name in enumerate(data_name):
        fig.add_trace(
            go.Scatter(x=data.index * x_axis_scale, y=np.sqrt(data[name]), name=base_names[i]),
            row=i + 1,
            col=1,
        )
    # fig.update_xaxes(title_text="t[s]")
    # fig.update_yaxes(title_text="Q[" + unit + "]")
    fig_output(fig, output_path + file_name_base + "_" + suffix)


def plot_R(data: pd.DataFrame(), observable: str, m_t: str) -> None:
    if observable == "GRAPHIC":
        R_name = "Rgr"
        suffix = get_suffix(m_t)
        col_suffix = "_" + m_t
    elif observable == "SDCP":
        R_name = "Rcp"
        suffix = ""
        col_suffix = ""

    fig = fig_init(data, ["R"], "m")
    for i in range(GNSS_CH_NUM):
        col_name = R_name + str(i + 1) + col_suffix
        fig.add_trace(
            go.Scatter(
                x=data.index * x_axis_scale,
                y=np.sqrt(data[col_name]),
                name="R " + observable + str(i + 1),
            )
        )

    fig_output(fig, output_path + "R_" + observable + "_" + suffix)


# 受信点の重心からのずれをプロットする．
def plot_receive_position(data: pd.DataFrame, data_s2e: pd.DataFrame, m_t: str) -> None:
    pos_base_name = ["x", "y", "z"]
    pos_names = [base + "_" + m_t + "_t" for base in pos_base_name]
    arp_names = ["gnss_arp_true_eci(X)[m]", "gnss_arp_true_eci(Y)[m]", "gnss_arp_true_eci(Z)[m]"]

    fig = fig_init(data, pos_base_name, "m")
    for i in range(3):
        diff = data_s2e.loc[:, arp_names[i]] - data.loc[:, pos_names[i]]
        fig.add_trace(
            go.Scatter(x=data.index * x_axis_scale, y=diff, name=pos_base_name[i]),
            row=i + 1,
            col=1,
        )

    fig_output(fig, output_path + "arp_diff_" + get_suffix(m_t))


def plot_determined_position_precision(
    data: pd.DataFrame, data_s2e: pd.DataFrame, m_t: str
) -> None:
    pos_base_name = ["x", "y", "z"]
    pos_names = [base + "_" + m_t + "_t" for base in pos_base_name]
    s2e_pos_names = ["sat_position_i(X)[m]", "sat_position_i(Y)[m]", "sat_position_i(Z)[m]"]
    gnss_pos_names = [
        "gnss_position_eci(X)[m]",
        "gnss_position_eci(Y)[m]",
        "gnss_position_eci(Z)[m]",
    ]

    fig = fig_init(data, pos_base_name, "m")
    for i in range(3):
        diff = data_s2e.loc[:, gnss_pos_names[i]] - data_s2e.loc[:, s2e_pos_names[i]]
        # diff = data_s2e.loc[:,gnss_pos_names[i]] - data.loc[:,pos_names[i]]
        fig.add_trace(
            go.Scatter(x=data.index * x_axis_scale, y=diff, name=pos_base_name[i]),
            row=i + 1,
            col=1,
        )
    fig_output(fig, output_path + "navigated_pos_diff_" + get_suffix(m_t))


def plot_gnss_direction(data, m_t):
    fig = px.scatter_polar(range_theta=[0, 360], start_angle=0, direction="counterclockwise")

    for i in range(GNSS_CH_NUM):
        azi_name = "azi" + str(i + 1) + "_" + m_t
        ele_name = "ele" + str(i + 1) + "_" + m_t
        fig.add_trace(
            go.Scatterpolar(
                r=90 - data[ele_name],
                theta=data[azi_name],
                mode="markers",
                name="ch " + str(i + 1),
                marker=dict(color="blue", size=1),
                showlegend=False,
                # dr=15,
            ),
        )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(dtick=15),
        ),
        margin=dict(t=1, b=1, l=1, r=1),
    )
    suffix = get_suffix(m_t)
    fig_output(fig, output_path + "gnss_observed_direction_" + suffix)


# バグってるのでまた今度実施する．
def plot_gnss_direction_animation(data: pd.DataFrame(), m_t: str) -> None:
    fig = px.scatter_polar(range_theta=[0, 360], start_angle=0, direction="counterclockwise")

    data["step"] = data.index
    # for i in range(GNSS_CH_NUM):
    i = 0
    azi_name = "azi" + str(i + 1) + "_" + m_t
    ele_name = "ele" + str(i + 1) + "_" + m_t
    zenith_name = "zenith" + str(i + 1) + "_" + m_t
    data[zenith_name] = 90 - data[ele_name]
    fig.add_trace(
        # go.Scatterpolar(
        px.scatter_polar(
            data_frame=data,
            r=zenith_name,
            theta=azi_name,
            # mode="markers",
            # name="ch " + str(i + 1),
            # hover_data=data.loc[:, azi_name:ele_name],
            animation_frame="step",
            animation_group=zenith_name,
        ),
    )
    fig_output(fig, output_path + "gnss_observed_direction_animation_" + get_suffix(m_t))


def plot_pco(data: pd.DataFrame, m_t: str) -> None:
    pco_base_name = ["pco_x", "pco_y", "pco_z"]
    fig = fig_init(data, pco_base_name, "mm")
    col_names = [base + "_" + m_t for base in pco_base_name]
    for i in range(len(pco_base_name)):
        fig.add_trace(
            go.Scatter(x=data.index * x_axis_scale, y=data[col_names[i]], name=pco_base_name[i]),
            row=i + 1,
            col=1,
        )
    suffix = get_suffix(m_t)
    fig_output(fig, output_path + "pco_" + suffix)


def plot_pcv_grid(pcc_path, out_fname):
    pcv_df = pd.read_csv(pcc_path, skiprows=1, header=None)
    fig = px.scatter_polar(
        range_theta=[0, 360],
        start_angle=0,
        direction="counterclockwise",
    )
    # fig = go.Figure()
    num_azi, num_ele = pcv_df.shape
    azi_increment = 360 / (num_azi - 1)
    ele_increment = 90 / (num_ele - 1)

    azi = np.arange(0, 365, azi_increment)
    ele = np.arange(0, 95, ele_increment)
    # azi, ele = np.mgrid[0:365:azi_increment, 90:-5:-ele_increment]
    # azi, ele = np.mgrid[0:365:azi_increment, 0:95:ele_increment]
    azi, ele = np.meshgrid(azi, ele)

    # azi, ele = np.mgrid[0:365:azi_increment, 90:-5:-ele_increment]
    # ele, azi = np.mgrid[90:-5:-ele_increment, 0:365:azi_increment]
    # azi, ele = np.mgrid[0:365:azi_increment, 0:95:ele_increment]  # mgridですると等分にならないのか？
    # color = np.fliplr(pcv_df.values)
    # color = pcv_df.values
    color = pcv_df.values.T
    azi = azi.ravel()  # radにする必要はなさそう．
    ele = ele.ravel()
    color = color.ravel()

    fig.add_trace(
        # px.bar_polar(
        go.Barpolar(
            r=ele,
            theta=azi,
            marker=dict(
                color=color,
                colorscale="portland",  # viridis
                colorbar_thickness=24,
                line_width=0,
                cmin=-5.0,
                cmax=9.0,
            ),
            theta0=0,
            # dr=5,
            # customdata=color,
            hovertext=color,
            hoverinfo="r+text",
        )
    )
    fig.update_layout(
        # template=None,
        polar=dict(
            bargap=0,
            radialaxis=dict(
                # tick0=0,
                # dtick="L5.0",
                ticks="",
                # tickmode="linear",
                showticklabels=False,
                showgrid=False,
                showline=False,
                gridwidth=0,
                linewidth=0,
                rangemode="normal",
                # ticklabelstep=10,
                # range=[90, 0],
            ),  # type="linear",
            angularaxis=dict(
                showgrid=False,
                showline=False,
                ticks="",
                # tickmode="linear",
            ),
            hole=0,
            gridshape="linear",  # circular
        ),
        font=dict(size=20),
        hovermode="closest",
    )

    fig_output(fig, output_path + out_fname)


def plot_phase_center_distribution(pc_df, out_fname, crange, norm=None, cmap=cm.jet):
    num_azi, num_ele = pc_df.shape
    azi_increment = 360 / (num_azi - 1)
    ele_increment = 90 / (num_ele - 1)
    azi = np.deg2rad(np.arange(0, 365, azi_increment))
    # ele = np.arange(0, 90, ele_increment)
    ele = np.linspace(0, 5 * (num_ele - 1), num_ele)  # 5°ずつであると仮定
    # azi, ele = np.mgrid[0:365:azi_increment, 90:-5:-ele_increment]
    # azi, ele = np.mgrid[0:365:azi_increment, 0:95:ele_increment]
    azi, ele = np.meshgrid(azi, ele)

    fig = plt.figure()
    # ax = Axes3D(fig)
    z = pc_df.values.T
    # z = z[:-1, :]
    # z = np.fliplr(pcv_df.values)

    # ax = plt.subplot(projection="polar")
    ax = plt.subplot(polar=True)
    cmin, cmax = crange
    if not norm:
        norm = Normalize(vmin=cmin, vmax=cmax)
    pcolor = ax.pcolormesh(azi, ele, z, cmap=cmap, norm=norm)
    # pcolor = ax.pcolormesh(azi, ele, z, cmap=cm.jet, norm=colors.TwoSlopeNorm(0))
    c_label = np.linspace(cmin, cmax, 7)
    # ticksを指定すると0が中心でなくなるので微妙かもしれない．
    # colb = fig.colorbar(pcolor, ax=ax, ticks=np.append(c_label, cmax), orientation="vertical")
    colb = fig.colorbar(pcolor, ax=ax, orientation="vertical")

    # plt.grid()
    # colb.set_label("label", fontname="Arial", fontsize=20)

    plt.savefig(output_path + out_fname + ".jpg", dpi=900, bbox_inches="tight", pad_inches=0)
    plt.savefig(output_path + out_fname + ".eps", bbox_inches="tight", pad_inches=0)
    # plt.savefig(output_path + out_fname + ".pdf", bbox_inches="tight", pad_inches=0)
    # plt.show()


def plot_pcv_by_matplotlib(pcc_path, out_fname, crange=(-5.0, 9.0)):
    pcv_df = pd.read_csv(pcc_path, skiprows=1, header=None)
    # pcv_df = pcv_df.iloc[:, :-1]  # 90 - 5 degまで <- trueは全plotした方がいいな．
    plot_phase_center_distribution(pcv_df, out_fname, crange)


def plot_pc_accuracy_by_matplotlib(est_fname, true_fname, out_fname, crange=(-5, 5)) -> None:
    est_pc_df = pd.read_csv(est_fname, skiprows=1, header=None)
    true_pc_df = pd.read_csv(true_fname, skiprows=1, header=None)
    pc_df = est_pc_df - true_pc_df
    # pc_df = pc_df.iloc[:, :-1]  # 90 - 5 degまで

    # for two slope plot ++++++++++++++++++++++++++++++++++++
    # crange = (pc_df.min().min(), pc_df.max().max())
    # plot_phase_center_distribution(pc_df, out_fname, crange, colors.TwoSlopeNorm(0), cm.bwr)

    # for fix range +++++++++++++++++++++++++++++++++++++++++
    plot_phase_center_distribution(
        pc_df, out_fname, crange, Normalize(vmin=crange[0], vmax=crange[1]), cm.bwr
    )


def plot_sdcp_residual_by_matplotlib(fname, out_fname, crange=(-10, 10)) -> None:
    res_df = pd.read_csv(fname, header=None)

    # for fix range +++++++++++++++++++++++++++++++++++++++++
    plot_phase_center_distribution(
        res_df, out_fname, crange, Normalize(vmin=crange[0], vmax=crange[1]), cm.bwr
    )


def plot_residual_by_elevation(data: pd.DataFrame()) -> None:
    (
        global_font_size,
        legend_font_size,
        axis_title_font_size,
        width,
        height,
        v_space,
        h_space,
        x_pos,
    ) = set_fig_params()

    fig = make_subplots()  # subplot_titles=tuple(base_names)
    fig.update_layout(
        plot_bgcolor="white",
        font=dict(size=global_font_size, family="Times New Roman"),  # globalなfont設定
        width=700,
        height=400,
        legend=dict(
            x=x_pos,
            y=0.99,
            xanchor="left",
            yanchor="top",
            font=dict(size=legend_font_size),
            bordercolor="black",
            borderwidth=1,
            orientation="h",
            itemsizing="constant",
        ),
        margin=dict(t=1, b=1, l=1, r=1),
    )
    fig.update_xaxes(
        linecolor="black",
        gridcolor="silver",
        mirror=True,
        range=(90, 0),
        title=dict(text="$\\text{elevation}[^\\circ]$"),
        titlefont_size=axis_title_font_size,
    )
    fig.update_yaxes(
        linecolor="black",
        mirror=True,
        zeroline=True,
        zerolinecolor="silver",
        zerolinewidth=1,
        title=dict(
            text="residual [mm]",
            standoff=2,
        ),
        titlefont_size=axis_title_font_size,
    )
    angles = np.arange(90, -5, -5)
    # 平均だと微妙なのでRMSで評価する？
    mean = data[data != 0].mean(axis=0)  # 完全に0の項は観測がなかった部分なので計算から除外．
    mean.iat[-1] = 0
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=angles,
            y=mean,
            name="$\mu$",
            legendgroup=str(1),
            marker=dict(size=5, color="red"),
        ),
    )
    # 1 sigmaを計算してプロットする．
    sigma = data[data != 0].std(axis=0)
    sigma.iat[-1] = 0
    fig.add_trace(
        go.Scatter(
            mode="lines",
            x=angles,
            y=sigma,
            name="$\sigma$",
            legendgroup=str(2),
            line=dict(width=1, color="black"),
            # showlegend=False,
        ),
    )
    fig.add_trace(
        go.Scatter(
            mode="lines",
            x=angles,
            y=-sigma,  # name="$\sigma$",
            legendgroup=str(2),
            line=dict(width=1, color="black"),
            showlegend=False,
        ),
    )

    fig_output(fig, output_path + "sdcp_residual_by_elevation")


# # plot
# plot_3d_orbit(data)
# ここでの計算はECI
baseline = calc_relinfo("r", "t", data)
relative_v = calc_relinfo("v", "t", data)
# plot_3d_relative_orbit(baseline, "eci")
plot_separation_eci(baseline)
# 主衛星固定のlvlhに変換
omega_chief_lvlh = np.sqrt(mu0 / a**3)  # これは平均運動なので実際の角速度とは少し違う．
omega_lvlh = np.array([0, -omega_chief_lvlh, 0])
for i in range(len(baseline)):
    DCM = DCM_from_eci_to_lvlh((data.loc[i, "x_m_t":"z_m_t"], data.loc[i, "vx_m_t":"vz_m_t"]))
    baseline.iloc[i, :] = np.dot(DCM, baseline.iloc[i, :])
    rotation_element = np.cross(omega_lvlh, baseline.iloc[i, :])
    relative_v.iloc[i, :] = (
        np.dot(DCM, relative_v.iloc[i, :]) - rotation_element
    )  # ここの座標変換の式はまとめておく．
plot_3d_relative_orbit(baseline, "lvlh")

# plot_precision("r", "m", data)
# plot_precision("r", "t", data)
# plot_precision("v", "m", data)
# plot_precision("v", "t", data)
plot_precision_rtn("r", "m", data)
plot_precision_rtn("r", "t", data)
plot_precision_rtn("v", "m", data)
plot_precision_rtn("v", "t", data)
pbd_precision = calc_relinfo("r", "e", data) - calc_relinfo("r", "t", data)
plot_differential_precision(data, pbd_precision, "r", "ECI")
dv_precision = calc_relinfo("v", "e", data) - calc_relinfo("v", "t", data)
plot_differential_precision(data, dv_precision, "v", "ECI")

plot_rel_state("r", baseline)
plot_rel_state("v", relative_v)

if REDUCE_DYNAMIC:
    a_m_precision = calc_a_precision(data, data_s2e_csv, "m", "rtn")
    a_t_precision = calc_a_precision(data, data_s2e_csv, "t", "rtn")
    plot_precision_rtn("a", "m", a_m_precision)
    plot_precision_rtn("a", "t", a_t_precision)
    plot_differential_precision(
        data, a_t_precision.iloc[:, 0:3] - a_m_precision.iloc[:, 0:3], "a", "RTN"
    )


# plot_a(data, "m")
# plot_a(data, "t")
# plot_a_precision(data, data_s2e_csv, "m", "eci")
# plot_a_precision(data, data_s2e_csv, "t", "eci")
# plot_a_eci(data, "m")
# plot_a_eci(data, "t")
# plot_a_dist(data, "m", "eci")
# plot_a_dist(data, "t", "eci")
# plot_a_dist(data, "m", "rtn")
# plot_a_dist(data, "t", "rtn")
# plot_a_eci_true(data_s2e_csv, "m")
plot_a_rtn_true(data_s2e_csv, "m")

# cdt_plot(data, output_path + "cdt")
# cdt_plot(data[data.index % 10 == 9], output_path + "cdt_sparse")

plot_N_precision(data, "m")
plot_N_precision(data, "t")
# plot_dN_precision(data)  # 計算コスト高いので必要な時だけにする．
plot_N_fix_flag(data, "m")
# plot_N_fix_flag(data, "t")
# N_plot("m", "t")
# N_plot("t", "t")
# N_plot("m", "e")
# N_plot("t", "e")

# plot_R(data, "GRAPHIC", "m")
# plot_R(data, "GRAPHIC", "t")
# plot_R(data, "SDCP", "")

# plot_QM_N(data, "Q", "m")
# plot_QM_N(data, "Q", "t")
# plot_QM_N(data, "M", "m")
# plot_QM_N(data, "M", "t")
plot_Q(data, "r", "m")
# plot_Q(data, "r", "t")
plot_Q(data, "v", "m")
# plot_Q(data, "v", "t")
plot_Q(data, "t", "")
if REDUCE_DYNAMIC:
    plot_Q(data, "a", "m")
    # plot_Q(data, "a", "t")
    plot_Ma(data, "m")
    # plot_Ma(data, "t")
# plot_receive_position(data, data_s2e_csv, "m")
# plot_determined_position_precision(data, data_s2e_csv, "m")
plot_gnss_direction(data, "m")
plot_gnss_direction(data, "t")
# plot_gnss_direction_animation(data, "m")
plot_visible_gnss_sat(data)
plot_gnss_id(data, "m")
plot_gnss_id(data, "t")
# plot_pco(data, "m")
plot_pco(data, "t")

if PCV:
    # plot_pcv_grid(pcc_log_path, "pcv_true")
    # plot_pcv_grid(s2e_debug + "target_antenna_pcv.csv", "estimated_target_pcv")

    plot_pcv_by_matplotlib(pcv_log_path, "pcv_true")
    plot_pcv_by_matplotlib(pcc_log_path, "pcc_true", (-128, 10))
    plot_pcv_by_matplotlib(s2e_log_dir + "target_antenna_pcv.csv", "estimated_target_pcv")
    # plot_pcv_by_matplotlib(s2e_log_dir + "_pcv.csv", "estimated_target_pcv")
    plot_pcv_by_matplotlib(
        s2e_log_dir + "target_antenna_pcc.csv",
        "estimated_target_pcc",
        (-128, 10)
        # s2e_log_dir + "_pcc.csv",
        # "estimated_target_pcc",
        # (-128, 10),
    )
    plot_pc_accuracy_by_matplotlib(
        s2e_log_dir + "target_antenna_pcv.csv",
        pcv_log_path,
        "target_pcv_accuracy"
        # s2e_log_dir + "_pcv.csv",
        # pcv_log_path,
        # "target_pcv_accuracy",
    )
    plot_pc_accuracy_by_matplotlib(
        s2e_log_dir + "target_antenna_pcc.csv",
        pcc_log_path,
        "target_pcc_accuracy",
        (-10, 10)
        # s2e_log_dir + "_pcc.csv",
        # pcc_log_path,
        # "target_pcc_accuracy",
        # (-10, 10),
    )

plot_sdcp_residual_by_matplotlib(sdcp_residual_log_path, "sdcp_residual")
res_data = pd.read_csv(sdcp_residual_log_path, header=None)
plot_residual_by_elevation(res_data)

accuracy_log.to_csv(accuracy_file, sep=",", index=False)
# 最後に全グラフをまとめてコピー
# shutil.move(output_path, copy_dir + "figure/")
shutil.move(output_path, copy_dir)
