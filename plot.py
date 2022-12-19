import datetime
import os
import shutil
from asyncio import constants
from glob import glob

import matplotlib.cm as cm
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


def get_latest_modified_file_path(dirname):
    target = os.path.join(dirname, "*")
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    latest_modified_file_path = sorted(files, key=lambda files: files[1])[-1]
    return latest_modified_file_path[0]


s2e_dir = "../../s2e_pbd/"
s2e_log_path_base = s2e_dir + "data/logs"
s2e_debug = s2e_dir + "CMakeBuilds/Debug/"
s2e_log_dir = get_latest_modified_file_path(s2e_log_path_base)
s2e_csv_log_path = (
    s2e_log_dir + "/" + [file for file in os.listdir(s2e_log_dir) if file.endswith(".csv")][0]
)
# ここは適宜修正する．
# copy_base_dir = "/mnt/g/マイドライブ/Documents/University/lab/Research/FFGNSS/plot_result/"
copy_base_dir = "G:/マイドライブ/Documents/University/lab/Research/FFGNSS/plot_result/"
sim_config_path = s2e_debug + "readme_new.txt"
log_path = s2e_debug + "result_new.csv"
pcc_log_path = s2e_debug + "phase_center_correction.csv"

# コピーしたログファイルからグラフ描くとき===========
# log_base = copy_base_dir + "20221215_101826"  # ここを修正
# sim_config_path = log_base + "/readme.txt"
# log_path = log_base + "/result.csv"
# s2e_log_dir = log_base + "/s2e_logs"
# s2e_csv_log_path = (
#     s2e_log_dir + "/" + [file for file in os.listdir(s2e_log_dir) if file.endswith(".csv")][0]
# )
# ================================================

# 実行時にメモ残すようにするといいかも

output_path = "figure/"
os.makedirs(output_path, exist_ok=True)
dt_now = datetime.datetime.now()
copy_dir = copy_base_dir + dt_now.strftime("%Y%m%d_%H%M%S")
os.mkdir(copy_dir)
# os.mkdir(copy_dir + '/figure')
shutil.copyfile(sim_config_path, copy_dir + "/readme.txt")
shutil.copytree(get_latest_modified_file_path(s2e_log_path_base), copy_dir + "/s2e_logs")
# 独自csvをすべてコピー．
for file in glob(s2e_debug + "*.csv"):
    shutil.copy(file, copy_dir)

data = pd.read_csv(log_path, header=None)

REDUCE_DYNAMIC = 1

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
    "res_pos_r_m",
    "res_pos_t_m",
    "res_pos_n_m",
    "res_vel_r_m",
    "res_vel_t_m",
    "res_vel_n_m",
    "res_pos_r_t",
    "res_pos_t_t",
    "res_pos_n_t",
    "res_vel_r_t",
    "res_vel_t_t",
    "res_vel_n_t",
]
col_ambiguity = [
    "N1_m_t",
    "N2_m_t",
    "N3_m_t",
    "N4_m_t",
    "N5_m_t",
    "N6_m_t",
    "N7_m_t",
    "N8_m_t",
    "N9_m_t",
    "N10_m_t",
    "N11_m_t",
    "N12_m_t",
    "N1_m_e",
    "N2_m_e",
    "N3_m_e",
    "N4_m_e",
    "N5_m_e",
    "N6_m_e",
    "N7_m_e",
    "N8_m_e",
    "N9_m_e",
    "N10_m_e",
    "N11_m_e",
    "N12_m_e",
    "N1_t_t",
    "N2_t_t",
    "N3_t_t",
    "N4_t_t",
    "N5_t_t",
    "N6_t_t",
    "N7_t_t",
    "N8_t_t",
    "N9_t_t",
    "N10_t_t",
    "N11_t_t",
    "N12_t_t",
    "N1_t_e",
    "N2_t_e",
    "N3_t_e",
    "N4_t_e",
    "N5_t_e",
    "N6_t_e",
    "N7_t_e",
    "N8_t_e",
    "N9_t_e",
    "N10_t_e",
    "N11_t_e",
    "N12_t_e",
]
col_M = [
    "Mx_m",
    "My_m",
    "Mz_m",
    "Mt_m",
    "Mvx_m",
    "Mvy_m",
    "Mvz_m",
    "Mar_m",
    "Mat_m",
    "Man_m",
    "MN1_m",
    "MN2_m",
    "MN3_m",
    "MN4_m",
    "MN5_m",
    "MN6_m",
    "MN7_m",
    "MN8_m",
    "MN9_m",
    "MN10_m",
    "MN11_m",
    "MN12_m",
    "Mx_t",
    "My_t",
    "Mz_t",
    "Mt_t",
    "Mvx_t",
    "Mvy_t",
    "Mvz_t",
    "Mar_t",
    "Mat_t",
    "Man_t",
    "MN1_t",
    "MN2_t",
    "MN3_t",
    "MN4_t",
    "MN5_t",
    "MN6_t",
    "MN7_t",
    "MN8_t",
    "MN9_t",
    "MN10_t",
    "MN11_t",
    "MN12_t",
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
col_gnss_dir = [
    "azi1_m",
    "azi2_m",
    "azi3_m",
    "azi4_m",
    "azi5_m",
    "azi6_m",
    "azi7_m",
    "azi8_m",
    "azi9_m",
    "azi10_m",
    "azi11_m",
    "azi12_m",
    "ele1_m",
    "ele2_m",
    "ele3_m",
    "ele4_m",
    "ele5_m",
    "ele6_m",
    "ele7_m",
    "ele8_m",
    "ele9_m",
    "ele10_m",
    "ele11_m",
    "ele12_m",
    "azi1_t",
    "azi2_t",
    "azi3_t",
    "azi4_t",
    "azi5_t",
    "azi6_t",
    "azi7_t",
    "azi8_t",
    "azi9_t",
    "azi10_t",
    "azi11_t",
    "azi12_t",
    "ele1_t",
    "ele2_t",
    "ele3_t",
    "ele4_t",
    "ele5_t",
    "ele6_t",
    "ele7_t",
    "ele8_t",
    "ele9_t",
    "ele10_t",
    "ele11_t",
    "ele12_t",
]
col_pco = ["pco_x_m", "pco_y_m", "pco_z_m", "pco_x_t", "pco_y_t", "pco_z_t"]
col_ambiguity_flag = [
    "N1_m",
    "N2_m",
    "N3_m",
    "N4_m",
    "N5_m",
    "N6_m",
    "N7_m",
    "N8_m",
    "N9_m",
    "N10_m",
    "N11_m",
    "N12_m",
    "N1_t",
    "N2_t",
    "N3_t",
    "N4_t",
    "N5_t",
    "N6_t",
    "N7_t",
    "N8_t",
    "N9_t",
    "N10_t",
    "N11_t",
    "N12_t",
]
data_col = (
    col_true_info
    + col_est_info
    + col_residual
    + col_ambiguity
    + col_M
    + [
        "sat_num_main",
        "sat_num_target",
        "sat_num_common",
        "id_ch1_m",
        "id_ch2_m",
        "id_ch3_m",
        "id_ch4_m",
        "id_ch5_m",
        "id_ch6_m",
        "id_ch7_m",
        "id_ch8_m",
        "id_ch9_m",
        "id_ch10_m",
        "id_ch11_m",
        "id_ch12_m",
        "id_ch1_t",
        "id_ch2_t",
        "id_ch3_t",
        "id_ch4_t",
        "id_ch5_t",
        "id_ch6_t",
        "id_ch7_t",
        "id_ch8_t",
        "id_ch9_t",
        "id_ch10_t",
        "id_ch11_t",
        "id_ch12_t",
        "Qx_m",
        "Qy_m",
        "Qz_m",
        "Qt_m",
        "Qvx_m",
        "Qvy_m",
        "Qvz_m",
        "Qar_m",
        "Qat_m",
        "Qan_m",
        "QN1_m",
        "QN2_m",
        "QN3_m",
        "QN4_m",
        "QN5_m",
        "QN6_m",
        "QN7_m",
        "QN8_m",
        "QN9_m",
        "QN10_m",
        "QN11_m",
        "QN12_m",
        "Qx_t",
        "Qy_t",
        "Qz_t",
        "Qt_t",
        "Qvx_t",
        "Qvy_t",
        "Qvz_t",
        "Qar_t",
        "Qat_t",
        "Qan_t",
        "QN1_t",
        "QN2_t",
        "QN3_t",
        "QN4_t",
        "QN5_t",
        "QN6_t",
        "QN7_t",
        "QN8_t",
        "QN9_t",
        "QN10_t",
        "QN11_t",
        "QN12_t",
        "Rgr1_m",
        "Rgr2_m",
        "Rgr3_m",
        "Rgr4_m",
        "Rgr5_m",
        "Rgr6_m",
        "Rgr7_m",
        "Rgr8_m",
        "Rgr9_m",
        "Rgr10_m",
        "Rgr11_m",
        "Rgr12_m",
        "Rgr1_t",
        "Rgr2_t",
        "Rgr3_t",
        "Rgr4_t",
        "Rgr5_t",
        "Rgr6_t",
        "Rgr7_t",
        "Rgr8_t",
        "Rgr9_t",
        "Rgr10_t",
        "Rgr11_t",
        "Rgr12_t",
        "Rcp1",
        "Rcp2",
        "Rcp3",
        "Rcp4",
        "Rcp5",
        "Rcp6",
        "Rcp7",
        "Rcp8",
        "Rcp9",
        "Rcp10",
        "Rcp11",
        "Rcp12",
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


def fig_init(data, names, unit):
    fig = make_subplots(
        rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.02
    )  # subplot_titles=tuple(base_names)
    fig.update_layout(
        plot_bgcolor="white", font=dict(size=20, family="Times New Roman"), width=1200, height=700
    )
    for i, name in enumerate(names):
        fig.update_xaxes(
            linecolor="black",
            gridcolor="silver",
            mirror=True,
            range=(data.index[0], data.index[-1]),
            row=(i + 1),
        )
        axis_name = "$" + name + "[" + unit + "]$"
        fig.update_yaxes(
            linecolor="black",
            mirror=True,
            zeroline=True,
            zerolinecolor="silver",
            zerolinewidth=1,
            title=dict(text=axis_name, standoff=2),
            row=(i + 1),
        )
    fig.update_xaxes(title_text="$t[\\text{s}]$", row=len(names))
    return fig


def fig_output(fig, name):
    fig.write_html(name, include_mathjax="cdn")


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


base = calc_relinfo("r", "t", data)
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=base.iloc[:, 0], name=r"$b_x$"))
fig.add_trace(go.Scatter(x=data.index, y=base.iloc[:, 1], name=r"$b_y$"))
fig.add_trace(go.Scatter(x=data.index, y=base.iloc[:, 2], name=r"$b_z$"))
fig.update_xaxes(title_text="$t[\\text{s}]$")
fig.update_yaxes(title_text="$b[\\text{m}]$")
fig_output(fig, output_path + "baseline.html")
# fig.show()


def trans_eci2rtn(position, velocity):
    DCM_eci_to_rtn = 1


# 精度評価には最後の1000sくらいのデータを使うようにする．
# data_offset = len(data) - 1000  # s
data_offset = 600  # s

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
                x=data_for_plot.index,
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
                x=data.index,
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
                x=data.index,
                y=-np.sqrt(data[M_names[i]]) * scale_param,
                legendgroup=str(i),
                line=dict(width=1, color="black"),
                showlegend=False,
            ),
            row=(i + 1),
            col=1,
        )
    return fig


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
    # fig = go.Figure()
    fig = fig_init(data=data, names=axis_names, unit=unit)
    colors = ["red", "blue", "green"]
    for i, name in enumerate(base_names):
        est_name = name + "_" + m_t + "_e"
        true_name = name + "_" + m_t + "_t"
        precision[name] = (data[est_name] - data[true_name]) * scale_param
        RMS = np.sqrt(np.mean(precision.loc[data_offset:, name] ** 2))
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=data.index,
                y=precision[name],
                name="RMS:" + "{:.2f}".format(RMS) + "[" + unit + "]",
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
                x=data.index,
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
                x=data.index,
                y=-np.sqrt(data[M_name]) * scale_param,
                legendgroup=str(i),
                line=dict(width=1, color="black"),
                showlegend=False,
            ),
            row=(i + 1),
            col=1,
        )
        # fig.update_layout(yaxis=dict(title_text=r"$\delta$" + name +"[" + unit + "]"))
    # fig.update_layout(plot_bgcolor="#f5f5f5", paper_bgcolor="white", legend_tracegroupgap = 180, font=dict(size=15)) # lightgray
    filename = r_v_a + "_eci_precision_" + suffix + ".html"
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
    elif r_v_a == "v":
        col_base_name = "res_vel_"
        unit = "mm/s"
        scale_param = 1000
        output_name = "velocity_rtn"
        M_names_base = ["Mvr", "Mvt", "Mvn"]
    elif r_v_a == "a":
        col_names = ["ar", "at", "an"]
        unit = "um/s^2"
        scale_param = 1
        output_name = "a_rtn"
        M_names_base = ["Mar", "Mat", "Man"]
    else:
        print("false input")
        return

    frame_names = ["r", "t", "n"]
    if r_v_a != "a":
        col_names = [col_base_name + frame + "_" + m_t for frame in frame_names]

    suffix = get_suffix(m_t)

    data_for_plot = data * scale_param
    names = [r_v_a + "_" + frame for frame in frame_names]
    M_names = [M_name_base + "_" + m_t for M_name_base in M_names_base]
    fig = fig_init(data, names, unit)
    colors = ["red", "blue", "green"]
    for i in range(len(names)):
        RMS = np.sqrt(np.mean(data_for_plot.loc[data_offset:, col_names[i]] ** 2))
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=data_for_plot.index,
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
                x=data.index,
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
                x=data.index,
                y=-np.sqrt(data[M_names[i]]) * scale_param,
                legendgroup=str(i),
                line=dict(width=1, color="black"),
                showlegend=False,
            ),
            row=(i + 1),
            col=1,
        )
    fig_output(fig, output_path + output_name + "_precision_" + suffix + ".html")


def plot_differential_precision(
    data: pd.DataFrame(), precision_data: pd.DataFrame(), r_v_a: str, input_frame: str
) -> None:
    if r_v_a == "r":
        base_names = ["x", "y", "z"]
        axis_names = ["r_r", "r_t", "r_n"]
        unit = "m"
        scale_param = 1
        output_name = "relative_position"
        M_names = ["Mrr", "Mrt", "Mrn"]
    elif r_v_a == "v":
        base_names = ["vx", "vy", "vz"]
        axis_names = ["v_r", "v_t", "v_n"]
        unit = "mm/s"
        scale_param = 1000
        output_name = "relative_velocity"
        M_names = ["Mvr", "Mvt", "Mvn"]
    elif r_v_a == "a":
        base_names = ["ar", "at", "an"]
        unit = "um/s^2"
        scale_param = 1
        axis_names = ["a_r", "a_t", "a_n"]
        output_name = "relative_a"
        M_names = ["Mar", "Mat", "Man"]
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
        RMS = np.sqrt(np.mean(precision_data.iloc[data_offset:, i] ** 2))
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=precision_data.index,
                y=precision_data.iloc[:, i],
                name="RMS:" + "{:.3f}".format(RMS) + "[" + unit + "]",
                legendgroup=str(i + 1),
                marker=dict(size=2, color=colors[i]),
            ),
            row=(i + 1),
            col=1,
        )
        M_name_m = M_names[i] + "_m"
        M_name_t = M_names[i] + "_t"
        dM_name = "d" + M_names[i]
        data[dM_name] = data[M_name_m] + data[M_name_t]
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=np.sqrt(data[dM_name]) * scale_param,
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
                x=data.index,
                y=-np.sqrt(data[dM_name]) * scale_param,
                legendgroup=str(i),
                line=dict(width=1, color="black"),
                showlegend=False,
            ),
            row=(i + 1),
            col=1,
        )
    # fig.update_layout(plot_bgcolor="#f5f5f5", paper_bgcolor="white", showlegend=True, legend_tracegroupgap = 200, font=dict(size=15)) # lightgray
    fig_output(fig, output_path + output_name + "_precision.html")


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
                x=data.index,
                y=data[a_names[i]] * scale_factor,
                name=a_names[i],
                marker=dict(size=2, color="red"),
            ),
            row=i + 1,
            col=1,
        )
    # fig.update_xaxes(title_text="t[s]")
    # fig.update_yaxes(title_text="acc[nm/s2]")
    fig_output(fig, output_path + "a_emp_est_" + m_t + ".html")


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
                x=data.index,
                y=data[a_names[i]] * scale_factor,
                name=a_base_names[i],
                marker=dict(size=2, color="red"),
            ),
            row=i + 1,
            col=1,
        )
    fig_output(fig, output_path + "a_emp_eci_" + m_t + ".html")


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
                x=data.index,
                y=data[a_names[i]] * 1e3,
                name=a_base_names[i],
                marker=dict(size=2, color="red"),
            ),
            row=i + 1,
            col=1,
        )
    fig_output(fig, output_path + "a_dist_" + frame + "_" + m_t + ".html")


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
                x=a_t.index,
                y=a_t[a_base_names[i]],
                name=a_base_names[i],
                marker=dict(size=2, color="red"),
            ),
            row=i + 1,
            col=1,
        )
    fig_output(fig, output_path + "a_true_eci_" + m_t + ".html")


def plot_a_rtn_true(data_s2e_log, m_t):
    a_base_names = ["a_r", "a_t", "a_n"]
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
                x=a_t.index,
                y=a_t[a_base_names[i]],
                name=a_base_names[i],
                marker=dict(size=2, color="red"),
            ),
            row=i + 1,
            col=1,
        )
    fig_output(fig, output_path + "a_true_rtn_" + m_t + ".html")


# dataをRTNで入れてしまうとここでちゃんとしたプロットができない．
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
fig_output(fig, output_path + "orbit_3d.html")

# これはRTNで入れんと変じゃない？
baseline = calc_relinfo("r", "t", data)
fig = go.Figure()
fig.add_trace(
    go.Scatter3d(
        x=baseline["x"],
        y=baseline["y"],
        z=baseline["z"],
        name="relative orbit",
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
fig_output(fig, output_path + "relative_orbit_3d.html")


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


# 時系列データに対して座標変換するの全て変換行列変わるしだるいな．．．最初からLVLHで出したい．．．
fig = go.Figure()
for i in range(len(baseline)):
    DCM = DCM_from_eci_to_lvlh((data.loc[i, "x_m_t":"z_m_t"], data.loc[i, "vx_m_t":"vz_m_t"]))
    baseline.iloc[i, :] = np.dot(DCM, baseline.iloc[i, :])
fig.add_trace(
    go.Scatter3d(
        x=baseline["x"],
        y=baseline["y"],
        z=baseline["z"],
        name="relative orbit",
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
fig_output(fig, output_path + "relative_orbit_3d_lvlh.html")


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
                x=data.index,
                y=data[data_true_key],
                name=names[i] + "_t",
                line=dict(width=2, color="black"),
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[data_est_key],
                name=names[i] + "_e",
                line=dict(width=2, color="red"),
            ),
            row=i + 1,
            col=1,
        )
    fig_output(fig, output_name + ".html")


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
            x=cdt_precision.index,
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
            x=cdt_precision.index,
            y=np.sqrt(cdt_precision[M_name]),
            line=dict(width=1, color="black"),
            name="1 sigma",
        ),
        row=i + 1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cdt_precision.index,
            y=-np.sqrt(cdt_precision[M_name]),
            line=dict(width=1, color="black"),
            showlegend=False,
        ),
        row=i + 1,
        col=1,
    )
fig_output(fig, output_path + "cdt_sparse_precision.html")

cdt_precision.loc[:, "dcdt"] = cdt_precision.loc[:, t_names[1]] - cdt_precision.loc[:, t_names[0]]
fig = fig_init(cdt_precision, ["\Delta c\delta t"], unit="m")
fig.add_trace(
    go.Scatter(
        x=cdt_precision.index,
        y=(cdt_precision["dcdt"]),
        mode="markers",
        marker=dict(size=2, color="red"),
    ),
)
fig_output(fig, output_path + "dcdt_sparse_precision.html")


def calc_N_precision(data, m_t):
    precision = pd.DataFrame()
    for i in range(12):
        t_col_name = "N" + str(i + 1) + "_" + m_t + "_t"
        e_col_name = "N" + str(i + 1) + "_" + m_t + "_e"
        precision[i + 1] = data[e_col_name] - data[t_col_name]
    return precision


def plot_N_precision(data, m_t):
    fig = go.Figure()
    precision = calc_N_precision(data, m_t)
    for i in range(12):
        fig.add_trace(go.Scatter(x=data.index, y=precision[i + 1], name="N" + str(i + 1)))
    fig.update_xaxes(title_text="$t[s]$")
    fig.update_yaxes(title_text="$N[cycle]$")
    suffix = get_suffix(m_t)
    fig_output(fig, output_path + "N_precision_" + suffix + ".html")


def plot_dN_precision(data: pd.DataFrame):
    precision_m = calc_N_precision(data, "m")
    precision_t = calc_N_precision(data, "t")
    precision = pd.DataFrame(columns=[i + 1 for i in range(12)])
    for row in data.itertuples():
        # このやり方しかないんか？コスト高すぎしんどい．
        row_array = np.zeros(12)
        for i in range(12):
            id_m = row._asdict()["id_ch" + str(i + 1) + "_m"]
            for j in range(12):
                if row._asdict()["id_ch" + str(j + 1) + "_t"] == id_m:
                    row_array[i] = (
                        precision_m.at[row.Index, i + 1] - precision_t.at[row.Index, j + 1]
                    )
                    break
        precision.loc[row.Index] = row_array
    fig = go.Figure()
    for i in range(12):
        fig.add_trace(go.Scatter(x=data.index, y=precision[i + 1], name="dN" + str(i + 1)))
    fig.update_xaxes(title_text="$t[s]$")
    fig.update_yaxes(title_text="$dN[cycle]$")
    fig_output(fig, output_path + "dN_precision" + ".html")


def plot_N_fix_flag(data, m_t):
    precision = pd.DataFrame()
    fig = go.Figure()
    for i in range(12):
        col_name = "N" + str(i + 1) + "_" + m_t
        fig.add_trace(go.Scatter(x=data.index, y=data[col_name], name="N" + str(i + 1)))
    fig.update_xaxes(title_text="$t[s]$")
    fig.update_yaxes(title_text="is_fixed")
    suffix = get_suffix(m_t)

    fig_output(fig, output_path + "N_is_fixed_" + suffix + ".html")


def N_plot(m_t: str, t_e: str) -> None:
    fig = go.Figure()
    for i in range(12):
        t_col_name = "N" + str(i + 1) + "_" + m_t + "_" + t_e
        fig.add_trace(go.Scatter(x=data.index, y=data[t_col_name], name="N" + str(i + 1)))
    fig.update_xaxes(title_text=" $t[s]$ ")
    fig.update_yaxes(title_text=" $N[cycle]$ ")
    if t_e == "t":
        suffix1 = "true"
    else:
        suffix1 = "est"
    suffix2 = get_suffix(m_t)

    fig_output(fig, output_path + "N_" + suffix1 + "_" + suffix2 + ".html")


def plot_QM_N(data, Q_M, m_t):
    fig = go.Figure()
    out_fname_base = Q_M + "N"
    for i in range(12):
        col_name = out_fname_base + str(i + 1) + "_" + m_t
        fig.add_trace(
            go.Scatter(
                x=data.index, y=np.sqrt(data[col_name]), name="$" + Q_M + "_{N" + str(i + 1) + "}$"
            )
        )
    fig.update_xaxes(title_text="$t[s]$")
    fig.update_yaxes(title_text="$" + Q_M + "[cycle]$")
    suffix = get_suffix(m_t)

    fig_output(fig, output_path + out_fname_base + "_" + suffix + ".html")


def plot_Ma(data, m_t):
    Ma_base_names = ["Mar", "Mat", "Man"]
    Ma_names = [base_name + "_" + m_t for base_name in Ma_base_names]
    # fig = make_subplots(rows=3, cols=1, subplot_titles=tuple(Ma_base_names))
    fig = fig_init(data, Ma_base_names, unit="nm/s2")
    for i in range(3):
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=data.index,
                y=np.sqrt(data[Ma_names[i]]),
                name=Ma_base_names[i],
                marker=dict(size=2, color="red"),
            ),
            row=i + 1,
            col=1,
        )
    fig_output(fig, output_path + "Ma_emp_" + get_suffix(m_t) + ".html")


def plot_visible_gnss_sat(data: pd.DataFrame()):
    fig = go.Figure()
    names = ["main", "target", "common"]
    fig = fig_init(data, names, unit="nm/s2")
    for i in range(3):
        fig.add_trace(
            go.Scatter(
                mode="lines",
                x=data.index,
                y=np.sqrt(data["sat_num_" + names[i]]),
                name=names[i],
            ),
            row=i + 1,
            col=1,
        )
    fig_output(fig, output_path + "visible_gnss_sat.html")


def plot_gnss_id(data: pd.DataFrame(), m_t: str) -> None:
    fig = go.Figure()
    for i in range(12):
        ch_col_name = "id_ch" + str(i + 1) + "_" + m_t
        fig.add_trace(go.Scatter(x=data.index, y=data[ch_col_name], name="ch" + str(i + 1)))
    fig.update_xaxes(title_text="t[s]")
    fig.update_yaxes(title_text="gnss sat id")
    suffix = get_suffix(m_t)
    fig_output(fig, output_path + "gnss_sat_id_" + suffix + ".html")


def plot_Q(data, rvat, m_t):
    # fig = go.Figure()
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
            go.Scatter(x=data.index, y=np.sqrt(data[name]), name=base_names[i]),
            row=i + 1,
            col=1,
        )
    # fig.update_xaxes(title_text="t[s]")
    # fig.update_yaxes(title_text="Q[" + unit + "]")
    fig_output(fig, output_path + file_name_base + "_" + suffix + ".html")


def plot_R(data: pd.DataFrame(), observable: str, m_t: str) -> None:
    fig = go.Figure()
    if observable == "GRAPHIC":
        R_name = "Rgr"
        suffix = get_suffix(m_t)
        col_suffix = "_" + m_t
    elif observable == "SDCP":
        R_name = "Rcp"
        suffix = ""
        col_suffix = ""

    for i in range(12):
        col_name = R_name + str(i + 1) + col_suffix
        fig.add_trace(
            go.Scatter(
                x=data.index, y=np.sqrt(data[col_name]), name="R " + observable + str(i + 1)
            )
        )

    fig.update_xaxes(title_text="$t[s]$")
    fig.update_yaxes(title_text="$R[m]$")
    fig_output(fig, output_path + "R_" + observable + "_" + suffix + ".html")


# 受信点の重心からのずれをプロットする．
def plot_receive_position(data: pd.DataFrame, data_s2e: pd.DataFrame, m_t: str) -> None:
    pos_base_name = ["x", "y", "z"]
    pos_names = [base + "_" + m_t + "_t" for base in pos_base_name]
    arp_names = ["gnss_arp_true_eci(X)[m]", "gnss_arp_true_eci(Y)[m]", "gnss_arp_true_eci(Z)[m]"]

    fig = fig_init(data, pos_base_name, "m")
    for i in range(3):
        diff = data_s2e.loc[:, arp_names[i]] - data.loc[:, pos_names[i]]
        fig.add_trace(go.Scatter(x=data.index, y=diff, name=pos_base_name[i]), row=i + 1, col=1)

    fig_output(fig, output_path + "arp_diff_" + get_suffix(m_t) + ".html")


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
        fig.add_trace(go.Scatter(x=data.index, y=diff, name=pos_base_name[i]), row=i + 1, col=1)
    fig_output(fig, output_path + "navigated_pos_diff_" + get_suffix(m_t) + ".html")


def plot_gnss_direction(data, m_t):
    fig = px.scatter_polar(range_theta=[0, 360], start_angle=0, direction="counterclockwise")

    for i in range(12):
        azi_name = "azi" + str(i + 1) + "_" + m_t
        ele_name = "ele" + str(i + 1) + "_" + m_t
        fig.add_trace(
            go.Scatterpolar(
                r=90 - data[ele_name],
                theta=data[azi_name],
                mode="markers",
                name="ch " + str(i + 1),
            ),
        )
    suffix = get_suffix(m_t)
    fig_output(fig, output_path + "gnss_observed_direction_" + suffix + ".html")


def plot_pco(data: pd.DataFrame, m_t: str) -> None:
    pco_base_name = ["pco_x", "pco_y", "pco_z"]
    fig = fig_init(data, pco_base_name, "mm")
    col_names = [base + "_" + m_t for base in pco_base_name]
    for i in range(len(pco_base_name)):
        fig.add_trace(
            go.Scatter(x=data.index, y=data[col_names[i]], name=pco_base_name[i]), row=i + 1, col=1
        )
    suffix = get_suffix(m_t)
    fig_output(fig, output_path + "pco_" + suffix + ".html")


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

    fig_output(fig, output_path + out_fname + ".html")


def plot_pcv_by_matplotlib(pcc_path, out_fname):
    pcv_df = pd.read_csv(pcc_path, skiprows=1, header=None)
    num_azi, num_ele = pcv_df.shape
    azi_increment = 360 / (num_azi - 1)
    ele_increment = 90 / (num_ele - 1)
    azi = np.deg2rad(np.arange(0, 365, azi_increment))
    ele = np.arange(0, 95, ele_increment)
    # azi, ele = np.mgrid[0:365:azi_increment, 90:-5:-ele_increment]
    # azi, ele = np.mgrid[0:365:azi_increment, 0:95:ele_increment]
    azi, ele = np.meshgrid(azi, ele)

    fig = plt.figure()
    # ax = Axes3D(fig)
    z = pcv_df.values.T
    # z = np.fliplr(pcv_df.values)

    # ax = plt.subplot(projection="polar")
    ax = plt.subplot(polar=True)
    cmin = -5.0
    cmax = 9.0
    pcolor = ax.pcolormesh(azi, ele, z, cmap=cm.jet, norm=Normalize(vmin=cmin, vmax=cmax))
    colb = fig.colorbar(pcolor, ax=ax, orientation="vertical")

    # plt.grid()
    # colb.set_label("label", fontname="Arial", fontsize=20)

    plt.savefig(output_path + out_fname + ".jpg", dpi=600)
    plt.savefig(output_path + out_fname + ".eps")
    # plt.show()


# plot
plot_precision("r", "m", data)
plot_precision("r", "t", data)
plot_precision("v", "m", data)
plot_precision("v", "t", data)
plot_precision_rtn("r", "m", data)
plot_precision_rtn("r", "t", data)
plot_precision_rtn("v", "m", data)
plot_precision_rtn("v", "t", data)
a_m_precision = calc_a_precision(data, data_s2e_csv, "m", "rtn")
a_t_precision = calc_a_precision(data, data_s2e_csv, "t", "rtn")
plot_precision_rtn("a", "m", a_m_precision)
plot_precision_rtn("a", "t", a_t_precision)
pbd_precision = calc_relinfo("r", "e", data) - calc_relinfo("r", "t", data)
plot_differential_precision(data, pbd_precision, "r", "ECI")
dv_precision = calc_relinfo("v", "e", data) - calc_relinfo("v", "t", data)
plot_differential_precision(data, dv_precision, "v", "ECI")
plot_differential_precision(
    data, a_t_precision.iloc[:, 0:3] - a_m_precision.iloc[:, 0:3], "a", "RTN"
)

plot_a(data, "m")
plot_a(data, "t")
# plot_a_precision(data, data_s2e_csv, "m", "eci")
# plot_a_precision(data, data_s2e_csv, "t", "eci")
# plot_a_eci(data, "m")
# plot_a_eci(data, "t")
# plot_a_dist(data, "m", "eci")
# plot_a_dist(data, "t", "eci")
# plot_a_dist(data, "m", "rtn")
# plot_a_dist(data, "t", "rtn")
# plot_a_eci_true(data_s2e_csv, "m")
# plot_a_rtn_true(data_s2e_csv, "m")

# cdt_plot(data, output_path + "cdt")
# cdt_plot(data[data.index % 10 == 9], output_path + "cdt_sparse")

plot_N_precision(data, "m")
plot_N_precision(data, "t")
# plot_dN_precision(data)  # 計算コスト高いので必要な時だけにする．
plot_N_fix_flag(data, "m")
plot_N_fix_flag(data, "t")
# N_plot("m", "t")
# N_plot("t", "t")
# N_plot("m", "e")
# N_plot("t", "e")

plot_R(data, "GRAPHIC", "m")
plot_R(data, "GRAPHIC", "t")
plot_R(data, "SDCP", "")

plot_QM_N(data, "Q", "m")
plot_QM_N(data, "Q", "t")
plot_QM_N(data, "M", "m")
plot_QM_N(data, "M", "t")
plot_Q(data, "r", "m")
plot_Q(data, "r", "t")
plot_Q(data, "v", "m")
plot_Q(data, "v", "t")
plot_Q(data, "t", "")
if REDUCE_DYNAMIC:
    plot_Q(data, "a", "m")
    plot_Q(data, "a", "t")
    plot_Ma(data, "m")
    plot_Ma(data, "t")
# plot_receive_position(data, data_s2e_csv, "m")
# plot_determined_position_precision(data, data_s2e_csv, "m")
plot_gnss_direction(data, "m")
plot_gnss_direction(data, "t")
# plot_visible_gnss_sat(data)
plot_gnss_id(data, "m")
plot_gnss_id(data, "t")
plot_pco(data, "m")
plot_pco(data, "t")


plot_pcv_grid(pcc_log_path, "pcv_true")
# plot_pcv_grid(s2e_debug + "target_pcv.csv", "estimated_target_pcv")

plot_pcv_by_matplotlib(pcc_log_path, "pcv_true")
plot_pcv_by_matplotlib(s2e_debug + "target_pcv.csv", "estimated_target_pcv")

# 最後に全グラフをまとめてコピー
shutil.move(output_path, copy_dir + "/figure/")
