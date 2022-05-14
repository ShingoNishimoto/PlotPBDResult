from matplotlib.axis import YAxis
import pandas as pd
import plotly as py
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import shutil
import os
from glob import glob
import datetime
import numpy as np

def get_latest_modified_file_path(dirname):
  target = os.path.join(dirname, '*')
  files = [(f, os.path.getmtime(f)) for f in glob(target)]
  latest_modified_file_path = sorted(files, key=lambda files: files[1])[-1]
  return latest_modified_file_path[0]


s2e_dir = '../../s2e_pbd/'
s2e_log_path = s2e_dir + 'data/logs'
sim_config_path = s2e_dir + 'CMakeBuilds/Debug/readme_new.txt'
log_path = s2e_dir + 'CMakeBuilds/Debug/result_new.csv'
output_path = 'figure/'
# ここは適宜修正する．
copy_base_dir = '/mnt/g/マイドライブ/Documents/University/lab/Research/FFGNSS/plot_result/'
dt_now = datetime.datetime.now()
copy_dir = copy_base_dir + dt_now.strftime('%Y%m%d_%H%M%S')
os.mkdir(copy_dir)
# os.mkdir(copy_dir + '/figure')
shutil.copyfile(sim_config_path, copy_dir + '/readme.txt')
shutil.copyfile(log_path, copy_dir + '/result.txt')
shutil.copytree(get_latest_modified_file_path(s2e_log_path), copy_dir + '/s2e_logs')

data = pd.read_csv(log_path, header=None)
data_col = ['x_m_t', 'y_m_t', 'z_m_t', 't_m_t', 'vx_m_t', 'vy_m_t', 'vz_m_t', 'x_t_t', 'y_t_t', 'z_t_t', 't_t_t', 'vx_t_t', 'vy_t_t', 'vz_t_t', 'x_m_e', 'y_m_e', 'z_m_e', 't_m_e', 'vx_m_e', 'vy_m_e', 'vz_m_e', 'ar_m_e', 'at_m_e', 'an_m_e','x_t_e', 'y_t_e', 'z_t_e', 't_t_e', 'vx_t_e', 'vy_t_e', 'vz_t_e', 'ar_t_e', 'at_t_e', 'an_t_e', 'N1_m_t', 'N2_m_t', 'N3_m_t', 'N4_m_t', 'N5_m_t', 'N6_m_t', 'N7_m_t', 'N8_m_t', 'N9_m_t', 'N10_m_t', 'N11_m_t', 'N12_m_t', 'N1_m_e', 'N2_m_e', 'N3_m_e', 'N4_m_e', 'N5_m_e', 'N6_m_e', 'N7_m_e', 'N8_m_e', 'N9_m_e', 'N10_m_e', 'N11_m_e', 'N12_m_e', 'N1_t_t', 'N2_t_t', 'N3_t_t', 'N4_t_t', 'N5_t_t', 'N6_t_t', 'N7_t_t', 'N8_t_t', 'N9_t_t', 'N10_t_t', 'N11_t_t', 'N12_t_t', 'N1_t_e', 'N2_t_e', 'N3_t_e', 'N4_t_e', 'N5_t_e', 'N6_t_e', 'N7_t_e', 'N8_t_e', 'N9_t_e', 'N10_t_e', 'N11_t_e', 'N12_t_e', 'Mx_m', 'My_m', 'Mz_m', 'Mt_m', 'Mvx_m', 'Mvy_m', 'Mvz_m', 'Mar_m', 'Mat_m', 'Man_m', 'MNm1', 'MNm2', 'MNm3', 'MNm4', 'MNm5', 'MNm6', 'MNm7', 'MNm8', 'MNm9', 'MNm10', 'MNm11', 'MNm12', 'Mx_t', 'My_t', 'Mz_t', 'Mt_t', 'Mvx_t', 'Mvy_t', 'Mvz_t', 'Mar_t', 'Mat_t', 'Man_t', 'MNt1', 'MNt2', 'MNt3', 'MNt4', 'MNt5', 'MNt6', 'MNt7', 'MNt8', 'MNt9', 'MNt10', 'MNt11', 'MNt12', 'sat_num_main', 'sat_num_target','sat_num_common', 'id_ch1_m', 'id_ch2_m', 'id_ch3_m', 'id_ch4_m', 'id_ch5_m', 'id_ch6_m', 'id_ch7_m', 'id_ch8_m', 'id_ch9_m', 'id_ch10_m', 'id_ch11_m', 'id_ch12_m', 'id_ch1_t', 'id_ch2_t', 'id_ch3_t', 'id_ch4_t', 'id_ch5_t', 'id_ch6_t', 'id_ch7_t', 'id_ch8_t', 'id_ch9_t', 'id_ch10_t', 'id_ch11_t', 'id_ch12_t','']
data = data.set_axis(data_col, axis=1)


def calc_relinfo(x_v_a, t_e, data):
    """
    Args:
        x_v_a: position or velocity or acceleration
        t_e: true or estimation
        data: data(PandasDataframe)
    """
    if x_v_a == 'x':
        base_names = ['x', 'y', 'z']
    elif x_v_a == 'v':
        base_names = ['vx', 'vy', 'vz']
    else:
        base_names = ['ar', 'at', 'an']
    names_main   = [base_name + '_m_' + t_e for base_name in base_names]
    names_target = [base_name + '_t_' + t_e for base_name in base_names]
    base = pd.DataFrame()
    for i in range(len(base_names)):
        base[base_names[i]] = data[names_target[i]] - data[names_main[i]]
    return base

def calc_baseline_precision(data):
    base_true = calc_relinfo('x', 't', data)
    base_est  = calc_relinfo('x', 'e', data)
    precision = pd.DataFrame()
    precision['x'] = base_est['x'] - base_true['x']
    precision['y'] = base_est['y'] - base_true['y']
    precision['z'] = base_est['z'] - base_true['z']
    return precision
# print(data['sat_num'].head(30))

base = calc_relinfo('x', 't', data)
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=base.iloc[:,0], name=r'$b_x$'))
fig.add_trace(go.Scatter(x=data.index, y=base.iloc[:,1], name=r'$b_y$'))
fig.add_trace(go.Scatter(x=data.index, y=base.iloc[:,2], name=r'$b_z$'))
fig.update_xaxes(title_text='t[s]')
fig.update_yaxes(title_text='b[m]')
fig.write_html(output_path + "baseline.html")
# fig.show()

# x, y, zを分けて3段でプロットしたい．
def plot_precision(x_v, m_t, data):
    if x_v == 'x':
        base_names = ['x', 'y', 'z']
        unit = 'm'
        scale_param = 1
    else:
        base_names = ['vx', 'vy', 'vz']
        unit = 'mm/s'
        scale_param = 1000
    if m_t == 'm':
        suffix = 'main'
    else:
        suffix = 'target'
    precision = pd.DataFrame()
    # fig = go.Figure()
    fig = make_subplots(rows=3, cols=1, subplot_titles=tuple(base_names))
    data_offset = 1000 # s
    for i, name in enumerate(base_names):
        est_name = name + '_' + m_t + '_e'
        true_name = name + '_' + m_t + '_t'
        precision[name] = (data[est_name] - data[true_name])*scale_param
        RMS = np.sqrt(np.mean(precision.loc[data_offset:, name]**2))
        fig.add_trace(go.Scatter(mode='markers', x=data.index, y=precision[name], name='RMS:'+'{:.2f}'.format(RMS)+'['+unit+']', legendgroup=str(i) , marker=dict(size=2, color='red'), showlegend=True), row=(i+1), col=1)
        M_name = 'M' + name + '_' + m_t
        fig.add_trace(go.Scatter(x=data.index, y=np.sqrt(data[M_name])*scale_param, name="1 sigma", legendgroup=str(i), line=dict(width=1, color='black'), showlegend=True), row=(i+1), col=1)
        fig.add_trace(go.Scatter(x=data.index, y=-np.sqrt(data[M_name])*scale_param, legendgroup=str(i), line=dict(width=1, color='black'), showlegend=False), row=(i+1), col=1)
        # fig.update_layout(yaxis=dict(title_text=r"$\delta$" + name +"[" + unit + "]"))
    fig.update_layout(plot_bgcolor="#f5f5f5", paper_bgcolor="white", legend_tracegroupgap = 180, font=dict(size=15)) # lightgray
    fig.update_xaxes(title_text="t[s]")
    fig.update_yaxes(title_text="residual[" + unit + "]")
    filename = x_v + '_precision_' + suffix + '.html'
    fig.write_html(output_path + filename)

plot_precision('x', 'm', data)
plot_precision('x', 't', data)
plot_precision('v', 'm', data)
plot_precision('v', 't', data)

def plot_differential_precision(precision_data, x_v):
    if x_v == 'x':
        base_names = ['x', 'y', 'z']
        unit = 'm'
        scale_param = 1
        output_name = 'relative_position'
    else:
        base_names = ['vx', 'vy', 'vz']
        unit = 'mm/s'
        scale_param = 1000
        output_name = 'relative_velocity'
    precision_data *= scale_param
    names = ['d'+base_name for base_name in base_names]
    fig = make_subplots(rows=3, cols=1, subplot_titles=(names))
    data_offset = 1000
    for i in range(len(base_names)):
        RMS = np.sqrt(np.mean(precision_data.iloc[data_offset:,i]**2))
        fig.add_trace(go.Scatter(mode='markers', x=precision_data.index, y=precision_data.iloc[:,i], name='RMS:'+'{:.3f}'.format(RMS)+'['+unit+']', legendgroup=str(i+1), marker=dict(size=2, color='red')), row=(i+1), col=1)
    fig.update_layout(plot_bgcolor="#f5f5f5", paper_bgcolor="white", showlegend=True, legend_tracegroupgap = 200, font=dict(size=15)) # lightgray
    fig.update_xaxes(title_text="t[s]")
    fig.update_yaxes(title_text="residual["+unit+"]")
    fig.write_html(output_path + output_name +"_precision.html")

pbd_precision = calc_baseline_precision(data)
plot_differential_precision(pbd_precision, 'x')
v_rel_true = calc_relinfo('v', 't', data)
v_rel_est  = calc_relinfo('v', 'e', data)
dv_precision = v_rel_est - v_rel_true
plot_differential_precision(dv_precision, 'v')


fig = go.Figure()
fig.add_trace(go.Scatter3d(x=data['x_m_t'], y=data['y_m_t'], z=data['z_m_t'], name='main', mode='lines', line=dict(width=2, color='red')))
fig.add_trace(go.Scatter3d(x=data['x_t_t'], y=data['y_t_t'], z=data['z_t_t'], name='target', mode='lines', line=dict(width=2, color='blue')))
# fig.update_xaxes(title_text="t[s]")
# fig.update_yaxes(title_text="residual[m]")
fig.write_html(output_path + "orbit_3d.html")

baseline = calc_relinfo('x', 't', data)
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=baseline['x'], y=baseline['y'], z=baseline['z'], name='relative orbit', mode='lines', line=dict(width=2, color='red')))
# fig.update_xaxes(title_text="t[s]")
# fig.update_yaxes(title_text="residual[m]")
fig.write_html(output_path + "relative_orbit_3d.html")

def cdt_plot(data, output_name):
    names = ['cdt_main', 'cdt_target']
    suffix = ['m', 't']
    fig = make_subplots(rows=2, cols=1, subplot_titles=(tuple(names)))
    for i in range(len(names)):
        data_true_key = 't_' + suffix[i] + '_t'
        data_est_key  = 't_' + suffix[i] + '_e'
        fig.add_trace(go.Scatter(x=data.index, y=data[data_true_key], name=names[i]+'_t', line=dict(width=2, color='black')), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data[data_est_key], name=names[i]+'_e', line=dict(width=2, color='red')), row=i+1, col=1)
    fig.update_xaxes(title_text="t[s]")
    fig.update_yaxes(title_text="cdt[m]")
    fig.write_html(output_name +".html")

cdt_plot(data, output_path+'cdt')
cdt_plot(data[data.index%10==9], output_path+'cdt_sparse')

data_sparse = data[data.index%10==9]
t_names = ['cdt_main', 'cdt_target']
suffix  = ['m', 't']
fig = make_subplots(rows=2, cols=1, subplot_titles=tuple(t_names))
for i in range(len(t_names)):
    true_name_key = 't_' + suffix[i] + '_t'
    est_name_key  = 't_' + suffix[i] + '_e'
    fig.add_trace(go.Scatter(x=data_sparse.index, y=(data_sparse[est_name_key] - data_sparse[true_name_key]), name=t_names[i], mode='markers', marker=dict(size=2, color='red')), row=i+1, col=1)
    M_name = 'Mt_'+suffix[i]
    fig.add_trace(go.Scatter(x=data_sparse.index, y=np.sqrt(data_sparse[M_name]), line=dict(width=1, color='black'), name='1 sigma'), row=i+1, col=1)
    fig.add_trace(go.Scatter(x=data_sparse.index, y=-np.sqrt(data_sparse[M_name]), line=dict(width=1, color='black'), showlegend=False), row=i+1, col=1)
fig.update_xaxes(title_text="t[s]")
fig.update_yaxes(title_text="residual[m]")
fig.write_html(output_path + "cdt_sparse_precision.html")


precision = pd.DataFrame()
fig = go.Figure()
for i in range(12):
    t_col_name = 'N' + str(i+1) +'_m_t'
    e_col_name = 'N' + str(i+1) +'_m_e'
    precision[i+1] = data[e_col_name] - data[t_col_name]
    fig.add_trace(go.Scatter(x=data.index, y=precision[i+1], name='N'+str(i+1)))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$bias[m]$")
fig.write_html(output_path + "bias_precision_main.html")
# fig.show()

fig = go.Figure()
for i in range(12):
    t_col_name = 'N' + str(i+1) +'_m_t'
    fig.add_trace(go.Scatter(x=data.index, y=data[t_col_name], name='N'+str(i+1)))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$bias[m]$")
fig.write_html(output_path + "bias_true_main.html")
# fig.show()

fig = go.Figure()
for i in range(12):
    e_col_name = 'N' + str(i+1) +'_m_e'
    fig.add_trace(go.Scatter(x=data.index, y=data[e_col_name], name='N'+str(i+1)))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$bias[m]$")
fig.write_html(output_path + "bias_est_main.html")
# fig.show()

fig = go.Figure()
for i in range(12):
    e_col_name = 'N' + str(i+1) +'_t_e'
    fig.add_trace(go.Scatter(x=data.index, y=data[e_col_name], name='N'+str(i+1)))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$bias[m]$")
fig.write_html(output_path + "bias_est_target.html")
# fig.show()

def plot_a(data, m_t):
    a_base_names = ['ar', 'at', 'an']
    a_names = [base_name+'_'+m_t+'_e' for base_name in a_base_names]
    fig = make_subplots(rows=3, cols=1, subplot_titles=tuple(a_names))
    for i in range(len(a_base_names)):
        fig.add_trace(go.Scatter(mode='markers', x=data.index, y=data[a_names[i]], name=a_names[i], marker=dict(size=2, color='red')), row=i+1, col=1)
    fig.update_xaxes(title_text="t[s]")
    fig.update_yaxes(title_text="acc[nm/s2]")
    fig.write_html(output_path + "a_emp_est_"+m_t+".html")

plot_a(data, 'm')
plot_a(data, 't')

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Mt_m'], name='Mt_m'))
fig.add_trace(go.Scatter(x=data.index, y=data['Mt_t'], name='Mt_t'))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$M[m]$")
fig.write_html(output_path + "Mt.html")
# fig.show()

fig = go.Figure()
for i in range(12):
    M_col_name = 'MNm' + str(i+1)
    fig.add_trace(go.Scatter(x=data.index, y=data[M_col_name], name='MN'+str(i+1)))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$M[m]$")
fig.write_html(output_path + "MN_main.html")
# fig.show()

fig = go.Figure()
for i in range(12):
    M_col_name = 'MNt' + str(i+1)
    fig.add_trace(go.Scatter(x=data.index, y=data[M_col_name], name='MN'+str(i+1)))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$M[m]$")
fig.write_html(output_path + "MN_target.html")
# fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['sat_num_main'], name='main'))
fig.add_trace(go.Scatter(x=data.index, y=data['sat_num_target'], name='target'))
fig.add_trace(go.Scatter(x=data.index, y=data['sat_num_common'], name='common'))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="number")
fig.write_html(output_path + "visible_gnss_sat.html")
# fig.show()

fig = go.Figure()
for i in range(12):
    ch_col_name = 'id_ch' + str(i+1) + '_m'
    fig.add_trace(go.Scatter(x=data.index, y=data[ch_col_name], name='ch'+str(i+1)))
fig.update_xaxes(title_text="t[s]")
fig.update_yaxes(title_text="gnss sat id")
fig.write_html(output_path + "gnss_sat_id_main.html")
# fig.show()

fig = go.Figure()
for i in range(12):
    ch_col_name = 'id_ch' + str(i+1) + '_t'
    fig.add_trace(go.Scatter(x=data.index, y=data[ch_col_name], name='ch'+str(i+1)))
fig.update_xaxes(title_text="t[s]")
fig.update_yaxes(title_text="gnss sat id")
fig.write_html(output_path + "gnss_sat_id_target.html")
# fig.show()

# 最後に全グラフをまとめてコピー
shutil.copytree(output_path, copy_dir + '/figure')