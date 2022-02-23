import pandas as pd
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

sim_config_path = '../s2e_pbd/CMakeBuilds/Debug/readdme_new.txt'
log_path = '../s2e_pbd/CMakeBuilds/Debug/result_new.csv'

data = pd.read_csv(log_path, header=None)
num_of_status = 20
data_col = ['x_m_t', 'y_m_t', 'z_m_t', 't_m_t', 'vx_m_t', 'vy_m_t', 'vz_m_t', 'x_t_t', 'y_t_t', 'z_t_t', 't_t_t', 'vx_t_t', 'vy_t_t', 'vz_t_t', 'x_m_e', 'y_m_e', 'z_m_e', 't_m_e', 'vx_m_e', 'vy_m_e', 'vz_m_e', 'ax_m_e', 'ay_m_e', 'az_m_e','x_t_e', 'y_t_e', 'z_t_e', 't_t_e', 'vx_t_e', 'vy_t_e', 'vz_t_e', 'ax_t_e', 'ay_t_e', 'az_t_e', 'N1_m_t', 'N2_m_t', 'N3_m_t', 'N4_m_t', 'N5_m_t', 'N6_m_t', 'N7_m_t', 'N8_m_t', 'N9_m_t', 'N10_m_t', 'N11_m_t', 'N12_m_t', 'N1_m_e', 'N2_m_e', 'N3_m_e', 'N4_m_e', 'N5_m_e', 'N6_m_e', 'N7_m_e', 'N8_m_e', 'N9_m_e', 'N10_m_e', 'N11_m_e', 'N12_m_e', 'N1_t_t', 'N2_t_t', 'N3_t_t', 'N4_t_t', 'N5_t_t', 'N6_t_t', 'N7_t_t', 'N8_t_t', 'N9_t_t', 'N10_t_t', 'N11_t_t', 'N12_t_t', 'N1_t_e', 'N2_t_e', 'N3_t_e', 'N4_t_e', 'N5_t_e', 'N6_t_e', 'N7_t_e', 'N8_t_e', 'N9_t_e', 'N10_t_e', 'N11_t_e', 'N12_t_e', 'Mx_m', 'My_m', 'Mz_m', 'Mt_m', 'Mvx_m', 'Mvy_m', 'Mvz_m', 'Max_m', 'May_m', 'Maz_m', 'MNm1', 'MNm2', 'MNm3', 'MNm4', 'MNm5', 'MNm6', 'MNm7', 'MNm8', 'MNm9', 'MNm10', 'MNm11', 'MNm12', 'Mx_t', 'My_t', 'Mz_t', 'Mt_t', 'Mvx_t', 'Mvy_t', 'Mvz_t', 'Max_t', 'May_t', 'Maz_t', 'MNt1', 'MNt2', 'MNt3', 'MNt4', 'MNt5', 'MNt6', 'MNt7', 'MNt8', 'MNt9', 'MNt10', 'MNt11', 'MNt12', 'sat_num']
data = data.set_axis(data_col, axis=1)

def true_baseline(data):
    base = pd.DataFrame()
    base['x'] = data['x_t_t'] - data['x_m_t']
    base['y'] = data['y_t_t'] - data['y_m_t']
    base['z'] = data['z_t_t'] - data['z_m_t']
    return base

def calc_baseline_precision(data):
    base_true = true_baseline(data)
    precision = pd.DataFrame()
    precision['x'] = data['x_t_e'] - data['x_m_e'] - base_true['x']
    precision['y'] = data['y_t_e'] - data['y_m_e'] - base_true['y']
    precision['z'] = data['z_t_e'] - data['z_m_e'] - base_true['z']
    return precision
# print(data['sat_num'].head(30))

base = true_baseline(data)
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=base.iloc[:,0], name='$b_x$'))
fig.add_trace(go.Scatter(x=data.index, y=base.iloc[:,1], name='$b_y$'))
fig.add_trace(go.Scatter(x=data.index, y=base.iloc[:,2], name='$b_z$'))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$b[m]$")
fig.write_html("baseline.html")
# fig.show()

pbd_precision = calc_baseline_precision(data)
fig = go.Figure()
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=pbd_precision.iloc[:,0], name='$dx$', marker=dict(size=2)))
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=pbd_precision.iloc[:,1], name='$dy$', marker=dict(size=2)))
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=pbd_precision.iloc[:,2], name='$dz$', marker=dict(size=2)))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$residual[m]$")
fig.write_html("pbd_precision.html")
# fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Mx_t'], name='Mx'))
fig.add_trace(go.Scatter(x=data.index, y=data['My_t'], name='My'))
fig.add_trace(go.Scatter(x=data.index, y=data['Mz_t'], name='Mz'))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$Mr_{target}[m]$")
fig.write_html("Mr_target.html")
# fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Mx_m'], name='Mx'))
fig.add_trace(go.Scatter(x=data.index, y=data['My_m'], name='My'))
fig.add_trace(go.Scatter(x=data.index, y=data['Mz_m'], name='Mz'))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$Mr_{main}[m]$")
fig.write_html("Mr_main.html")
# fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Mvx_t'], name='Mvx'))
fig.add_trace(go.Scatter(x=data.index, y=data['Mvy_t'], name='Mvy'))
fig.add_trace(go.Scatter(x=data.index, y=data['Mvz_t'], name='Mvz'))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$Mv_{target}[m]$")
fig.write_html("Mv_target.html")
# fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Mvx_m'], name='Mvx'))
fig.add_trace(go.Scatter(x=data.index, y=data['Mvy_m'], name='Mvy'))
fig.add_trace(go.Scatter(x=data.index, y=data['Mvz_m'], name='Mvz'))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$Mv_{main}[m]$")
fig.write_html("Mv_main.html")
# fig.show()

precision = pd.DataFrame()
precision['x'] = data['x_m_e'] - data['x_m_t']
precision['y'] = data['y_m_e'] - data['y_m_t']
precision['z'] = data['z_m_e'] - data['z_m_t']
fig = go.Figure()
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=precision['x'], name='$x$', marker=dict(size=2)))
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=precision['y'], name='$y$', marker=dict(size=2)))
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=precision['z'], name='$z$', marker=dict(size=2)))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$residual[m]$")
fig.write_html("pod_precision.html")
# fig.show()

precision['x'] = data['x_t_e'] - data['x_t_t']
precision['y'] = data['y_t_e'] - data['y_t_t']
precision['z'] = data['z_t_e'] - data['z_t_t']
fig = go.Figure()
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=precision['x'], name='$x$', marker=dict(size=2)))
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=precision['y'], name='$y$', marker=dict(size=2)))
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=precision['z'], name='$z$', marker=dict(size=2)))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$residual[m]$")
fig.write_html("pod_precision_target.html")
# fig.show()

precision = pd.DataFrame()
precision['t_m'] = data['t_m_e'] - data['t_m_t']
precision['t_t'] = data['t_t_e'] - data['t_t_t']
precision['dt'] = precision['t_m'] - precision['t_t']
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=precision['t_m'], name='$cdt_{main}$'))
fig.add_trace(go.Scatter(x=data.index, y=precision['t_t'], name='$cdt_{target}$'))
fig.add_trace(go.Scatter(x=data.index, y=precision['dt'], name='$\Delta cdt$'))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$residual[m]$")
fig.write_html("dt_precision.html")
# fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['t_m_t'], name='$t_{main_{t}}$'))
fig.add_trace(go.Scatter(x=data.index, y=data['t_m_e'], name='$t_{main_{e}}$'))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$cdt[m]$")
fig.write_html("t.html")

v_abs = pd.DataFrame()
v_abs['vx'] = data['vx_m_t']
v_abs['vy'] = data['vy_m_t']
v_abs['vz'] = data['vz_m_t']
precision['x'] = data['vx_m_e'] - v_abs['vx']
precision['y'] = data['vy_m_e'] - v_abs['vy']
precision['z'] = data['vz_m_e'] - v_abs['vz']
fig = go.Figure()
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=precision['x'], name='$v_x$', marker=dict(size=2)))
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=precision['y'], name='$v_y$', marker=dict(size=2)))
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=precision['z'], name='$v_z$', marker=dict(size=2)))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$residual[m/s]$")
fig.write_html("v_precision.html")
# fig.show()

precision['x'] = data['vx_t_e'] - data['vx_t_t']
precision['y'] = data['vy_t_e'] - data['vy_t_t']
precision['z'] = data['vz_t_e'] - data['vz_t_t']
fig = go.Figure()
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=precision['x'], name='$v_x$', marker=dict(size=2)))
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=precision['y'], name='$v_y$', marker=dict(size=2)))
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=precision['z'], name='$v_z$', marker=dict(size=2)))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$residual[m/s]$")
fig.write_html("v_precision_target.html")

v_rel = pd.DataFrame()
v_rel['vx'] = data['vx_t_t'] - data['vx_m_t']
v_rel['vy'] = data['vy_t_t'] - data['vy_m_t']
v_rel['vz'] = data['vz_t_t'] - data['vz_m_t']
precision['x'] = data['vx_t_e'] - data['vx_m_e'] - v_rel['vx']
precision['y'] = data['vy_t_e'] - data['vy_m_e'] - v_rel['vy']
precision['z'] = data['vz_t_e'] - data['vz_m_e'] - v_rel['vz']
fig = go.Figure()
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=precision['x'], name='$dv_x$', marker=dict(size=2)))
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=precision['y'], name='$dv_y$', marker=dict(size=2)))
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=precision['z'], name='$dv_z$', marker=dict(size=2)))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$residual[m/s]$")
fig.write_html("dv_precision.html")
# fig.show()

precision = pd.DataFrame()
fig = go.Figure()
for i in range(12):
    t_col_name = 'N' + str(i+1) +'_m_t'
    e_col_name = 'N' + str(i+1) +'_m_e'
    precision[i+1] = data[e_col_name] - data[t_col_name]
    fig.add_trace(go.Scatter(x=data.index, y=precision[i+1], name='N'+str(i+1)))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$bias[m]$")
fig.write_html("bias_precision_main.html")
# fig.show()

fig = go.Figure()
for i in range(12):
    t_col_name = 'N' + str(i+1) +'_m_t'
    fig.add_trace(go.Scatter(x=data.index, y=data[t_col_name], name='N'+str(i+1)))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$bias[m]$")
fig.write_html("bias_true_main.html")
# fig.show()

fig = go.Figure()
for i in range(12):
    e_col_name = 'N' + str(i+1) +'_m_e'
    fig.add_trace(go.Scatter(x=data.index, y=data[e_col_name], name='N'+str(i+1)))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$bias[m]$")
fig.write_html("bias_est_main.html")
# fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=data['ax_m_e'], name='ax_m_e', marker=dict(size=2)))
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=data['ay_m_e'], name='ay_m_e', marker=dict(size=2)))
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=data['az_m_e'], name='az_m_e', marker=dict(size=2)))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$acc[nm/s2]$")
fig.write_html("a_emp_est_main.html")
# fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=data['ax_t_e'], name='ax_t_e', marker=dict(size=2)))
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=data['ay_t_e'], name='ay_t_e', marker=dict(size=2)))
fig.add_trace(go.Scatter(mode='markers', x=data.index, y=data['az_t_e'], name='az_t_e', marker=dict(size=2)))
fig.update_xaxes(title_text="$t[s]$")
fig.update_yaxes(title_text="$acc[nm/s2]$")
fig.write_html("a_emp_est_target.html")
# fig.show()
