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

FOR_RESUME = 1
x_axis_scale = 1.0 / 3600  # sec -> hour
SVG_ENABLE = 0


class FigViewSetting:
    def __init__(self):
        (
            self.global_font_size,
            self.legend_font_size,
            self.axis_title_font_size,
            self.width,
            self.height,
            self.v_space,
            self.h_space,
            self.x_pos,
        ) = set_fig_params()

    def set_layout(self, fig: go.Figure()) -> None:
        fig.update_layout(
            plot_bgcolor="white",
            font=dict(size=self.global_font_size, family="Times New Roman"),  # globalなfont設定
            width=self.width,
            height=self.height,
            legend=dict(
                x=self.x_pos,
                y=0.99,
                xanchor="left",
                yanchor="top",
                font=dict(size=self.legend_font_size),
                bordercolor="black",
                borderwidth=1,
                orientation="h",
                itemsizing="constant",
            ),
            margin=dict(t=1, b=1, l=1, r=1),
        )


def set_fig_params() -> tuple:
    if FOR_RESUME:
        global_font_size = 23
        legend_font_size = 21  # 17
        axis_title_font_size = 25  # 謎にaxis_titleのサイズが変わらない．．．
        width = 1000  # 750
        height = 600
        v_space = 0.01
        h_space = 0.01
        x_pos = 0.35  # 0.40
    else:
        global_font_size = 20
        legend_font_size = 13
        axis_title_font_size = 20
        width = 1200
        height = 700
        v_space = 0.03
        h_space = 0.01
        x_pos = 0.53

    return (
        global_font_size,
        legend_font_size,
        axis_title_font_size,
        width,
        height,
        v_space,
        h_space,
        x_pos,
    )


# FIXME この辺はクラス化したい．
def fig_init(data, names, unit) -> go.Figure():
    view_set = FigViewSetting()

    fig = make_subplots(
        rows=len(names),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=view_set.v_space,  # , shared_yaxes=True
    )  # subplot_titles=tuple(base_names)
    view_set.set_layout(fig)

    for i, name in enumerate(names):
        fig.update_xaxes(
            linecolor="black",
            gridcolor="silver",
            mirror=True,
            range=(0, data.index[-1] * x_axis_scale),
            row=(i + 1),
        )
        axis_name = "$" + name + "[" + unit + "]$"
        fig.update_yaxes(
            linecolor="black",
            mirror=True,
            zeroline=True,
            zerolinecolor="silver",
            zerolinewidth=1,
            title=dict(
                text=axis_name,
                standoff=2,
            ),
            titlefont_size=view_set.axis_title_font_size,
            row=(i + 1),
        )
    fig.update_xaxes(
        title=dict(text="$t[\\text{hours}]$"),
        titlefont_size=view_set.axis_title_font_size,
        row=len(names),
    )
    return fig


# name, unitじゃなくてy_axis_label受け取ればいいかもな
def fig_init_horizon_bar(datas: list, categories: list, y_axis_label: str) -> go.Figure():
    view_set = FigViewSetting()

    fig = make_subplots(
        rows=1,
        cols=len(datas),
        # cols=1,
        shared_yaxes=True,
        horizontal_spacing=view_set.h_space,  # , shared_yaxes=True
    )  # subplot_titles=tuple(base_names)
    view_set.set_layout(fig)

    for i, data in enumerate(datas):
        # axis_name = category
        fig.update_xaxes(
            linecolor="black",
            # gridcolor="silver",
            mirror=True,
            # range=(0, data.index[-1] * x_axis_scale),
            col=(i + 1),
            # title=dict(
            #     text=axis_name,
            #     standoff=2,
            # ),
            titlefont_size=view_set.axis_title_font_size,
        )
        fig.update_yaxes(
            linecolor="black",
            mirror=True,
            zeroline=True,
            zerolinecolor="silver",
            zerolinewidth=1,
            col=(i + 1),
        )
    fig.update_yaxes(
        title=dict(text=y_axis_label),
        titlefont_size=view_set.axis_title_font_size,
        col=1,
    )
    return fig


def fig_output(fig: go.Figure(), name: str) -> None:
    fig.update_layout(
        margin=dict(t=1, b=1, l=1, r=1),
    )
    if FOR_RESUME:
        fig.write_image(name + ".jpg")
    else:
        fig.write_html(name + ".html", include_mathjax="cdn")

    if SVG_ENABLE:
        fig.write_image(name + ".svg")
