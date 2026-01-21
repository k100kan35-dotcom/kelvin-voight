#!/usr/bin/env python3
"""
Viscoelastic Rubber Compound Modeler v4.0
Literature-based Collocation Method with NNLS optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.font_manager as fm
import tkinter as tk
from tkinter import ttk, Frame, Label, Entry, Button, Text, Scrollbar, messagebox
from scipy.optimize import differential_evolution, nnls
from scipy.interpolate import UnivariateSpline
import threading


# Font setup with subscript support
def setup_korean_font():
    try:
        # Use DejaVu Sans first (supports subscripts), fallback to Korean fonts
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        # Check for DejaVu Sans (supports subscripts)
        if 'DejaVu Sans' in available_fonts:
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
            return 'DejaVu Sans'

        # Fallback to Korean fonts
        korean_fonts = ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'Noto Sans KR']
        for font in korean_fonts:
            if font in available_fonts:
                plt.rcParams['font.family'] = font
                plt.rcParams['axes.unicode_minus'] = False
                return font

        # Last resort
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        return 'sans-serif'
    except:
        return 'TkDefaultFont'


def get_korean_font_for_tk():
    try:
        import tkinter.font as tkfont
        for font in ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'Noto Sans KR']:
            if font in tkfont.families():
                return font
        return 'TkDefaultFont'
    except:
        return 'TkDefaultFont'


setup_korean_font()


class WLF_Equation:
    """WLF equation for TTS"""
    def __init__(self, C1=17.44, C2=51.6, Tg=-50, Tref=25):
        self.C1 = C1
        self.C2 = C2
        self.Tg = Tg
        self.Tref = Tref

    def calculate_aT(self, T):
        log_aT = -self.C1 * (T - self.Tref) / (self.C2 + T - self.Tref)
        return 10**log_aT


class ViscoelasticModeler:
    """Generalized Maxwell model"""
    def __init__(self, E0, E_i, tau_i):
        self.E0 = E0
        self.E_i = np.array(E_i) if len(E_i) > 0 else np.array([])
        self.tau_i = np.array(tau_i) if len(tau_i) > 0 else np.array([])

    def storage_modulus(self, omega):
        E_prime = self.E0
        for E, tau in zip(self.E_i, self.tau_i):
            E_prime += E * (omega * tau)**2 / (1 + (omega * tau)**2)
        return E_prime

    def loss_modulus(self, omega):
        E_double = 0
        for E, tau in zip(self.E_i, self.tau_i):
            E_double += E * omega * tau / (1 + (omega * tau)**2)
        return E_double

    def tan_delta(self, omega):
        E_prime = self.storage_modulus(omega)
        E_double = self.loss_modulus(omega)
        return E_double / (E_prime + 1e-10)

    def element_contribution(self, omega, E_i, tau_i):
        E_prime_i = E_i * (omega * tau_i)**2 / (1 + (omega * tau_i)**2)
        E_double_i = E_i * omega * tau_i / (1 + (omega * tau_i)**2)
        return E_prime_i, E_double_i


class MaxwellDiagramCanvas:
    """Maxwell diagram drawer"""
    def __init__(self, fig, ax, n_elements):
        self.fig = fig
        self.ax = ax
        self.n_elements = n_elements

    def draw_spring(self, x, y, height, width=0.3, color='black'):
        n_coils = 10
        coil_y = np.linspace(y, y + height, n_coils * 2 + 1)
        coil_x = [x] + [x + (-1)**i * width/2 for i in range(1, len(coil_y)-1)] + [x]
        self.ax.plot(coil_x, coil_y, color=color, linewidth=2.5)

    def draw_dashpot(self, x, y, height, width=0.4, color='black'):
        cylinder_h = height * 0.55
        cylinder_y = y + height * 0.1
        self.ax.add_patch(Rectangle((x - width/2, cylinder_y), width, cylinder_h,
                                   facecolor='white', edgecolor=color, linewidth=2))
        piston_h = cylinder_h * 0.4
        piston_y = cylinder_y + cylinder_h * 0.3
        piston_w = width * 0.6
        self.ax.add_patch(Rectangle((x - piston_w/2, piston_y), piston_w, piston_h,
                                   facecolor='gray', edgecolor=color, linewidth=2))
        self.ax.plot([x, x], [y, cylinder_y], color=color, linewidth=2.5)
        self.ax.plot([x, x], [cylinder_y + cylinder_h, y + height], color=color, linewidth=2.5)

    def draw_maxwell_element(self, x, y_base, y_top, idx=None):
        total_height = y_top - y_base
        spring_height = total_height * 0.45
        dashpot_height = total_height * 0.45
        gap = total_height * 0.1

        mid_y1 = y_base + spring_height
        self.ax.plot([x, x], [y_base, y_base + 0.05], 'k-', linewidth=2.5)
        self.draw_spring(x, y_base + 0.05, spring_height - 0.05, width=0.25)
        self.ax.plot([x, x], [mid_y1, mid_y1 + gap], 'k-', linewidth=2.5)
        self.draw_dashpot(x, mid_y1 + gap, dashpot_height, width=0.35)
        self.ax.plot([x, x], [y_top - 0.05, y_top], 'k-', linewidth=2.5)

        if idx is not None:
            self.ax.text(x, y_base - 0.4, f'τ{idx}', ha='center', fontsize=11, weight='bold')

    def draw_model(self):
        self.ax.clear()
        n_show = min(self.n_elements, 20)
        spacing = 1.0
        total_width = max(8, n_show * spacing + 2)

        self.ax.set_xlim(-1, total_width)
        self.ax.set_ylim(-1, 5)
        self.ax.axis('off')
        self.ax.set_aspect('equal')

        y_base, y_top = 0.5, 4.0
        bar_start, bar_end = -0.5, total_width - 0.5
        self.ax.plot([bar_start, bar_end], [y_top, y_top], 'k-', linewidth=4)
        self.ax.plot([bar_start, bar_end], [y_base, y_base], 'k-', linewidth=4)

        for x_pos in [bar_start, bar_end]:
            self.ax.plot([x_pos, x_pos], [y_top, y_top + 0.3], 'k-', linewidth=4)
            self.ax.scatter([x_pos], [y_top + 0.3], s=100, c='black', marker='s')

        x_e0 = 0.5
        self.draw_spring(x_e0, y_base + 0.1, y_top - y_base - 0.2, width=0.35, color='red')
        self.ax.text(x_e0, y_base - 0.4, 'E₀', ha='center', fontsize=13, weight='bold', color='red')

        for i in range(n_show):
            x_elem = 2.0 + i * spacing
            self.draw_maxwell_element(x_elem, y_base, y_top, idx=i+1)

        if self.n_elements > 20:
            x_dots = 2.0 + 20 * spacing
            self.ax.text(x_dots, (y_base + y_top)/2, '...', ha='center', fontsize=24, weight='bold')

        title_y = y_top + 0.7
        self.ax.text(total_width/2, title_y, 'Generalized Maxwell Model',
                    ha='center', fontsize=14, weight='bold')


class ViscoelasticGUI:
    def __init__(self, master):
        self.master = master
        master.title("Viscoelastic Modeler v3.2 - Advanced Fitting")
        master.geometry("1600x900")

        # Parameters
        self.n_elements = 4
        self.max_elements = 20
        self.E0_default = 10.0
        self.E_i_default = [1000.0, 5000.0, 10000.0, 15000.0]
        self.tau_i_default = [1e-6, 1e-4, 1e-2, 1.0]
        self.freq_min = -10
        self.freq_max = 10

        # Raw data
        self.master_freq = None
        self.master_E_prime = None
        self.master_E_double = None
        self.master_temperature = 25

        # Smoothed data (for fitting)
        self.smooth_freq = None
        self.smooth_E_prime = None
        self.smooth_E_double = None

        # WLF
        self.wlf = WLF_Equation(C1=17.44, C2=51.6, Tg=-50, Tref=25)

        # Element management
        self.element_frames = []
        self.E_entries = []
        self.tau_entries = []

        # Fitting control
        self.fitting_history = []
        self.fitting_in_progress = False
        self.stop_fitting = False

        self.create_widgets()
        self.update_all()

    def _on_mousewheel(self, event, canvas):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_widgets(self):
        main_frame = Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel
        left_panel = Frame(main_frame, width=550, relief=tk.RIDGE, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        left_panel.pack_propagate(False)

        Label(left_panel, text="모델 파라미터", font=('Arial', 13, 'bold')).pack(pady=8)

        self.param_notebook = ttk.Notebook(left_panel)
        self.param_notebook.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        params_tab = Frame(self.param_notebook)
        data_tab = Frame(self.param_notebook)
        tts_tab = Frame(self.param_notebook)
        fitting_tab = Frame(self.param_notebook)
        guide_tab = Frame(self.param_notebook)

        self.param_notebook.add(params_tab, text="Parameters")
        self.param_notebook.add(data_tab, text="Data")
        self.param_notebook.add(tts_tab, text="TTS")
        self.param_notebook.add(fitting_tab, text="Fitting")
        self.param_notebook.add(guide_tab, text="Guide")

        self.create_parameters_tab(params_tab)
        self.create_data_tab(data_tab)
        self.create_tts_tab(tts_tab)
        self.create_fitting_tab(fitting_tab)
        self.create_guide_tab(guide_tab)

        # Right panel
        right_panel = Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Diagram
        diagram_frame = Frame(right_panel, relief=tk.RIDGE, borderwidth=2, height=250)
        diagram_frame.pack(fill=tk.X, pady=(0, 5))
        diagram_frame.pack_propagate(False)

        self.diagram_fig = Figure(figsize=(10, 2.5), dpi=100)
        self.diagram_ax = self.diagram_fig.add_subplot(111)
        self.diagram_canvas = FigureCanvasTkAgg(self.diagram_fig, master=diagram_frame)
        self.diagram_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Plot
        plot_frame = Frame(right_panel, relief=tk.RIDGE, borderwidth=2)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.plot_fig = Figure(figsize=(10, 6), dpi=100)
        self.plot_ax = self.plot_fig.add_subplot(111)
        self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_parameters_tab(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=540)
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_enter(event):
            canvas.bind_all("<MouseWheel>", lambda e: self._on_mousewheel(e, canvas))
        def _on_leave(event):
            canvas.unbind_all("<MouseWheel>")
        canvas.bind("<Enter>", _on_enter)
        canvas.bind("<Leave>", _on_leave)

        # E0
        frame1 = Frame(scroll_frame, relief=tk.GROOVE, bd=1, bg='#f8f8f8')
        frame1.pack(fill=tk.X, padx=5, pady=3)
        Label(frame1, text="E₀ (MPa):", font=('Arial', 10, 'bold'), bg='#f8f8f8').pack(side=tk.LEFT, padx=5)
        self.E0_entry = Entry(frame1, width=15)
        self.E0_entry.insert(0, str(self.E0_default))
        self.E0_entry.pack(side=tk.LEFT, padx=5, pady=3)

        # Frequency
        frame2 = Frame(scroll_frame, relief=tk.GROOVE, bd=1, bg='#f8f8f8')
        frame2.pack(fill=tk.X, padx=5, pady=3)
        Label(frame2, text="주파수 범위 (log₁₀ Hz):", font=('Arial', 10, 'bold'), bg='#f8f8f8').pack(side=tk.LEFT, padx=5)
        Label(frame2, text="Min:", bg='#f8f8f8').pack(side=tk.LEFT, padx=2)
        self.freq_min_entry = Entry(frame2, width=6)
        self.freq_min_entry.insert(0, str(self.freq_min))
        self.freq_min_entry.pack(side=tk.LEFT, padx=2)
        Label(frame2, text="Max:", bg='#f8f8f8').pack(side=tk.LEFT, padx=2)
        self.freq_max_entry = Entry(frame2, width=6)
        self.freq_max_entry.insert(0, str(self.freq_max))
        self.freq_max_entry.pack(side=tk.LEFT, padx=2, pady=3)

        # Elements
        frame3 = Frame(scroll_frame, relief=tk.GROOVE, bd=1, bg='#e8f4f8')
        frame3.pack(fill=tk.X, padx=5, pady=3)
        Label(frame3, text="Maxwell 요소 개수:", font=('Arial', 10, 'bold'), bg='#e8f4f8').pack(side=tk.LEFT, padx=5)

        self.n_elem_var = tk.StringVar(value=str(self.n_elements))
        elem_spinbox = ttk.Spinbox(frame3, from_=1, to=self.max_elements,
                                   textvariable=self.n_elem_var, width=8)
        elem_spinbox.pack(side=tk.LEFT, padx=3)

        Button(frame3, text="적용", command=self.update_element_count,
               bg='#2196F3', fg='white', font=('Arial', 8, 'bold'), width=6).pack(side=tk.LEFT, padx=3, pady=3)

        Button(frame3, text="삭제", command=self.clear_all_elements,
               bg='#f44336', fg='white', font=('Arial', 8, 'bold'), width=6).pack(side=tk.LEFT, padx=3, pady=3)

        # Elements
        Label(scroll_frame, text="Maxwell 요소:", font=('Arial', 10, 'bold')).pack(pady=(8, 3))

        self.elements_outer_container = Frame(scroll_frame)
        self.elements_outer_container.pack(fill=tk.X, padx=5)
        self.create_element_entries()

        # Options
        opt_frame = Frame(scroll_frame)
        opt_frame.pack(pady=5)

        self.show_contributions_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opt_frame, text="요소별 기여도", variable=self.show_contributions_var,
                      font=('Arial', 8), command=self.update_all).pack(side=tk.LEFT, padx=3)

        self.show_tan_delta_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_frame, text="tan δ 표시", variable=self.show_tan_delta_var,
                      font=('Arial', 8), command=self.update_all).pack(side=tk.LEFT, padx=3)

        # Buttons
        btn_frame = Frame(scroll_frame)
        btn_frame.pack(pady=8)

        Button(btn_frame, text="그래프 업데이트", command=self.update_all,
               bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'), width=15).pack(side=tk.LEFT, padx=3)

        Button(btn_frame, text="초기화", command=self.reset_parameters,
               bg='#f44336', fg='white', font=('Arial', 10, 'bold'), width=10).pack(side=tk.LEFT, padx=3)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_data_tab(self, parent):
        Label(parent, text="점탄성 데이터 입력", font=('Arial', 11, 'bold')).pack(pady=5)
        Label(parent, text="형식: frequency(Hz)  E'(MPa)  E\"(MPa)", font=('Arial', 9)).pack()

        text_frame = Frame(parent)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.data_text = Text(text_frame, height=20, width=50, font=('Courier', 9))
        data_scrollbar = Scrollbar(text_frame, command=self.data_text.yview)
        self.data_text.configure(yscrollcommand=data_scrollbar.set)
        self.data_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        data_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        sample = """# 예시 (삭제 후 붙여넣기):
0.001  10.2  1.5
0.01   15.5  3.8
0.1    25.2  8.1
"""
        self.data_text.insert('1.0', sample)

        btn_frame = Frame(parent)
        btn_frame.pack(pady=8)

        Button(btn_frame, text="데이터 로드", command=self.load_pasted_data,
               bg='#9C27B0', fg='white', font=('Arial', 10, 'bold'), width=12).pack(side=tk.LEFT, padx=3)

        Button(btn_frame, text="지우기", command=lambda: self.data_text.delete('1.0', tk.END),
               bg='#FF5722', fg='white', font=('Arial', 10, 'bold'), width=10).pack(side=tk.LEFT, padx=3)

        Button(btn_frame, text="Fit Model", command=self.auto_fit_model,
               bg='#E91E63', fg='white', font=('Arial', 10, 'bold'), width=12).pack(side=tk.LEFT, padx=3)

        self.stop_button = Button(btn_frame, text="중지", command=self.stop_fitting_process,
                                  bg='#f44336', fg='white', font=('Arial', 10, 'bold'), width=8, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=3)

    def create_tts_tab(self, parent):
        Label(parent, text="시간-온도 중첩 (TTS)", font=('Arial', 12, 'bold')).pack(pady=10)

        wlf_frame = tk.LabelFrame(parent, text="WLF 방정식 파라미터", font=('Arial', 10, 'bold'))
        wlf_frame.pack(fill=tk.X, padx=10, pady=5)

        frame1 = Frame(wlf_frame)
        frame1.pack(fill=tk.X, padx=5, pady=3)
        Label(frame1, text="Tg (유리전이온도, °C):", width=25, anchor='w').pack(side=tk.LEFT)
        self.tg_entry = Entry(frame1, width=12)
        self.tg_entry.insert(0, str(self.wlf.Tg))
        self.tg_entry.pack(side=tk.LEFT, padx=5)

        frame2 = Frame(wlf_frame)
        frame2.pack(fill=tk.X, padx=5, pady=3)
        Label(frame2, text="Tref (참조온도, °C):", width=25, anchor='w').pack(side=tk.LEFT)
        self.tref_entry = Entry(frame2, width=12)
        self.tref_entry.insert(0, str(self.wlf.Tref))
        self.tref_entry.pack(side=tk.LEFT, padx=5)

        frame3 = Frame(wlf_frame)
        frame3.pack(fill=tk.X, padx=5, pady=3)
        Label(frame3, text="C₁ (WLF 상수):", width=25, anchor='w').pack(side=tk.LEFT)
        self.c1_entry = Entry(frame3, width=12)
        self.c1_entry.insert(0, str(self.wlf.C1))
        self.c1_entry.pack(side=tk.LEFT, padx=5)

        frame4 = Frame(wlf_frame)
        frame4.pack(fill=tk.X, padx=5, pady=3)
        Label(frame4, text="C₂ (WLF 상수):", width=25, anchor='w').pack(side=tk.LEFT)
        self.c2_entry = Entry(frame4, width=12)
        self.c2_entry.insert(0, str(self.wlf.C2))
        self.c2_entry.pack(side=tk.LEFT, padx=5)

        Button(wlf_frame, text="WLF 파라미터 업데이트", command=self.update_wlf_parameters,
               bg='#2196F3', fg='white', font=('Arial', 9, 'bold')).pack(pady=5)

        shift_frame = tk.LabelFrame(parent, text="온도 Shift", font=('Arial', 10, 'bold'))
        shift_frame.pack(fill=tk.X, padx=10, pady=10)

        frame5 = Frame(shift_frame)
        frame5.pack(fill=tk.X, padx=5, pady=5)
        Label(frame5, text="측정 온도 (°C):", width=25, anchor='w').pack(side=tk.LEFT)
        self.meas_temp_entry = Entry(frame5, width=12)
        self.meas_temp_entry.insert(0, "25")
        self.meas_temp_entry.pack(side=tk.LEFT, padx=5)

        frame6 = Frame(shift_frame)
        frame6.pack(fill=tk.X, padx=5, pady=5)
        Label(frame6, text="목표 온도 (°C):", width=25, anchor='w').pack(side=tk.LEFT)
        self.target_temp_entry = Entry(frame6, width=12)
        self.target_temp_entry.insert(0, "25")
        self.target_temp_entry.pack(side=tk.LEFT, padx=5)

        Button(shift_frame, text="TTS 적용", command=self.apply_tts,
               bg='#FF9800', fg='white', font=('Arial', 10, 'bold')).pack(pady=5)

    def create_fitting_tab(self, parent):
        Label(parent, text="피팅 진행상황", font=('Arial', 12, 'bold')).pack(pady=5)

        info_frame = Frame(parent)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fitting_info_text = Text(info_frame, height=12, width=50, font=('Courier', 8))
        fitting_scrollbar = Scrollbar(info_frame, command=self.fitting_info_text.yview)
        self.fitting_info_text.configure(yscrollcommand=fitting_scrollbar.set)
        self.fitting_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        fitting_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        initial_msg = """데이터 로드 후 "Fit Model" 클릭
Collocation Method로 자동 피팅
요소 개수 자동 선택 (decade당 4개)
"""
        self.fitting_info_text.insert('1.0', initial_msg)
        self.fitting_info_text.config(state=tk.DISABLED)

        conv_frame = Frame(parent)
        conv_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.convergence_fig = Figure(figsize=(7, 3.5), dpi=80)
        self.convergence_ax = self.convergence_fig.add_subplot(111)
        self.convergence_canvas = FigureCanvasTkAgg(self.convergence_fig, master=conv_frame)
        self.convergence_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.convergence_ax.set_xlabel('Iteration', fontsize=9, weight='bold')
        self.convergence_ax.set_ylabel('Error', fontsize=9, weight='bold')
        self.convergence_ax.set_title('수렴 과정', fontsize=10, weight='bold')
        self.convergence_ax.grid(True, alpha=0.3)
        self.convergence_fig.tight_layout()

    def create_guide_tab(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        guide_frame = Frame(canvas)

        guide_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=guide_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_enter(event):
            canvas.bind_all("<MouseWheel>", lambda e: self._on_mousewheel(e, canvas))
        def _on_leave(event):
            canvas.unbind_all("<MouseWheel>")
        canvas.bind("<Enter>", _on_enter)
        canvas.bind("<Leave>", _on_leave)

        guide_text = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 v3.2 개선사항
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

자동 피팅 프로세스:
1. 데이터 스무딩 (스플라인 보간)
2. 최적 요소 개수 자동 선택 (2-8개)
3. 스무딩된 데이터에 피팅
4. 결과 표시

사용법:
1. Data 탭 → 데이터 붙여넣기
2. "데이터 로드" 클릭
3. "Fit Model" 클릭 → 자동 피팅!

버튼:
• 삭제: 모든 요소 지우기
• Fit Model: Maxwell 모델 피팅 (자동 요소 선택)
• 중지: 피팅 중단

v4.0 새로운 알고리즘 (문헌 기반):
• Collocation Method (표준 방법)
• NNLS (Non-negative Least Squares)
• 자동 요소 개수 선택 (각 decade당 4개)
• Log-spaced relaxation times (고정)
• 매우 빠름 (< 1초)
• 높은 정확도 (R² > 0.99 목표)

참고 문헌:
- NREL pyvisco library
- MDPI Applied Sciences (2018)
- Springer Materials & Structures (2019)
"""

        korean_font = get_korean_font_for_tk()
        text_widget = Text(guide_frame, wrap=tk.WORD, font=(korean_font, 9),
                          bg='#f5f5f5', padx=10, pady=10, height=30)
        text_widget.insert('1.0', guide_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_element_entries(self):
        for widget in self.elements_outer_container.winfo_children():
            widget.destroy()

        self.element_frames.clear()
        self.E_entries.clear()
        self.tau_entries.clear()

        col1 = Frame(self.elements_outer_container)
        col2 = Frame(self.elements_outer_container)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1)
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1)

        for i in range(self.n_elements):
            parent = col1 if i < 10 else col2

            elem_frame = Frame(parent, relief=tk.RIDGE, bd=1, bg='#fafafa')
            elem_frame.pack(fill=tk.X, pady=1)

            Label(elem_frame, text=f"#{i+1}", font=('Arial', 8, 'bold'), bg='#fafafa', width=3).grid(
                row=0, column=0, rowspan=2)

            Label(elem_frame, text=f"E{i+1}:", bg='#fafafa', font=('Arial', 7)).grid(row=0, column=1, sticky='e')
            E_entry = Entry(elem_frame, width=11, font=('Arial', 7))
            E_val = self.E_i_default[i] if i < len(self.E_i_default) else 1000.0
            E_entry.insert(0, f"{E_val:.1f}")
            E_entry.grid(row=0, column=2, padx=2, pady=1)
            self.E_entries.append(E_entry)

            Label(elem_frame, text=f"τ{i+1}:", bg='#fafafa', font=('Arial', 7)).grid(row=1, column=1, sticky='e')
            tau_entry = Entry(elem_frame, width=11, font=('Arial', 7))
            tau_val = self.tau_i_default[i] if i < len(self.tau_i_default) else 1e-3
            tau_entry.insert(0, f"{tau_val:.2e}")
            tau_entry.grid(row=1, column=2, padx=2, pady=1)
            self.tau_entries.append(tau_entry)

            self.element_frames.append(elem_frame)

    def clear_all_elements(self):
        """Clear all element values"""
        for E_entry in self.E_entries:
            E_entry.delete(0, tk.END)
            E_entry.insert(0, "0")
        for tau_entry in self.tau_entries:
            tau_entry.delete(0, tk.END)
            tau_entry.insert(0, "0.001")
        self.update_all()

    def update_element_count(self):
        try:
            new_n = int(self.n_elem_var.get())
            if 1 <= new_n <= self.max_elements:
                self.n_elements = new_n
                self.create_element_entries()
                self.update_all()
            else:
                messagebox.showerror("오류", f"요소 개수는 1-{self.max_elements} 사이")
        except ValueError:
            messagebox.showerror("오류", "잘못된 숫자")

    def load_pasted_data(self):
        """Load and smooth data"""
        try:
            text = self.data_text.get('1.0', tk.END)
            lines = [line.strip() for line in text.split('\n')
                    if line.strip() and not line.strip().startswith('#')]

            freq_list, E_prime_list, E_double_list = [], [], []

            for line in lines:
                parts = [x.strip() for x in line.replace(',', ' ').split()]
                if len(parts) >= 3:
                    freq_list.append(float(parts[0]))
                    E_prime_list.append(float(parts[1]))
                    E_double_list.append(float(parts[2]))

            if len(freq_list) > 0:
                # Store raw data
                self.master_freq = np.array(freq_list)
                self.master_E_prime = np.array(E_prime_list)
                self.master_E_double = np.array(E_double_list)

                # Create smoothed data using spline
                freq_log = np.log10(np.maximum(self.master_freq, 1e-10))
                E_prime_log = np.log10(np.maximum(self.master_E_prime, 1e-10))
                E_double_log = np.log10(np.maximum(self.master_E_double, 1e-10))

                # Sort by frequency
                sort_idx = np.argsort(freq_log)
                freq_log_sorted = freq_log[sort_idx]
                E_prime_log_sorted = E_prime_log[sort_idx]
                E_double_log_sorted = E_double_log[sort_idx]

                # Spline smoothing
                try:
                    # Create dense frequency grid
                    freq_log_dense = np.linspace(freq_log_sorted.min(), freq_log_sorted.max(), 500)

                    # Smooth E'
                    spl_prime = UnivariateSpline(freq_log_sorted, E_prime_log_sorted, s=len(freq_log)*0.5, k=3)
                    E_prime_smooth = spl_prime(freq_log_dense)

                    # Smooth E"
                    spl_double = UnivariateSpline(freq_log_sorted, E_double_log_sorted, s=len(freq_log)*0.5, k=3)
                    E_double_smooth = spl_double(freq_log_dense)

                    # Store smoothed data in linear scale
                    self.smooth_freq = 10**freq_log_dense
                    self.smooth_E_prime = 10**E_prime_smooth
                    self.smooth_E_double = 10**E_double_smooth

                    messagebox.showinfo("성공", f"{len(freq_list)}개 데이터 로드 + 스무딩 완료")
                except Exception as e:
                    # If smoothing fails, use raw data
                    self.smooth_freq = self.master_freq
                    self.smooth_E_prime = self.master_E_prime
                    self.smooth_E_double = self.master_E_double
                    messagebox.showinfo("성공", f"{len(freq_list)}개 데이터 로드 (스무딩 실패)")

                self.update_modulus_plot()
            else:
                messagebox.showerror("오류", "유효한 데이터 없음")

        except Exception as e:
            messagebox.showerror("오류", f"데이터 로드 실패: {e}")

    def auto_fit_model(self):
        """Fit model directly (no auto-selection)"""
        if self.master_freq is None:
            messagebox.showerror("오류", "먼저 데이터를 로드하세요")
            return

        if self.fitting_in_progress:
            messagebox.showwarning("경고", "피팅이 이미 진행 중")
            return

        # Switch to fitting tab
        self.param_notebook.select(3)

        # Start fitting thread directly
        fit_thread = threading.Thread(target=self.auto_fit_process, daemon=True)
        fit_thread.start()

    def auto_fit_process(self):
        """Direct fitting without auto-selection"""
        try:
            self.stop_fitting = False
            self.stop_button.config(state=tk.NORMAL)

            # Direct fitting (no element selection step)
            self.update_fitting_info("=" * 50 + "\n")
            self.update_fitting_info("Maxwell 모델 피팅 시작\n")
            self.update_fitting_info("=" * 50 + "\n\n")

            self.fit_to_master_curve()

        except Exception as e:
            import traceback
            error_msg = f"\n오류: {str(e)}\n{traceback.format_exc()}\n"
            self.update_fitting_info(error_msg)
            self.master.after(0, lambda: messagebox.showerror("오류", str(e)))
        finally:
            self.stop_button.config(state=tk.DISABLED)

    def update_wlf_parameters(self):
        try:
            self.wlf.Tg = float(self.tg_entry.get())
            self.wlf.Tref = float(self.tref_entry.get())
            self.wlf.C1 = float(self.c1_entry.get())
            self.wlf.C2 = float(self.c2_entry.get())
            messagebox.showinfo("성공", "WLF 파라미터 업데이트")
        except ValueError:
            messagebox.showerror("오류", "잘못된 값")

    def apply_tts(self):
        if self.master_freq is None:
            messagebox.showerror("오류", "먼저 데이터 로드")
            return

        try:
            meas_temp = float(self.meas_temp_entry.get())
            target_temp = float(self.target_temp_entry.get())

            aT_meas = self.wlf.calculate_aT(meas_temp)
            aT_target = self.wlf.calculate_aT(target_temp)
            aT_total = aT_target / aT_meas

            self.master_freq = self.master_freq * aT_total
            if self.smooth_freq is not None:
                self.smooth_freq = self.smooth_freq * aT_total
            self.master_temperature = target_temp

            self.update_modulus_plot()
            messagebox.showinfo("TTS 완료", f"Shift factor: {aT_total:.4e}")

        except ValueError:
            messagebox.showerror("오류", "잘못된 온도")

    def get_parameters(self):
        try:
            E0 = float(self.E0_entry.get())
            E_i = [float(entry.get()) for entry in self.E_entries]
            tau_i = [float(entry.get()) for entry in self.tau_entries]
            freq_min = float(self.freq_min_entry.get())
            freq_max = float(self.freq_max_entry.get())
            return E0, E_i, tau_i, freq_min, freq_max
        except ValueError as e:
            messagebox.showerror("오류", f"잘못된 파라미터: {e}")
            return None

    def set_parameters(self, E0, E_i, tau_i):
        self.E0_entry.delete(0, tk.END)
        self.E0_entry.insert(0, f"{E0:.2f}")

        for i, (E_entry, tau_entry) in enumerate(zip(self.E_entries, self.tau_entries)):
            if i < len(E_i):
                E_entry.delete(0, tk.END)
                E_entry.insert(0, f"{E_i[i]:.2e}")

                tau_entry.delete(0, tk.END)
                tau_entry.insert(0, f"{tau_i[i]:.2e}")

    def reset_parameters(self):
        self.n_elem_var.set(str(4))
        self.n_elements = 4
        self.create_element_entries()

        self.E0_entry.delete(0, tk.END)
        self.E0_entry.insert(0, str(self.E0_default))

        self.freq_min_entry.delete(0, tk.END)
        self.freq_min_entry.insert(0, str(self.freq_min))

        self.freq_max_entry.delete(0, tk.END)
        self.freq_max_entry.insert(0, str(self.freq_max))

        self.update_all()

    def stop_fitting_process(self):
        self.stop_fitting = True
        self.stop_button.config(state=tk.DISABLED)
        self.update_fitting_info("\n>>> 사용자가 중지 <<<\n")

    def fit_to_master_curve(self):
        """Fit using Collocation Method (literature-based)"""
        try:
            self.fitting_in_progress = True
            self.fitting_history = []

            # Use smoothed data if available
            freq = self.smooth_freq if self.smooth_freq is not None else self.master_freq
            E_prime_data = self.smooth_E_prime if self.smooth_E_prime is not None else self.master_E_prime
            E_double_data = self.smooth_E_double if self.smooth_E_double is not None else self.master_E_double

            omega = 2 * np.pi * freq

            # Auto-determine number of elements based on frequency range (literature method)
            freq_min_val = np.min(freq)
            freq_max_val = np.max(freq)
            n_decades = np.log10(freq_max_val / freq_min_val)
            elements_per_decade = 4  # Standard from literature
            n_elem = max(int(np.ceil(n_decades * elements_per_decade)), 3)

            # Update GUI with auto-determined element count
            self.master.after(0, lambda: self.n_elem_var.set(str(n_elem)))
            self.master.after(0, lambda: setattr(self, 'n_elements', n_elem))
            self.master.after(0, self.create_element_entries)

            self.update_fitting_info(f"주파수 범위: {n_decades:.2f} decades\n")
            self.update_fitting_info(f"자동 선택: {n_elem}개 요소 ({elements_per_decade}/decade)\n")
            self.update_fitting_info(f"문헌 기반 Collocation Method 사용\n\n")

            # STEP 1: Fix relaxation times (log-spaced) - Standard collocation method
            tau_i = np.logspace(
                np.log10(1.0 / (2 * np.pi * freq_max_val)),
                np.log10(1.0 / (2 * np.pi * freq_min_val)),
                n_elem
            )

            self.update_fitting_info(f"Relaxation times (고정):\n")
            for i, tau in enumerate(tau_i):
                self.update_fitting_info(f"  τ{i+1} = {tau:.3e} s\n")

            # STEP 2: Build linear system for E_0 and E_i (NNLS)
            # For storage modulus E'(ω) = E_0 + Σ E_i * (ω*τ_i)^2 / (1 + (ω*τ_i)^2)
            # For loss modulus E''(ω) = Σ E_i * ω*τ_i / (1 + (ω*τ_i)^2)

            n_freq = len(omega)

            # Build matrix A for storage modulus
            A_prime = np.zeros((n_freq, n_elem + 1))
            A_prime[:, 0] = 1.0  # E_0 contribution (constant)
            for i in range(n_elem):
                A_prime[:, i+1] = (omega * tau_i[i])**2 / (1 + (omega * tau_i[i])**2)

            # Build matrix A for loss modulus
            A_double = np.zeros((n_freq, n_elem + 1))
            A_double[:, 0] = 0.0  # E_0 doesn't contribute to E''
            for i in range(n_elem):
                A_double[:, i+1] = omega * tau_i[i] / (1 + (omega * tau_i[i])**2)

            # Combine both (weight loss modulus more)
            weight_loss = 1.5
            A_combined = np.vstack([A_prime, weight_loss * A_double])
            b_combined = np.concatenate([E_prime_data, weight_loss * E_double_data])

            self.update_fitting_info(f"\n선형 시스템 구성:\n")
            self.update_fitting_info(f"  행렬 크기: {A_combined.shape}\n")
            self.update_fitting_info(f"  Loss modulus 가중치: {weight_loss}\n")

            # STEP 3: Solve with NNLS (non-negative least squares)
            self.update_fitting_info(f"\nNNLS 최적화 시작...\n")

            E_params, residual = nnls(A_combined, b_combined)

            E0_fit = E_params[0]
            E_i_fit = E_params[1:]
            tau_i_fit = tau_i

            # Calculate final error
            model = ViscoelasticModeler(E0_fit, E_i_fit, tau_i_fit)
            E_prime_pred = np.array([model.storage_modulus(w) for w in omega])
            E_double_pred = np.array([model.loss_modulus(w) for w in omega])

            error_prime = np.sum((E_prime_pred - E_prime_data)**2)
            error_double = np.sum((E_double_pred - E_double_data)**2)
            total_error = error_prime + error_double

            # Calculate R²
            ss_res = total_error
            ss_tot_prime = np.sum((E_prime_data - np.mean(E_prime_data))**2)
            ss_tot_double = np.sum((E_double_data - np.mean(E_double_data))**2)
            R2 = 1 - ss_res / (ss_tot_prime + ss_tot_double)

            if not self.stop_fitting:
                self.master.after(0, lambda: self.set_parameters(E0_fit, E_i_fit, tau_i_fit))
                self.master.after(0, self.update_all)

                result_msg = f"\n{'='*45}\n피팅 완료! (Collocation Method)\n{'='*45}\n"
                result_msg += f"NNLS Residual: {residual:.4e}\n"
                result_msg += f"Total Error: {total_error:.4e}\n"
                result_msg += f"R² = {R2:.6f}\n\n"
                result_msg += f"E₀ = {E0_fit:.2e} MPa\n"
                for i in range(n_elem):
                    result_msg += f"E{i+1}={E_i_fit[i]:.2e}, τ{i+1}={tau_i_fit[i]:.2e}\n"
                result_msg += f"{'='*45}\n"

                self.update_fitting_info(result_msg)
                self.master.after(0, lambda: messagebox.showinfo("완료",
                    f"피팅 완료!\nElements: {n_elem}\nR²: {R2:.4f}\nError: {total_error:.2e}"))

        except Exception as e:
            import traceback
            error_msg = f"\n오류:\n{str(e)}\n{traceback.format_exc()}\n"
            self.update_fitting_info(error_msg)
            self.master.after(0, lambda: messagebox.showerror("오류", str(e)))

        finally:
            self.fitting_in_progress = False
    def update_fitting_info(self, message):
        def _update():
            self.fitting_info_text.config(state=tk.NORMAL)
            self.fitting_info_text.insert(tk.END, message)
            self.fitting_info_text.see(tk.END)
            self.fitting_info_text.config(state=tk.DISABLED)

        if threading.current_thread() == threading.main_thread():
            _update()
        else:
            self.master.after(0, _update)

    def update_convergence_plot(self):
        if len(self.fitting_history) < 2:
            return

        self.convergence_ax.clear()
        iterations = np.arange(1, len(self.fitting_history) + 1) * 50
        self.convergence_ax.plot(iterations, self.fitting_history, 'b-', linewidth=2)
        self.convergence_ax.set_xlabel('Iteration', fontsize=9, weight='bold')
        self.convergence_ax.set_ylabel('Error', fontsize=9, weight='bold')
        self.convergence_ax.set_title('수렴 과정', fontsize=10, weight='bold')
        self.convergence_ax.set_yscale('log')
        self.convergence_ax.grid(True, alpha=0.3)
        self.convergence_fig.tight_layout()
        self.convergence_canvas.draw()

    def update_maxwell_diagram(self):
        diagram = MaxwellDiagramCanvas(self.diagram_fig, self.diagram_ax, self.n_elements)
        diagram.draw_model()
        self.diagram_fig.tight_layout()
        self.diagram_canvas.draw()

    def update_modulus_plot(self):
        """Update plot - FIX: Clear axis properly"""
        params = self.get_parameters()
        if params is None:
            return

        E0, E_i, tau_i, freq_min, freq_max = params
        model = ViscoelasticModeler(E0, E_i, tau_i)

        freq_log = np.linspace(freq_min, freq_max, 1000)
        freq = 10**freq_log
        omega = 2 * np.pi * freq

        E_prime = np.array([model.storage_modulus(w) for w in omega])
        E_double = np.array([model.loss_modulus(w) for w in omega])
        tan_delta = np.array([model.tan_delta(w) for w in omega])

        # CRITICAL: Clear both axes properly
        self.plot_ax.clear()
        # Remove any twin axes
        for ax in self.plot_fig.get_axes()[1:]:
            self.plot_fig.delaxes(ax)

        # Element contributions
        if self.show_contributions_var.get() and len(E_i) > 0:
            colors = plt.cm.tab10(np.linspace(0, 1, len(E_i)))
            for i, (E, tau, color) in enumerate(zip(E_i, tau_i, colors)):
                E_p_i = np.array([model.element_contribution(w, E, tau)[0] for w in omega])
                E_d_i = np.array([model.element_contribution(w, E, tau)[1] for w in omega])
                self.plot_ax.plot(freq_log, np.log10(E_p_i + 1e-10), '--',
                                color=color, linewidth=1, alpha=0.5)
                self.plot_ax.plot(freq_log, np.log10(E_d_i + 1e-10), ':',
                                color=color, linewidth=1, alpha=0.5)

        # Total moduli
        self.plot_ax.plot(freq_log, np.log10(E_prime), 'r-', linewidth=2.5, label="E' (model)")
        self.plot_ax.plot(freq_log, np.log10(E_double), 'g-', linewidth=2.5, label='E" (model)')

        # Smoothed data (if available)
        if self.smooth_freq is not None:
            freq_log_smooth = np.log10(np.maximum(self.smooth_freq, 1e-10))
            E_p_smooth = np.log10(np.maximum(self.smooth_E_prime, 1e-10))
            E_d_smooth = np.log10(np.maximum(self.smooth_E_double, 1e-10))
            self.plot_ax.plot(freq_log_smooth, E_p_smooth, 'r--', linewidth=1.5, alpha=0.6, label="E' (smoothed)")
            self.plot_ax.plot(freq_log_smooth, E_d_smooth, 'g--', linewidth=1.5, alpha=0.6, label='E" (smoothed)')

        # Raw data
        if self.master_freq is not None:
            freq_log_data = np.log10(np.maximum(self.master_freq, 1e-10))
            E_p_log = np.log10(np.maximum(self.master_E_prime, 1e-10))
            E_d_log = np.log10(np.maximum(self.master_E_double, 1e-10))

            self.plot_ax.scatter(freq_log_data, E_p_log, c='darkred', marker='o',
                               s=20, alpha=0.4, label="E' (raw)", zorder=10)
            self.plot_ax.scatter(freq_log_data, E_d_log, c='darkgreen', marker='s',
                               s=20, alpha=0.4, label='E" (raw)', zorder=10)

        # tan delta
        if self.show_tan_delta_var.get():
            ax2 = self.plot_ax.twinx()
            ax2.plot(freq_log, tan_delta, 'b-', linewidth=2, alpha=0.7, label='tan δ')

            tg_idx = np.argmax(tan_delta)
            ax2.plot(freq_log[tg_idx], tan_delta[tg_idx], 'b*', markersize=12,
                    label=f'Tg@{freq_log[tg_idx]:.1f}')

            ax2.set_ylabel('tan δ', fontsize=11, weight='bold', color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            ax2.legend(fontsize=8, loc='upper right')
            ax2.set_ylim(0, max(tan_delta) * 1.2)

        self.plot_ax.set_xlabel('log10 f (Hz)', fontsize=11, weight='bold')
        self.plot_ax.set_ylabel('log10 E (MPa)', fontsize=11, weight='bold')
        self.plot_ax.set_title('Complex Modulus', fontsize=12, weight='bold')
        self.plot_ax.set_ylim(-1, None)
        self.plot_ax.legend(fontsize=7, loc='upper left', ncol=2)
        self.plot_ax.grid(True, alpha=0.3)

        self.plot_fig.tight_layout()
        self.plot_canvas.draw()

    def update_all(self):
        self.update_maxwell_diagram()
        self.update_modulus_plot()


def main():
    root = tk.Tk()
    app = ViscoelasticGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
