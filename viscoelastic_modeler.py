#!/usr/bin/env python3
"""
Viscoelastic Rubber Compound Modeler v3.0
Generalized Maxwell Model for Complex Modulus Calculation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.font_manager as fm
import tkinter as tk
from tkinter import ttk, Frame, Label, Entry, Button, Text, Scrollbar, messagebox
from scipy.optimize import differential_evolution
import threading


# Configure Korean font for matplotlib and get font for Tkinter
def setup_korean_font():
    """Setup Korean font for matplotlib to prevent encoding issues"""
    try:
        korean_fonts = ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'Noto Sans KR', 'Noto Sans CJK KR']
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        # Find first available Korean font
        selected_font = None
        for font in korean_fonts:
            if font in available_fonts:
                selected_font = font
                plt.rcParams['font.family'] = font
                plt.rcParams['axes.unicode_minus'] = False
                break

        if selected_font is None:
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
            selected_font = 'DejaVu Sans'

        return selected_font
    except:
        plt.rcParams['axes.unicode_minus'] = False
        return 'TkDefaultFont'


def get_korean_font_for_tk():
    """Get available Korean font for Tkinter widgets"""
    try:
        import tkinter.font as tkfont
        available_tk_fonts = tkfont.families()

        # Try to find Korean fonts in order of preference
        korean_fonts = ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'Noto Sans KR',
                       'Noto Sans CJK KR', 'NanumBarunGothic', 'D2Coding']

        for font in korean_fonts:
            if font in available_tk_fonts:
                return font

        # Fallback to system default
        return 'TkDefaultFont'
    except:
        return 'TkDefaultFont'


setup_korean_font()


class ViscoelasticModeler:
    """Generalized Maxwell model for viscoelastic materials."""

    def __init__(self, E0, E_i, tau_i):
        self.E0 = E0
        self.E_i = np.array(E_i) if len(E_i) > 0 else np.array([])
        self.tau_i = np.array(tau_i) if len(tau_i) > 0 else np.array([])

    def storage_modulus(self, omega):
        """Calculate storage modulus E'(ω)"""
        E_prime = self.E0
        for E, tau in zip(self.E_i, self.tau_i):
            E_prime += E * (omega * tau)**2 / (1 + (omega * tau)**2)
        return E_prime

    def loss_modulus(self, omega):
        """Calculate loss modulus E"(ω)"""
        E_double_prime = 0
        for E, tau in zip(self.E_i, self.tau_i):
            E_double_prime += E * omega * tau / (1 + (omega * tau)**2)
        return E_double_prime

    def tan_delta(self, omega):
        """Calculate tan δ = E"/E'"""
        E_prime = self.storage_modulus(omega)
        E_double = self.loss_modulus(omega)
        return E_double / (E_prime + 1e-10)  # Avoid division by zero

    def element_contribution(self, omega, E_i, tau_i):
        """Calculate individual element contribution to E' and E\""""
        E_prime_i = E_i * (omega * tau_i)**2 / (1 + (omega * tau_i)**2)
        E_double_i = E_i * omega * tau_i / (1 + (omega * tau_i)**2)
        return E_prime_i, E_double_i

    def complex_modulus(self, omega):
        """Calculate complex modulus E*(ω) = E'(ω) + i·E"(ω)"""
        return self.storage_modulus(omega), self.loss_modulus(omega)


class MaxwellDiagramCanvas:
    """Canvas for drawing Generalized Maxwell model diagram"""

    def __init__(self, fig, ax, n_elements):
        self.fig = fig
        self.ax = ax
        self.n_elements = n_elements

    def draw_spring(self, x, y, height, width=0.3, color='black'):
        """Draw a spring symbol"""
        n_coils = 10
        coil_y = np.linspace(y, y + height, n_coils * 2 + 1)
        coil_x = [x] + [x + (-1)**i * width/2 for i in range(1, len(coil_y)-1)] + [x]
        self.ax.plot(coil_x, coil_y, color=color, linewidth=2.5)

    def draw_dashpot(self, x, y, height, width=0.4, color='black'):
        """Draw a dashpot (damper) symbol"""
        # Outer cylinder (rectangular)
        cylinder_h = height * 0.55
        cylinder_y = y + height * 0.1
        self.ax.add_patch(Rectangle((x - width/2, cylinder_y), width, cylinder_h,
                                   facecolor='white', edgecolor=color, linewidth=2))

        # Inner piston (smaller rectangle, filled)
        piston_h = cylinder_h * 0.4
        piston_y = cylinder_y + cylinder_h * 0.3
        piston_w = width * 0.6
        self.ax.add_patch(Rectangle((x - piston_w/2, piston_y), piston_w, piston_h,
                                   facecolor='gray', edgecolor=color, linewidth=2))

        # Rod extending from bottom
        rod_bottom_y = y
        self.ax.plot([x, x], [rod_bottom_y, cylinder_y], color=color, linewidth=2.5)

        # Rod extending from top
        rod_top_y = cylinder_y + cylinder_h
        self.ax.plot([x, x], [rod_top_y, y + height], color=color, linewidth=2.5)

    def draw_maxwell_element(self, x, y_base, y_top, show_label=True, idx=None):
        """Draw a Maxwell element (spring and dashpot in series)"""
        total_height = y_top - y_base
        spring_height = total_height * 0.45
        dashpot_height = total_height * 0.45
        gap = total_height * 0.1

        # Connection from bottom
        mid_y1 = y_base + spring_height
        self.ax.plot([x, x], [y_base, y_base + 0.05], 'k-', linewidth=2.5)

        # Spring (bottom part)
        self.draw_spring(x, y_base + 0.05, spring_height - 0.05, width=0.25)

        # Connection between spring and dashpot
        self.ax.plot([x, x], [mid_y1, mid_y1 + gap], 'k-', linewidth=2.5)

        # Dashpot (top part)
        self.draw_dashpot(x, mid_y1 + gap, dashpot_height, width=0.35)

        # Connection to top
        self.ax.plot([x, x], [y_top - 0.05, y_top], 'k-', linewidth=2.5)

        # Label
        if show_label and idx is not None:
            self.ax.text(x, y_base - 0.4, f'τ{idx}', ha='center', fontsize=11, weight='bold')

    def draw_model(self):
        """Draw the complete Generalized Maxwell model"""
        self.ax.clear()

        # Calculate dimensions
        n_show = min(self.n_elements, 20)
        spacing = 1.0
        total_width = max(8, n_show * spacing + 2)

        self.ax.set_xlim(-1, total_width)
        self.ax.set_ylim(-1, 5)
        self.ax.axis('off')
        self.ax.set_aspect('equal')

        y_base = 0.5
        y_top = 4.0
        element_height = y_top - y_base

        # Draw top and bottom rigid bars
        bar_start = -0.5
        bar_end = total_width - 0.5
        self.ax.plot([bar_start, bar_end], [y_top, y_top], 'k-', linewidth=4)
        self.ax.plot([bar_start, bar_end], [y_base, y_base], 'k-', linewidth=4)

        # Add fixed support symbols at ends
        for x_pos in [bar_start, bar_end]:
            self.ax.plot([x_pos, x_pos], [y_top, y_top + 0.3], 'k-', linewidth=4)
            self.ax.scatter([x_pos], [y_top + 0.3], s=100, c='black', marker='s')

        # Draw E0 spring (leftmost)
        x_e0 = 0.5
        self.draw_spring(x_e0, y_base + 0.1, element_height - 0.2, width=0.35, color='red')
        self.ax.text(x_e0, y_base - 0.4, 'E₀', ha='center', fontsize=13, weight='bold', color='red')

        # Draw Maxwell elements
        for i in range(n_show):
            x_elem = 2.0 + i * spacing
            self.draw_maxwell_element(x_elem, y_base, y_top, show_label=True, idx=i+1)

        # Add "..." if more than 20 elements
        if self.n_elements > 20:
            x_dots = 2.0 + 20 * spacing
            self.ax.text(x_dots, (y_base + y_top)/2, '...', ha='center', fontsize=24, weight='bold')
            self.ax.text(x_dots, y_base - 0.4, f'({self.n_elements} total)', ha='center', fontsize=9)

        # Add title
        title_y = y_top + 0.7
        self.ax.text(total_width/2, title_y, 'Generalized Maxwell Model',
                    ha='center', fontsize=15, weight='bold')


class ViscoelasticGUI:
    def __init__(self, master):
        self.master = master
        master.title("Viscoelastic Rubber Compound Modeler v3.0 (Generalized Maxwell Model)")
        master.geometry("1700x950")

        # Default parameters
        self.n_elements = 4
        self.max_elements = 20
        self.E0_default = 10.0
        self.E_i_default = [1000.0, 5000.0, 10000.0, 15000.0]
        self.tau_i_default = [1e-6, 1e-4, 1e-2, 1.0]

        self.freq_min = -10
        self.freq_max = 10

        # Master curve data
        self.master_freq = None
        self.master_E_prime = None
        self.master_E_double = None

        # Tg parameter
        self.Tg_freq = None  # Frequency at Tg (tan δ peak)

        # Element management
        self.element_frames = []
        self.E_entries = []
        self.tau_entries = []

        # Fitting progress tracking
        self.fitting_history = []
        self.fitting_in_progress = False

        # Create GUI layout
        self.create_widgets()
        self.update_all()

    def _on_mousewheel(self, event, canvas):
        """Enable mouse wheel scrolling"""
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_widgets(self):
        # Main container
        main_frame = Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Parameters
        left_panel = Frame(main_frame, width=400, relief=tk.RIDGE, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

        # Title
        title_label = Label(left_panel, text="Model Parameters", font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)

        # Create notebook for tabs
        self.param_notebook = ttk.Notebook(left_panel)
        self.param_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Parameters
        params_tab = Frame(self.param_notebook)
        self.param_notebook.add(params_tab, text="Parameters")

        # Tab 2: Data Input
        data_tab = Frame(self.param_notebook)
        self.param_notebook.add(data_tab, text="Paste Data")

        # Tab 3: Fitting Progress
        fitting_tab = Frame(self.param_notebook)
        self.param_notebook.add(fitting_tab, text="Fitting Progress")

        # Tab 4: Physics Guide
        guide_tab = Frame(self.param_notebook)
        self.param_notebook.add(guide_tab, text="Physics Guide")

        # ===== PARAMETERS TAB =====
        self.create_parameters_tab(params_tab)

        # ===== DATA INPUT TAB =====
        self.create_data_tab(data_tab)

        # ===== FITTING PROGRESS TAB =====
        self.create_fitting_tab(fitting_tab)

        # ===== PHYSICS GUIDE TAB =====
        self.create_guide_tab(guide_tab)

        # Right panel - Plots (shared)
        right_panel = Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Top: Maxwell diagram
        diagram_frame = Frame(right_panel, relief=tk.RIDGE, borderwidth=2)
        diagram_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.diagram_fig = Figure(figsize=(12, 4), dpi=100)
        self.diagram_ax = self.diagram_fig.add_subplot(111)
        self.diagram_canvas = FigureCanvasTkAgg(self.diagram_fig, master=diagram_frame)
        self.diagram_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom: Modulus plot (shared across all tabs)
        plot_frame = Frame(right_panel, relief=tk.RIDGE, borderwidth=2)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.plot_fig = Figure(figsize=(12, 5), dpi=100)
        self.plot_ax = self.plot_fig.add_subplot(111)
        self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_parameters_tab(self, parent):
        """Create parameters tab with scrollable content"""
        # Scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.scrollable_frame = Frame(canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Bind mouse wheel to canvas
        canvas.bind_all("<MouseWheel>", lambda event: self._on_mousewheel(event, canvas))

        # E0 parameter
        Label(self.scrollable_frame, text="E₀ (MPa):", font=('Arial', 11, 'bold')).pack(pady=(10, 5))
        self.E0_entry = Entry(self.scrollable_frame, width=20)
        self.E0_entry.insert(0, str(self.E0_default))
        self.E0_entry.pack()

        # Frequency range
        Label(self.scrollable_frame, text="Frequency Range (log₁₀ Hz):", font=('Arial', 11, 'bold')).pack(pady=(20, 5))
        freq_frame = Frame(self.scrollable_frame)
        freq_frame.pack()
        Label(freq_frame, text="Min:").grid(row=0, column=0, padx=5)
        self.freq_min_entry = Entry(freq_frame, width=10)
        self.freq_min_entry.insert(0, str(self.freq_min))
        self.freq_min_entry.grid(row=0, column=1, padx=5)
        Label(freq_frame, text="Max:").grid(row=0, column=2, padx=5)
        self.freq_max_entry = Entry(freq_frame, width=10)
        self.freq_max_entry.insert(0, str(self.freq_max))
        self.freq_max_entry.grid(row=0, column=3, padx=5)

        # Tg parameter
        Label(self.scrollable_frame, text="Tg (Glass Transition):", font=('Arial', 11, 'bold')).pack(pady=(20, 5))
        tg_frame = Frame(self.scrollable_frame)
        tg_frame.pack()
        Label(tg_frame, text="log₁₀ f @ Tg (Hz):").grid(row=0, column=0, padx=5)
        self.tg_freq_entry = Entry(tg_frame, width=10)
        self.tg_freq_entry.insert(0, "0.0")
        self.tg_freq_entry.grid(row=0, column=1, padx=5)
        Button(tg_frame, text="Auto-adjust τ", command=self.adjust_tau_for_tg,
               bg='#FF9800', fg='white', font=('Arial', 9, 'bold')).grid(row=0, column=2, padx=5)

        # Number of elements control
        Label(self.scrollable_frame, text="Number of Maxwell Elements:", font=('Arial', 11, 'bold')).pack(pady=(20, 5))
        elem_control_frame = Frame(self.scrollable_frame)
        elem_control_frame.pack()

        self.n_elem_var = tk.StringVar(value=str(self.n_elements))
        elem_spinbox = ttk.Spinbox(elem_control_frame, from_=1, to=self.max_elements,
                                   textvariable=self.n_elem_var, width=10)
        elem_spinbox.pack(side=tk.LEFT, padx=5)

        Button(elem_control_frame, text="Apply", command=self.update_element_count,
               bg='#2196F3', fg='white', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)

        # Maxwell elements container (2 columns)
        Label(self.scrollable_frame, text="Maxwell Elements:", font=('Arial', 11, 'bold')).pack(pady=(20, 10))

        # Container frame that will be recreated
        self.elements_outer_container = Frame(self.scrollable_frame)
        self.elements_outer_container.pack(fill=tk.X, padx=5)

        # Create initial elements
        self.create_element_entries()

        # Show options
        options_frame = Frame(self.scrollable_frame)
        options_frame.pack(pady=10)

        self.show_contributions_var = tk.BooleanVar(value=False)
        contrib_check = tk.Checkbutton(options_frame, text="Show element contributions",
                                      variable=self.show_contributions_var, font=('Arial', 9))
        contrib_check.pack(side=tk.LEFT, padx=5)

        self.show_tan_delta_var = tk.BooleanVar(value=True)
        tan_delta_check = tk.Checkbutton(options_frame, text="Show tan δ",
                                        variable=self.show_tan_delta_var, font=('Arial', 9))
        tan_delta_check.pack(side=tk.LEFT, padx=5)

        # Control buttons
        Button(self.scrollable_frame, text="Update Plots", command=self.update_all,
               bg='#4CAF50', fg='white', font=('Arial', 12, 'bold'), pady=10).pack(pady=20, fill=tk.X, padx=20)

        Button(self.scrollable_frame, text="Reset", command=self.reset_parameters,
               bg='#ff9800', fg='white', font=('Arial', 10), pady=5).pack(pady=(0, 20), fill=tk.X, padx=20)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_data_tab(self, parent):
        """Create data input tab for pasting data"""
        Label(parent, text="Paste Viscoelastic Data", font=('Arial', 12, 'bold')).pack(pady=10)

        Label(parent, text="Format: frequency(Hz) E'(MPa) E\"(MPa) - one row per line",
              font=('Arial', 9)).pack(pady=5)

        # Text area for pasting data
        text_frame = Frame(parent)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.data_text = Text(text_frame, height=20, width=50)
        data_scrollbar = Scrollbar(text_frame, command=self.data_text.yview)
        self.data_text.configure(yscrollcommand=data_scrollbar.set)

        self.data_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        data_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Sample data
        sample_text = """# Example data (delete and paste your own):
# frequency(Hz), E'(MPa), E"(MPa)
0.001, 10.2, 1.5
0.01, 15.5, 3.8
0.1, 25.2, 8.1
1.0, 45.0, 12.5
"""
        self.data_text.insert('1.0', sample_text)

        # Buttons
        btn_frame = Frame(parent)
        btn_frame.pack(pady=10)

        Button(btn_frame, text="Load Pasted Data", command=self.load_pasted_data,
               bg='#9C27B0', fg='white', font=('Arial', 11, 'bold'), width=15).pack(side=tk.LEFT, padx=5)

        Button(btn_frame, text="Clear Data", command=lambda: self.data_text.delete('1.0', tk.END),
               bg='#FF5722', fg='white', font=('Arial', 11, 'bold'), width=15).pack(side=tk.LEFT, padx=5)

        Button(btn_frame, text="Fit Model", command=self.start_fitting,
               bg='#E91E63', fg='white', font=('Arial', 11, 'bold'), width=15).pack(side=tk.LEFT, padx=5)

    def create_fitting_tab(self, parent):
        """Create fitting progress tab"""
        Label(parent, text="피팅 진행상황 및 결과", font=('Arial', 14, 'bold')).pack(pady=10)

        # Fitting info text area
        info_frame = Frame(parent)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fitting_info_text = Text(info_frame, height=15, width=50, font=('Courier', 9))
        fitting_scrollbar = Scrollbar(info_frame, command=self.fitting_info_text.yview)
        self.fitting_info_text.configure(yscrollcommand=fitting_scrollbar.set)

        self.fitting_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        fitting_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Initial message
        initial_msg = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  피팅 진행상황
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

데이터를 로드하고 "Fit Model" 버튼을 클릭하면
피팅 과정이 여기에 표시됩니다.

피팅 정보:
• 사용된 Maxwell 요소 개수
• 초기 추정값
• 최적화 알고리즘 설정
• 반복 횟수 및 에러 변화
• 최종 파라미터 값
• 피팅 품질 평가

"""
        self.fitting_info_text.insert('1.0', initial_msg)
        self.fitting_info_text.config(state=tk.DISABLED)

        # Convergence plot
        conv_frame = Frame(parent)
        conv_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.convergence_fig = Figure(figsize=(8, 4), dpi=80)
        self.convergence_ax = self.convergence_fig.add_subplot(111)
        self.convergence_canvas = FigureCanvasTkAgg(self.convergence_fig, master=conv_frame)
        self.convergence_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize empty convergence plot
        self.convergence_ax.set_xlabel('Iteration', fontsize=10, weight='bold')
        self.convergence_ax.set_ylabel('Error', fontsize=10, weight='bold')
        self.convergence_ax.set_title('피팅 수렴 과정', fontsize=12, weight='bold')
        self.convergence_ax.grid(True, alpha=0.3)
        self.convergence_fig.tight_layout()

    def create_guide_tab(self, parent):
        """Create physics guide tab with Korean text and visual examples"""
        # Scrollable container
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        guide_frame = Frame(canvas)

        guide_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=guide_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind_all("<MouseWheel>", lambda event: self._on_mousewheel(event, canvas))

        # Korean content
        guide_text = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   물리 가이드: 파라미터 이해하기
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. E₀ (평형 탄성계수)
   • 물리적 의미: 장시간 탄성 반응
   • E'에 미치는 영향: E' 곡선 전체를 위아래로 이동
   • E"에 미치는 영향: 직접적인 영향 없음
   • 예시: E₀ 증가 → 전체적으로 높은 강성

2. Eᵢ (Maxwell 요소 탄성계수)
   • 물리적 의미: i번째 완화 모드의 강도
   • E'에 미치는 영향: 플래토 높이 증가
   • E"에 미치는 영향: 피크 높이 증가
   • 예시: Eᵢ 증가 → τᵢ에서 더 강한 완화

3. τᵢ (완화시간)
   • 물리적 의미: i번째 모드의 시간 스케일
   • E'에 미치는 영향: 전이 위치 이동
   • E"에 미치는 영향: 피크 위치 이동
   • 예시: τᵢ 증가 → 더 낮은 주파수에서 전이
   • 관계식: f(peak) ≈ 1/(2πτᵢ)

4. Tg (유리전이온도)
   • 물리적 의미: 고무-유리 전이가 일어나는 온도/주파수
   • 측정 방법: tan δ가 최대인 주파수
   • tau와의 관계: Tg 변경 시 tau 값들이 자동으로 조정됨

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   저장탄성계수 (E')
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• 의미: 탄성 에너지 저장
• 거동:
  - 저주파: E' ≈ E₀ (평형)
  - 고주파: E' ≈ E₀ + ΣEᵢ (유리상)
  - 전이: 부드러운 증가

• 조절 방법:
  ✓ E₀ 증가 → 전체 곡선 위로 이동
  ✓ Eᵢ 증가 → 더 가파른 전이
  ✓ τᵢ 감소 → 전이가 오른쪽으로 이동

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   손실탄성계수 (E")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• 의미: 에너지 소산
• 거동:
  - 저주파: E" ≈ 0 (소산 없음)
  - 피크 위치: f ≈ 1/(2πτᵢ)
  - 고주파: E" → 0 (동결)

• 조절 방법:
  ✓ Eᵢ 증가 → 더 높은 피크
  ✓ τᵢ 증가 → 피크가 왼쪽으로 이동
  ✓ 요소 추가 → 다중 피크

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   tan δ (손실계수)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• 정의: tan δ = E"/E'
• 물리적 의미: 에너지 소산/저장 비율
• 중요성:
  - Tg 결정: tan δ 피크 위치 = Tg
  - 감쇠 성능: tan δ 클수록 진동 감쇠 우수
  - 타이어 성능: 구름저항, 습윤노면 접착력

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   피팅 팁
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 대부분의 재료에 3-6개 Maxwell 요소 사용
2. τᵢ 값을 로그 스케일로 분포시키기
3. 먼저 시각적으로 파라미터 조정
4. "Fit Model"로 자동 최적화
5. E'와 E" 모두 잘 맞는지 확인
6. Fitting Progress 탭에서 수렴 과정 확인

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   파라미터 변경 효과 예시
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

아래 그래프를 참고하세요:
"""

        # Text widget with Korean font support
        korean_font = get_korean_font_for_tk()
        text_widget = Text(guide_frame, wrap=tk.WORD, font=(korean_font, 10),
                          bg='#f5f5f5', padx=15, pady=15, height=30)
        text_widget.insert('1.0', guide_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.X, padx=10, pady=10)

        # Example plots
        self.create_example_plots(guide_frame)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_example_plots(self, parent):
        """Create example plots showing parameter effects"""
        # Create matplotlib figure
        example_fig = Figure(figsize=(10, 8), dpi=80)

        # Example 1: E0 effect
        ax1 = example_fig.add_subplot(221)
        freq_log = np.linspace(-6, 6, 500)
        omega = 2 * np.pi * 10**freq_log

        for E0 in [10, 50, 100]:
            model = ViscoelasticModeler(E0, [1000], [0.01])
            E_prime = [model.storage_modulus(w) for w in omega]
            ax1.plot(freq_log, np.log10(E_prime), linewidth=2, label=f'E₀ = {E0} MPa')

        ax1.set_xlabel('log10 f (Hz)', fontsize=10)
        ax1.set_ylabel("log10 E' (MPa)", fontsize=10)
        ax1.set_title('E₀ 변화 효과', fontsize=11, weight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Example 2: Ei effect
        ax2 = example_fig.add_subplot(222)
        for Ei in [500, 2000, 5000]:
            model = ViscoelasticModeler(10, [Ei], [0.01])
            E_double = [model.loss_modulus(w) for w in omega]
            ax2.plot(freq_log, np.log10(np.array(E_double) + 1e-10), linewidth=2, label=f'E₁ = {Ei} MPa')

        ax2.set_xlabel('log10 f (Hz)', fontsize=10)
        ax2.set_ylabel('log10 E" (MPa)', fontsize=10)
        ax2.set_title('E₁ 변화 효과 (피크 높이)', fontsize=11, weight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Example 3: tau effect
        ax3 = example_fig.add_subplot(223)
        for tau in [1e-3, 1e-2, 1e-1]:
            model = ViscoelasticModeler(10, [2000], [tau])
            E_double = [model.loss_modulus(w) for w in omega]
            ax3.plot(freq_log, np.log10(np.array(E_double) + 1e-10), linewidth=2, label=f'τ₁ = {tau} s')

        ax3.set_xlabel('log10 f (Hz)', fontsize=10)
        ax3.set_ylabel('log10 E" (MPa)', fontsize=10)
        ax3.set_title('τ₁ 변화 효과 (피크 위치)', fontsize=11, weight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # Example 4: tan delta
        ax4 = example_fig.add_subplot(224)
        for tau in [1e-3, 1e-2, 1e-1]:
            model = ViscoelasticModeler(10, [2000], [tau])
            tan_d = [model.tan_delta(w) for w in omega]
            f_peak = 1.0 / (2 * np.pi * tau)
            ax4.plot(freq_log, tan_d, linewidth=2, label=f'τ₁={tau}s, Tg@{np.log10(f_peak):.1f}')

        ax4.set_xlabel('log10 f (Hz)', fontsize=10)
        ax4.set_ylabel('tan δ', fontsize=10)
        ax4.set_title('tan δ와 Tg 관계', fontsize=11, weight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

        example_fig.tight_layout()

        # Embed in tkinter
        example_canvas = FigureCanvasTkAgg(example_fig, master=parent)
        example_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_element_entries(self):
        """Create entry fields for Maxwell elements in 2 columns"""
        # Completely destroy and recreate the container to avoid shifting
        for widget in self.elements_outer_container.winfo_children():
            widget.destroy()

        self.element_frames.clear()
        self.E_entries.clear()
        self.tau_entries.clear()

        # Create container with 2 columns
        col1_frame = Frame(self.elements_outer_container)
        col2_frame = Frame(self.elements_outer_container)
        col1_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        col2_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

        # Create elements (1-10 in col1, 11-20 in col2)
        for i in range(self.n_elements):
            parent = col1_frame if i < 10 else col2_frame

            element_frame = Frame(parent, relief=tk.GROOVE, borderwidth=2, bg='#f0f0f0')
            element_frame.pack(fill=tk.X, pady=2)

            Label(element_frame, text=f"Element {i+1}", font=('Arial', 9, 'bold'), bg='#f0f0f0').grid(
                row=0, column=0, columnspan=2, pady=2)

            Label(element_frame, text=f"E{i+1}:", bg='#f0f0f0', font=('Arial', 8)).grid(row=1, column=0, sticky='e', padx=3)
            E_entry = Entry(element_frame, width=12, font=('Arial', 8))
            E_val = self.E_i_default[i] if i < len(self.E_i_default) else 1000.0
            E_entry.insert(0, str(E_val))
            E_entry.grid(row=1, column=1, padx=3, pady=1)
            self.E_entries.append(E_entry)

            Label(element_frame, text=f"τ{i+1}:", bg='#f0f0f0', font=('Arial', 8)).grid(row=2, column=0, sticky='e', padx=3)
            tau_entry = Entry(element_frame, width=12, font=('Arial', 8))
            tau_val = self.tau_i_default[i] if i < len(self.tau_i_default) else 1e-3
            tau_entry.insert(0, str(tau_val))
            tau_entry.grid(row=2, column=1, padx=3, pady=1)
            self.tau_entries.append(tau_entry)

            self.element_frames.append(element_frame)

    def update_element_count(self):
        """Update the number of Maxwell elements"""
        try:
            new_n = int(self.n_elem_var.get())
            if 1 <= new_n <= self.max_elements:
                self.n_elements = new_n
                self.create_element_entries()
                self.update_all()
            else:
                messagebox.showerror("Error", f"Number of elements must be between 1 and {self.max_elements}")
        except ValueError:
            messagebox.showerror("Error", "Invalid number of elements")

    def load_pasted_data(self):
        """Load data from text area - expects linear scale data (freq, E', E")"""
        try:
            text = self.data_text.get('1.0', tk.END)
            lines = [line.strip() for line in text.split('\n') if line.strip() and not line.strip().startswith('#')]

            freq_list = []
            E_prime_list = []
            E_double_list = []

            for line in lines:
                parts = [x.strip() for x in line.replace(',', ' ').split()]
                if len(parts) >= 3:
                    freq_list.append(float(parts[0]))
                    E_prime_list.append(float(parts[1]))
                    E_double_list.append(float(parts[2]))

            if len(freq_list) > 0:
                # Store data in linear scale
                self.master_freq = np.array(freq_list)
                self.master_E_prime = np.array(E_prime_list)
                self.master_E_double = np.array(E_double_list)

                messagebox.showinfo("Success", f"Loaded {len(freq_list)} data points")
                self.update_modulus_plot()
            else:
                messagebox.showerror("Error", "No valid data found")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse data: {e}")

    def adjust_tau_for_tg(self):
        """Auto-adjust tau values based on Tg frequency"""
        try:
            tg_freq_log = float(self.tg_freq_entry.get())
            tg_freq = 10**tg_freq_log

            # Calculate tau at Tg: f = 1/(2πτ) → τ = 1/(2πf)
            tau_tg = 1.0 / (2 * np.pi * tg_freq)

            # Distribute tau values around tau_tg in log scale
            n = self.n_elements
            if n == 1:
                tau_values = [tau_tg]
            else:
                # Spread taus logarithmically around tau_tg
                log_tau_tg = np.log10(tau_tg)
                spread = 3.0  # ±3 orders of magnitude
                log_taus = np.linspace(log_tau_tg - spread, log_tau_tg + spread, n)
                tau_values = 10**log_taus

            # Update tau entries
            for i, tau_entry in enumerate(self.tau_entries):
                if i < len(tau_values):
                    tau_entry.delete(0, tk.END)
                    tau_entry.insert(0, f"{tau_values[i]:.2e}")

            self.update_all()
            messagebox.showinfo("Success", f"τ values adjusted for Tg at log10(f) = {tg_freq_log:.2f} Hz")

        except ValueError:
            messagebox.showerror("Error", "Invalid Tg frequency value")

    def get_parameters(self):
        """Read parameters from GUI entries"""
        try:
            E0 = float(self.E0_entry.get())
            E_i = [float(entry.get()) for entry in self.E_entries]
            tau_i = [float(entry.get()) for entry in self.tau_entries]
            freq_min = float(self.freq_min_entry.get())
            freq_max = float(self.freq_max_entry.get())
            return E0, E_i, tau_i, freq_min, freq_max
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter: {e}")
            return None

    def set_parameters(self, E0, E_i, tau_i):
        """Set parameters to GUI entries"""
        self.E0_entry.delete(0, tk.END)
        self.E0_entry.insert(0, f"{E0:.2f}")

        for i, (E_entry, tau_entry) in enumerate(zip(self.E_entries, self.tau_entries)):
            if i < len(E_i):
                E_entry.delete(0, tk.END)
                E_entry.insert(0, f"{E_i[i]:.2e}")

                tau_entry.delete(0, tk.END)
                tau_entry.insert(0, f"{tau_i[i]:.2e}")

    def reset_parameters(self):
        """Reset to default values"""
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

    def start_fitting(self):
        """Start fitting in a separate thread"""
        if self.master_freq is None:
            messagebox.showerror("Error", "Please load data first")
            return

        if self.fitting_in_progress:
            messagebox.showwarning("Warning", "Fitting already in progress")
            return

        # Switch to fitting progress tab
        self.param_notebook.select(2)

        # Start fitting in background thread
        fitting_thread = threading.Thread(target=self.fit_to_master_curve, daemon=True)
        fitting_thread.start()

    def fit_to_master_curve(self):
        """Fit model parameters to master curve data with improved algorithm"""
        if self.master_freq is None:
            messagebox.showerror("Error", "Please load data first")
            return

        try:
            self.fitting_in_progress = True
            self.fitting_history = []

            # Update fitting info
            self.update_fitting_info("피팅 시작...\n")

            n_elem = self.n_elements
            self.update_fitting_info(f"Maxwell 요소 개수: {n_elem}\n")

            # Estimate initial parameters from data
            self.update_fitting_info("\n초기값 추정 중...\n")
            E_all = np.concatenate([self.master_E_prime, self.master_E_double])
            E_all_positive = E_all[E_all > 0]

            if len(E_all_positive) == 0:
                raise ValueError("No positive modulus values in data")

            E_min_data = np.min(E_all_positive)
            E_max_data = np.max(E_all_positive)
            E0_init = E_min_data  # Initial guess for E0

            self.update_fitting_info(f"  E 범위: {E_min_data:.2e} - {E_max_data:.2e} MPa\n")
            self.update_fitting_info(f"  초기 E0: {E0_init:.2e} MPa\n")

            # Estimate tau from E" peak
            freq = self.master_freq
            E_double_data = self.master_E_double
            if len(E_double_data) > 0 and np.max(E_double_data) > 0:
                peak_idx = np.argmax(E_double_data)
                peak_freq = freq[peak_idx]
                tau_peak = 1.0 / (2 * np.pi * peak_freq)
                self.update_fitting_info(f"  E\" 피크 주파수: {peak_freq:.2e} Hz\n")
                self.update_fitting_info(f"  추정 tau (피크): {tau_peak:.2e} s\n")
            else:
                tau_peak = 1.0

            # Callback to track progress
            iteration_count = [0]

            def objective_with_callback(params):
                try:
                    E0 = params[0]
                    E_i = params[1:n_elem+1]
                    tau_i = params[n_elem+1:2*n_elem+1]

                    model = ViscoelasticModeler(E0, E_i, tau_i)

                    # Data is in linear scale
                    omega = 2 * np.pi * freq

                    E_prime_pred = np.array([model.storage_modulus(w) for w in omega])
                    E_double_pred = np.array([model.loss_modulus(w) for w in omega])

                    # Avoid log of zero or negative values
                    E_prime_pred = np.maximum(E_prime_pred, 1e-10)
                    E_double_pred = np.maximum(E_double_pred, 1e-10)
                    E_prime_data = np.maximum(self.master_E_prime, 1e-10)
                    E_double_data = np.maximum(self.master_E_double, 1e-10)

                    # Compare in log-log space for better fitting across orders of magnitude
                    error_prime = np.sum((np.log10(E_prime_pred) - np.log10(E_prime_data))**2)
                    error_double = np.sum((np.log10(E_double_pred) - np.log10(E_double_data))**2)

                    # Weight E" more heavily to capture peak better
                    total_error = error_prime + 2.0 * error_double

                    # Track progress
                    iteration_count[0] += 1
                    if iteration_count[0] % 50 == 0:
                        self.fitting_history.append(total_error)
                        self.master.after(0, self.update_convergence_plot)
                        self.master.after(0, lambda: self.update_fitting_info(
                            f"반복 {iteration_count[0]}: Error = {total_error:.4e}\n"))

                    return total_error
                except Exception as e:
                    return 1e10

            # Set bounds for parameters
            try:
                E_min = max(E_min_data * 0.001, 0.1)
                E_max = min(E_max_data * 100, 1e7)

                # Ensure bounds are valid
                if not (np.isfinite(E_min) and np.isfinite(E_max) and E_min < E_max):
                    raise ValueError("Invalid bounds from data")

            except Exception as e:
                self.update_fitting_info(f"경고: {e}, 기본값 사용\n")
                E_min = 0.1
                E_max = 1e6

            bounds = []
            # E0 bounds (narrower range for E0)
            bounds.append((E_min, E_max / 10))
            # E_i bounds
            for i in range(n_elem):
                bounds.append((E_min, E_max))
            # tau_i bounds - estimate from frequency range
            freq_min = np.min(freq)
            freq_max = np.max(freq)
            tau_min = max(1.0 / (2 * np.pi * freq_max * 1000), 1e-12)
            tau_max = min(1.0 / (2 * np.pi * freq_min * 0.001), 1e8)

            for i in range(n_elem):
                bounds.append((tau_min, tau_max))

            self.update_fitting_info(f"\n최적화 설정:\n")
            self.update_fitting_info(f"  E 범위: [{E_min:.2e}, {E_max:.2e}] MPa\n")
            self.update_fitting_info(f"  tau 범위: [{tau_min:.2e}, {tau_max:.2e}] s\n")
            self.update_fitting_info(f"  Population size: 30\n")
            self.update_fitting_info(f"  Max iterations: 300\n")
            self.update_fitting_info(f"\n피팅 진행 중...\n")

            # Verify all bounds are valid
            for b in bounds:
                if not (np.isfinite(b[0]) and np.isfinite(b[1]) and b[0] < b[1]):
                    raise ValueError(f"Invalid bound: {b}")

            # Run optimization with better parameters
            result = differential_evolution(
                objective_with_callback,
                bounds,
                maxiter=300,  # Increased iterations
                popsize=30,  # Increased population size
                seed=42,
                workers=1,
                atol=1e-10,  # Tighter tolerance
                tol=1e-8,
                updating='deferred',  # Better for noisy objectives
                polish=True  # Final local optimization
            )

            E0_fit = result.x[0]
            E_i_fit = result.x[1:n_elem+1]
            tau_i_fit = result.x[n_elem+1:2*n_elem+1]

            # Update parameters in GUI
            self.master.after(0, lambda: self.set_parameters(E0_fit, E_i_fit, tau_i_fit))

            # Update both diagram and plot
            self.master.after(0, self.update_all)

            # Calculate Tg from fitted parameters
            model_fit = ViscoelasticModeler(E0_fit, E_i_fit, tau_i_fit)
            freq_range = np.logspace(np.log10(freq_min), np.log10(freq_max), 1000)
            omega_range = 2 * np.pi * freq_range
            tan_delta_range = [model_fit.tan_delta(w) for w in omega_range]
            tg_idx = np.argmax(tan_delta_range)
            tg_freq_fit = freq_range[tg_idx]

            # Display results
            result_msg = f"\n{'='*50}\n피팅 완료!\n{'='*50}\n"
            result_msg += f"최종 에러: {result.fun:.4e}\n"
            result_msg += f"반복 횟수: {iteration_count[0]}\n\n"
            result_msg += f"최적화된 파라미터:\n"
            result_msg += f"  E₀ = {E0_fit:.2e} MPa\n"
            for i in range(n_elem):
                result_msg += f"  E{i+1} = {E_i_fit[i]:.2e} MPa, τ{i+1} = {tau_i_fit[i]:.2e} s\n"
            result_msg += f"\n유리전이 (Tg):\n"
            result_msg += f"  주파수: {tg_freq_fit:.2e} Hz (log10: {np.log10(tg_freq_fit):.2f})\n"
            result_msg += f"  tan δ (max): {np.max(tan_delta_range):.4f}\n"
            result_msg += f"\n{'='*50}\n"

            self.update_fitting_info(result_msg)

            # Update Tg entry
            self.master.after(0, lambda: self.tg_freq_entry.delete(0, tk.END))
            self.master.after(0, lambda: self.tg_freq_entry.insert(0, f"{np.log10(tg_freq_fit):.2f}"))

            self.master.after(0, lambda: messagebox.showinfo("Success",
                f"피팅 완료!\n최종 에러: {result.fun:.2e}\nTg @ {np.log10(tg_freq_fit):.2f} Hz"))

        except Exception as e:
            import traceback
            error_msg = f"피팅 실패: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            self.update_fitting_info(f"\n에러 발생:\n{error_msg}\n")
            self.master.after(0, lambda: messagebox.showerror("Error", f"Fitting failed: {str(e)}"))

        finally:
            self.fitting_in_progress = False

    def update_fitting_info(self, message):
        """Update fitting info text (thread-safe)"""
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
        """Update the convergence plot"""
        if len(self.fitting_history) < 2:
            return

        self.convergence_ax.clear()
        iterations = np.arange(1, len(self.fitting_history) + 1) * 50
        self.convergence_ax.plot(iterations, self.fitting_history, 'b-', linewidth=2)
        self.convergence_ax.set_xlabel('Iteration', fontsize=10, weight='bold')
        self.convergence_ax.set_ylabel('Error', fontsize=10, weight='bold')
        self.convergence_ax.set_title('피팅 수렴 과정', fontsize=12, weight='bold')
        self.convergence_ax.set_yscale('log')
        self.convergence_ax.grid(True, alpha=0.3)
        self.convergence_fig.tight_layout()
        self.convergence_canvas.draw()

    def update_maxwell_diagram(self):
        """Update the Maxwell model diagram"""
        diagram = MaxwellDiagramCanvas(self.diagram_fig, self.diagram_ax, self.n_elements)
        diagram.draw_model()
        self.diagram_fig.tight_layout()
        self.diagram_canvas.draw()

    def update_modulus_plot(self):
        """Update the modulus plot with tan delta"""
        params = self.get_parameters()
        if params is None:
            return

        E0, E_i, tau_i, freq_min, freq_max = params

        model = ViscoelasticModeler(E0, E_i, tau_i)

        freq_log = np.linspace(freq_min, freq_max, 1000)
        freq = 10**freq_log
        omega = 2 * np.pi * freq

        E_prime = np.array([model.storage_modulus(w) for w in omega])
        E_double_prime = np.array([model.loss_modulus(w) for w in omega])
        tan_delta = np.array([model.tan_delta(w) for w in omega])

        self.plot_ax.clear()

        # Plot individual element contributions if checkbox is checked
        if hasattr(self, 'show_contributions_var') and self.show_contributions_var.get():
            colors = plt.cm.tab10(np.linspace(0, 1, len(E_i)))
            for i, (E, tau, color) in enumerate(zip(E_i, tau_i, colors)):
                E_prime_i = np.array([model.element_contribution(w, E, tau)[0] for w in omega])
                E_double_i = np.array([model.element_contribution(w, E, tau)[1] for w in omega])

                self.plot_ax.plot(freq_log, np.log10(E_prime_i + 1e-10), '--',
                                color=color, linewidth=1.5, alpha=0.6, label=f"Element {i+1} E'")
                self.plot_ax.plot(freq_log, np.log10(E_double_i + 1e-10), ':',
                                color=color, linewidth=1.5, alpha=0.6, label=f'Element {i+1} E"')

        # Plot total moduli
        self.plot_ax.plot(freq_log, np.log10(E_prime), 'r-', linewidth=2.5, label="Total Storage modulus E'")
        self.plot_ax.plot(freq_log, np.log10(E_double_prime), 'g-', linewidth=2.5, label='Total Loss modulus E"')

        # Plot tan delta on secondary axis if enabled
        if hasattr(self, 'show_tan_delta_var') and self.show_tan_delta_var.get():
            ax2 = self.plot_ax.twinx()
            ax2.plot(freq_log, tan_delta, 'b-', linewidth=2.5, alpha=0.7, label='tan δ')

            # Mark Tg (tan delta peak)
            tg_idx = np.argmax(tan_delta)
            tg_freq_log = freq_log[tg_idx]
            tg_tan_delta = tan_delta[tg_idx]
            ax2.plot(tg_freq_log, tg_tan_delta, 'b*', markersize=15, label=f'Tg @ {tg_freq_log:.2f}')

            ax2.set_ylabel('tan δ', fontsize=12, weight='bold', color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            ax2.legend(fontsize=9, loc='upper right')
            ax2.set_ylim(0, max(tan_delta) * 1.2)

        # Plot master curve data if available (data is in linear scale, convert to log-log)
        if self.master_freq is not None:
            master_freq_log = np.log10(np.maximum(self.master_freq, 1e-10))
            master_E_prime_log = np.log10(np.maximum(self.master_E_prime, 1e-10))
            master_E_double_log = np.log10(np.maximum(self.master_E_double, 1e-10))

            self.plot_ax.scatter(master_freq_log, master_E_prime_log,
                               c='darkred', marker='o', s=30, alpha=0.6, label="Master E' data")
            self.plot_ax.scatter(master_freq_log, master_E_double_log,
                               c='darkgreen', marker='s', s=30, alpha=0.6, label='Master E" data')

        self.plot_ax.set_xlabel('log10 f (Hz)', fontsize=12, weight='bold')
        self.plot_ax.set_ylabel('log10 E (MPa)', fontsize=12, weight='bold')
        self.plot_ax.set_title('Complex Modulus of Viscoelasticity', fontsize=14, weight='bold')

        # Set y-axis limits (modulus doesn't go below 10^-1 = 0.1 MPa)
        self.plot_ax.set_ylim(-1, None)

        self.plot_ax.legend(fontsize=9, loc='upper left', ncol=2)
        self.plot_ax.grid(True, alpha=0.3)

        self.plot_fig.tight_layout()
        self.plot_canvas.draw()

    def update_all(self):
        """Update both diagram and plot"""
        self.update_maxwell_diagram()
        self.update_modulus_plot()


def main():
    root = tk.Tk()
    app = ViscoelasticGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
