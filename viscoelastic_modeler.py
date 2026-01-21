#!/usr/bin/env python3
"""
Viscoelastic Rubber Compound Modeler v2.1
Generalized Maxwell Model for Complex Modulus Calculation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, Polygon
import matplotlib.font_manager as fm
import tkinter as tk
from tkinter import ttk, Frame, Label, Entry, Button, Text, Scrollbar, messagebox
from scipy.optimize import differential_evolution


# Configure Korean font for matplotlib
def setup_korean_font():
    """Setup Korean font for matplotlib to prevent encoding issues"""
    try:
        korean_fonts = ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'Noto Sans KR']
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        for font in korean_fonts:
            if font in available_fonts:
                plt.rcParams['font.family'] = font
                plt.rcParams['axes.unicode_minus'] = False
                return
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        plt.rcParams['axes.unicode_minus'] = False


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

    def complex_modulus(self, omega):
        """Calculate complex modulus E*(ω) = E'(ω) + i·E"(ω)"""
        return self.storage_modulus(omega), self.loss_modulus(omega)


class MaxwellDiagramCanvas:
    """Canvas for drawing Generalized Maxwell model diagram - similar to reference image"""

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
        # Cylinder
        cylinder_h = height * 0.6
        self.ax.add_patch(Rectangle((x - width/2, y), width, cylinder_h,
                                   facecolor='white', edgecolor=color, linewidth=2))
        # Piston
        piston_h = height * 0.3
        piston_y = y + cylinder_h
        self.ax.add_patch(Rectangle((x - width/1.5, piston_y), width*1.3, piston_h,
                                   facecolor='lightgray', edgecolor=color, linewidth=2.5))
        # Rod
        self.ax.plot([x, x], [y, piston_y], color=color, linewidth=2.5)
        self.ax.plot([x, x], [piston_y + piston_h, y + height], color=color, linewidth=2.5)

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
        master.title("Viscoelastic Rubber Compound Modeler v2.1 (Generalized Maxwell Model)")
        master.geometry("1700x950")

        # Default parameters
        self.n_elements = 4
        self.max_elements = 20
        self.E0_default = 10.0
        self.E_i_default = [1000.0, 5000.0, 10000.0, 15000.0]
        self.tau_i_default = [1e-6, 1e-4, 1e-2, 1.0]

        self.freq_min = -10
        self.freq_max = 10  # Changed from 20 to 10

        # Master curve data
        self.master_freq = None
        self.master_E_prime = None
        self.master_E_double = None

        # Element management
        self.element_frames = []
        self.E_entries = []
        self.tau_entries = []

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

        # Tab 3: Physics Guide
        guide_tab = Frame(self.param_notebook)
        self.param_notebook.add(guide_tab, text="Physics Guide")

        # ===== PARAMETERS TAB =====
        self.create_parameters_tab(params_tab)

        # ===== DATA INPUT TAB =====
        self.create_data_tab(data_tab)

        # ===== PHYSICS GUIDE TAB =====
        self.create_guide_tab(guide_tab)

        # Right panel - Plots
        right_panel = Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Top: Maxwell diagram (bigger now)
        diagram_frame = Frame(right_panel, relief=tk.RIDGE, borderwidth=2)
        diagram_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.diagram_fig = Figure(figsize=(12, 4), dpi=100)
        self.diagram_ax = self.diagram_fig.add_subplot(111)
        self.diagram_canvas = FigureCanvasTkAgg(self.diagram_fig, master=diagram_frame)
        self.diagram_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom: Modulus plot
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
        self.elements_container = Frame(self.scrollable_frame)
        self.elements_container.pack(fill=tk.X, padx=5)

        # Create initial elements
        self.create_element_entries()

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

        Label(parent, text="Format: log10(f), log10(E'), log10(E\") - one row per line",
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
# log10(f), log10(E'), log10(E")
-8.0, 1.2, 0.5
-7.0, 1.5, 0.8
-6.0, 1.8, 1.1
-5.0, 2.2, 1.5
"""
        self.data_text.insert('1.0', sample_text)

        # Buttons
        btn_frame = Frame(parent)
        btn_frame.pack(pady=10)

        Button(btn_frame, text="Load Pasted Data", command=self.load_pasted_data,
               bg='#9C27B0', fg='white', font=('Arial', 11, 'bold'), width=15).pack(side=tk.LEFT, padx=5)

        Button(btn_frame, text="Clear Data", command=lambda: self.data_text.delete('1.0', tk.END),
               bg='#FF5722', fg='white', font=('Arial', 11, 'bold'), width=15).pack(side=tk.LEFT, padx=5)

        Button(btn_frame, text="Fit Model", command=self.fit_to_master_curve,
               bg='#E91E63', fg='white', font=('Arial', 11, 'bold'), width=15).pack(side=tk.LEFT, padx=5)

    def create_guide_tab(self, parent):
        """Create physics guide tab"""
        # Scrollable text area
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        guide_frame = Frame(canvas)

        guide_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=guide_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind_all("<MouseWheel>", lambda event: self._on_mousewheel(event, canvas))

        # Content
        guide_text = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   PHYSICS GUIDE: Understanding Parameters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. E₀ (Equilibrium Modulus)
   • Physical meaning: Long-term elastic response
   • Effect on E': Shifts entire E' curve up/down
   • Effect on E": No direct effect
   • Example: ↑E₀ → Higher baseline stiffness

2. Eᵢ (Maxwell Element Modulus)
   • Physical meaning: Strength of i-th relaxation mode
   • Effect on E': Increases plateau height
   • Effect on E": Increases peak height
   • Example: ↑Eᵢ → Stronger relaxation at τᵢ

3. τᵢ (Relaxation Time)
   • Physical meaning: Time scale of i-th mode
   • Effect on E': Shifts transition location
   • Effect on E": Shifts peak location
   • Example: ↑τᵢ → Transition at lower frequency

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Storage Modulus (E')
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Represents: Elastic energy storage
• Behavior:
  - Low freq: E' ≈ E₀ (equilibrium)
  - High freq: E' ≈ E₀ + ΣEᵢ (glassy)
  - Transition: Smooth increase

• How to modify:
  ✓ Increase E₀ → Shift entire curve up
  ✓ Increase Eᵢ → Steeper transition
  ✓ Decrease τᵢ → Shift transition right

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Loss Modulus (E")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Represents: Energy dissipation
• Behavior:
  - Low freq: E" ≈ 0 (no dissipation)
  - Peak at: f ≈ 1/(2πτᵢ)
  - High freq: E" → 0 (frozen)

• How to modify:
  ✓ Increase Eᵢ → Higher peak
  ✓ Increase τᵢ → Peak shifts left
  ✓ Add elements → Multiple peaks

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Practical Examples
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Example 1: Soft rubber (low stiffness)
  E₀ = 1 MPa, E₁ = 10 MPa, τ₁ = 0.01 s

Example 2: Hard rubber (high stiffness)
  E₀ = 100 MPa, E₁ = 1000 MPa, τ₁ = 0.01 s

Example 3: Multiple relaxations
  E₀ = 10 MPa
  E₁ = 100 MPa, τ₁ = 1e-6 s (fast)
  E₂ = 500 MPa, τ₂ = 1e-3 s (medium)
  E₃ = 1000 MPa, τ₃ = 1 s (slow)

Example 4: Broadening loss peak
  Add more elements with different τᵢ
  τ₁ = 1e-4, τ₂ = 1e-3, τ₃ = 1e-2, τ₄ = 1e-1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Tips for Good Fitting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Use 3-6 Maxwell elements for most materials
2. Distribute τᵢ values logarithmically
3. Start with visual parameter adjustment
4. Use "Fit Model" for fine-tuning
5. Check if E' and E" both fit well

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        text_widget = Text(guide_frame, wrap=tk.WORD, font=('Courier', 10),
                          bg='#f5f5f5', padx=15, pady=15)
        text_widget.insert('1.0', guide_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_element_entries(self):
        """Create entry fields for Maxwell elements in 2 columns"""
        # Clear existing
        for frame in self.element_frames:
            frame.destroy()
        self.element_frames.clear()
        self.E_entries.clear()
        self.tau_entries.clear()

        # Create container with 2 columns
        col1_frame = Frame(self.elements_container)
        col2_frame = Frame(self.elements_container)
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
        """Load data from text area"""
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
                self.master_freq = np.array(freq_list)
                self.master_E_prime = np.array(E_prime_list)
                self.master_E_double = np.array(E_double_list)

                messagebox.showinfo("Success", f"Loaded {len(freq_list)} data points")
                self.update_modulus_plot()
            else:
                messagebox.showerror("Error", "No valid data found")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse data: {e}")

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

    def fit_to_master_curve(self):
        """Fit model parameters to master curve data"""
        if self.master_freq is None:
            messagebox.showerror("Error", "Please load data first")
            return

        try:
            n_elem = self.n_elements

            def objective(params):
                E0 = params[0]
                E_i = params[1:n_elem+1]
                tau_i = params[n_elem+1:2*n_elem+1]

                model = ViscoelasticModeler(E0, E_i, tau_i)

                freq = 10**self.master_freq
                omega = 2 * np.pi * freq

                E_prime_pred = np.array([model.storage_modulus(w) for w in omega])
                E_double_pred = np.array([model.loss_modulus(w) for w in omega])

                error = np.sum((np.log10(E_prime_pred + 1e-10) - self.master_E_prime)**2) + \
                       np.sum((np.log10(E_double_pred + 1e-10) - self.master_E_double)**2)

                return error

            E_target_max = 10**np.max(self.master_E_prime)
            bounds = [(1, E_target_max)]
            for i in range(n_elem):
                bounds.append((10, E_target_max))
            for i in range(n_elem):
                bounds.append((1e-10, 1e5))

            messagebox.showinfo("Fitting", "Fitting in progress... This may take a while.")

            result = differential_evolution(objective, bounds, maxiter=100, popsize=15, seed=42)

            E0_fit = result.x[0]
            E_i_fit = result.x[1:n_elem+1]
            tau_i_fit = result.x[n_elem+1:2*n_elem+1]

            self.set_parameters(E0_fit, E_i_fit, tau_i_fit)
            self.update_all()

            messagebox.showinfo("Success", f"Fitting completed!\nFinal error: {result.fun:.2e}")

        except Exception as e:
            messagebox.showerror("Error", f"Fitting failed: {e}")

    def update_maxwell_diagram(self):
        """Update the Maxwell model diagram"""
        diagram = MaxwellDiagramCanvas(self.diagram_fig, self.diagram_ax, self.n_elements)
        diagram.draw_model()
        self.diagram_fig.tight_layout()
        self.diagram_canvas.draw()

    def update_modulus_plot(self):
        """Update the modulus plot"""
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

        self.plot_ax.clear()
        self.plot_ax.plot(freq_log, np.log10(E_prime), 'r-', linewidth=2.5, label="Storage modulus E'")
        self.plot_ax.plot(freq_log, np.log10(E_double_prime), 'g-', linewidth=2.5, label='Loss modulus E"')

        if self.master_freq is not None:
            self.plot_ax.scatter(self.master_freq, self.master_E_prime,
                               c='darkred', marker='o', s=30, alpha=0.6, label="Master E' data")
            self.plot_ax.scatter(self.master_freq, self.master_E_double,
                               c='darkgreen', marker='s', s=30, alpha=0.6, label='Master E" data')

        self.plot_ax.set_xlabel('log10 f (Hz)', fontsize=12, weight='bold')
        self.plot_ax.set_ylabel('log10 E (MPa)', fontsize=12, weight='bold')
        self.plot_ax.set_title('Complex Modulus of Viscoelasticity', fontsize=14, weight='bold')
        self.plot_ax.legend(fontsize=10, loc='best')
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
