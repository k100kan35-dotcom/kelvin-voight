#!/usr/bin/env python3
"""
Viscoelastic Rubber Compound Modeler
Generalized Maxwell Model for Complex Modulus Calculation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.font_manager as fm
import tkinter as tk
from tkinter import ttk, Frame, Label, Entry, Button, filedialog, messagebox
from scipy.optimize import curve_fit, differential_evolution
import json


# Configure Korean font for matplotlib
def setup_korean_font():
    """Setup Korean font for matplotlib to prevent encoding issues"""
    try:
        # Try common Korean fonts
        korean_fonts = ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'Noto Sans KR']
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        for font in korean_fonts:
            if font in available_fonts:
                plt.rcParams['font.family'] = font
                plt.rcParams['axes.unicode_minus'] = False
                return

        # Fallback: use DejaVu Sans which supports some unicode
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        # If all else fails, disable font warnings
        plt.rcParams['axes.unicode_minus'] = False


setup_korean_font()


class ViscoelasticModeler:
    """
    Generalized Maxwell model for viscoelastic materials.

    Complex modulus: E(ω) = E₀ + Σ[Eᵢ·ω²τᵢ²/(1 + ω²τᵢ²) + i·Eᵢ·ωτᵢ/(1 + ω²τᵢ²)]
    E' = Storage modulus (저장탄성계수)
    E" = Loss modulus (손실탄성계수)
    """

    def __init__(self, E0, E_i, tau_i):
        """
        Parameters:
        -----------
        E0 : float
            Equilibrium modulus (평형 탄성계수) [MPa]
        E_i : list of float
            Moduli of Maxwell elements [MPa]
        tau_i : list of float
            Relaxation times [s]
        """
        self.E0 = E0
        self.E_i = np.array(E_i) if len(E_i) > 0 else np.array([])
        self.tau_i = np.array(tau_i) if len(tau_i) > 0 else np.array([])

    def storage_modulus(self, omega):
        """
        Calculate storage modulus E'(ω)
        E' = E₀ + Σ[Eᵢ·ω²τᵢ²/(1 + ω²τᵢ²)]
        """
        E_prime = self.E0
        for E, tau in zip(self.E_i, self.tau_i):
            E_prime += E * (omega * tau)**2 / (1 + (omega * tau)**2)
        return E_prime

    def loss_modulus(self, omega):
        """
        Calculate loss modulus E"(ω)
        E" = Σ[Eᵢ·ωτᵢ/(1 + ω²τᵢ²)]
        """
        E_double_prime = 0
        for E, tau in zip(self.E_i, self.tau_i):
            E_double_prime += E * omega * tau / (1 + (omega * tau)**2)
        return E_double_prime

    def complex_modulus(self, omega):
        """
        Calculate complex modulus E*(ω) = E'(ω) + i·E"(ω)
        Returns: (E', E")
        """
        return self.storage_modulus(omega), self.loss_modulus(omega)


class MaxwellDiagramCanvas:
    """Canvas for drawing Generalized Maxwell model diagram"""

    def __init__(self, fig, ax, n_elements):
        self.fig = fig
        self.ax = ax
        self.n_elements = n_elements

    def draw_spring(self, x, y, height, width=0.15):
        """Draw a spring symbol"""
        n_coils = 8
        coil_y = np.linspace(y, y + height, n_coils * 2)
        coil_x = [x + (-1)**i * width/2 for i in range(len(coil_y))]
        self.ax.plot(coil_x, coil_y, 'k-', linewidth=2)

    def draw_dashpot(self, x, y, height, width=0.25):
        """Draw a dashpot (damper) symbol"""
        # Piston
        piston_height = height * 0.3
        self.ax.add_patch(Rectangle((x - width/2, y + height - piston_height),
                                   width, piston_height,
                                   facecolor='lightgray', edgecolor='black', linewidth=2))
        # Rod
        self.ax.plot([x, x], [y, y + height - piston_height], 'k-', linewidth=2)
        # Cylinder
        cylinder_height = height * 0.6
        self.ax.add_patch(Rectangle((x - width/3, y + height - piston_height - cylinder_height),
                                   width*2/3, cylinder_height,
                                   facecolor='white', edgecolor='black', linewidth=1.5))

    def draw_maxwell_element(self, x, y, element_height=1.5):
        """Draw a Maxwell element (spring and dashpot in series)"""
        spring_height = element_height * 0.5
        dashpot_height = element_height * 0.4
        gap = element_height * 0.1

        # Top connection
        self.ax.plot([x, x], [y + element_height, y + spring_height + gap], 'k-', linewidth=2)
        # Spring
        self.draw_spring(x, y + gap, spring_height)
        # Connection between spring and dashpot
        self.ax.plot([x, x], [y + gap, y], 'k-', linewidth=2)
        # Dashpot (offset slightly)
        dashpot_y = y + gap + spring_height + 0.05
        self.draw_dashpot(x, dashpot_y, dashpot_height)

    def draw_model(self):
        """Draw the complete Generalized Maxwell model"""
        self.ax.clear()
        self.ax.set_xlim(-0.5, max(4, self.n_elements + 2))
        self.ax.set_ylim(-0.5, 3.5)
        self.ax.axis('off')
        self.ax.set_aspect('equal')

        element_height = 2.0
        y_base = 0.5
        y_top = y_base + element_height

        # Draw E0 spring (leftmost)
        x_e0 = 0.5
        self.draw_spring(x_e0, y_base, element_height, width=0.2)
        self.ax.text(x_e0, y_base - 0.3, 'E₀', ha='center', fontsize=10, weight='bold')

        # Draw top and bottom bars
        bar_width = min(self.n_elements + 1.5, 10)
        self.ax.plot([0, bar_width], [y_top, y_top], 'k-', linewidth=3)
        self.ax.plot([0, bar_width], [y_base, y_base], 'k-', linewidth=3)

        # Draw Maxwell elements
        for i in range(min(self.n_elements, 15)):  # Limit display to 15 elements
            x_elem = 1.5 + i * 0.6
            self.draw_maxwell_element(x_elem, y_base + 0.1, element_height * 0.8)
            self.ax.text(x_elem, y_base - 0.3, f'τ{i+1}', ha='center', fontsize=8)

        if self.n_elements > 15:
            x_dots = 1.5 + 15 * 0.6
            self.ax.text(x_dots, y_base + element_height/2, '...', ha='center', fontsize=20, weight='bold')
            self.ax.text(x_dots, y_base - 0.3, f'({self.n_elements} elements)', ha='center', fontsize=8)

        # Add title
        self.ax.text(bar_width/2, y_top + 0.5, 'Generalized Maxwell Model',
                    ha='center', fontsize=12, weight='bold')


class ViscoelasticGUI:
    def __init__(self, master):
        self.master = master
        master.title("Viscoelastic Rubber Compound Modeler (Generalized Maxwell Model)")
        master.geometry("1600x900")

        # Default parameters
        self.n_elements = 4
        self.max_elements = 20
        self.E0_default = 10.0
        self.E_i_default = [1000.0, 5000.0, 10000.0, 15000.0]
        self.tau_i_default = [1e-6, 1e-4, 1e-2, 1.0]

        self.freq_min = -10
        self.freq_max = 20

        # Master curve data
        self.master_freq = None  # log10(f)
        self.master_E_prime = None  # log10(E')
        self.master_E_double = None  # log10(E")

        # Element management
        self.element_frames = []
        self.E_entries = []
        self.tau_entries = []

        # Create GUI layout
        self.create_widgets()
        self.update_all()

    def create_widgets(self):
        # Main container
        main_frame = Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Parameters
        left_panel = Frame(main_frame, width=350, relief=tk.RIDGE, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

        # Title
        title_label = Label(left_panel, text="Model Parameters", font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)

        # Scrollable frame
        canvas = tk.Canvas(left_panel)
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        self.scrollable_frame = Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

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

        # Maxwell elements container
        Label(self.scrollable_frame, text="Maxwell Elements:", font=('Arial', 11, 'bold')).pack(pady=(20, 10))
        self.elements_container = Frame(self.scrollable_frame)
        self.elements_container.pack(fill=tk.X, padx=10)

        # Create initial elements
        self.create_element_entries()

        # Data fitting section
        Label(self.scrollable_frame, text="Master Curve Fitting:", font=('Arial', 11, 'bold')).pack(pady=(20, 5))

        fit_btn_frame = Frame(self.scrollable_frame)
        fit_btn_frame.pack(pady=5)

        Button(fit_btn_frame, text="Load Data", command=self.load_master_curve,
               bg='#9C27B0', fg='white', font=('Arial', 10, 'bold'), width=12).pack(side=tk.LEFT, padx=5)

        Button(fit_btn_frame, text="Fit Model", command=self.fit_to_master_curve,
               bg='#E91E63', fg='white', font=('Arial', 10, 'bold'), width=12).pack(side=tk.LEFT, padx=5)

        # Control buttons
        Button(self.scrollable_frame, text="Update Plots", command=self.update_all,
               bg='#4CAF50', fg='white', font=('Arial', 12, 'bold'), pady=10).pack(pady=20, fill=tk.X, padx=20)

        Button(self.scrollable_frame, text="Reset", command=self.reset_parameters,
               bg='#ff9800', fg='white', font=('Arial', 10), pady=5).pack(pady=(0, 20), fill=tk.X, padx=20)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Right panel - Plots
        right_panel = Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Top: Maxwell diagram
        diagram_frame = Frame(right_panel, relief=tk.RIDGE, borderwidth=2)
        diagram_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.diagram_fig = Figure(figsize=(8, 3), dpi=80)
        self.diagram_ax = self.diagram_fig.add_subplot(111)
        self.diagram_canvas = FigureCanvasTkAgg(self.diagram_fig, master=diagram_frame)
        self.diagram_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom: Modulus plot
        plot_frame = Frame(right_panel, relief=tk.RIDGE, borderwidth=2)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.plot_fig = Figure(figsize=(8, 5), dpi=100)
        self.plot_ax = self.plot_fig.add_subplot(111)
        self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_element_entries(self):
        """Create entry fields for Maxwell elements"""
        # Clear existing
        for frame in self.element_frames:
            frame.destroy()
        self.element_frames.clear()
        self.E_entries.clear()
        self.tau_entries.clear()

        # Create new elements
        for i in range(self.n_elements):
            element_frame = Frame(self.elements_container, relief=tk.GROOVE, borderwidth=2, bg='#f0f0f0')
            element_frame.pack(fill=tk.X, pady=3)

            Label(element_frame, text=f"Element {i+1}", font=('Arial', 9, 'bold'), bg='#f0f0f0').grid(
                row=0, column=0, columnspan=2, pady=3)

            Label(element_frame, text=f"E{i+1} (MPa):", bg='#f0f0f0').grid(row=1, column=0, sticky='e', padx=5)
            E_entry = Entry(element_frame, width=15)
            E_val = self.E_i_default[i] if i < len(self.E_i_default) else 1000.0
            E_entry.insert(0, str(E_val))
            E_entry.grid(row=1, column=1, padx=5, pady=2)
            self.E_entries.append(E_entry)

            Label(element_frame, text=f"τ{i+1} (s):", bg='#f0f0f0').grid(row=2, column=0, sticky='e', padx=5)
            tau_entry = Entry(element_frame, width=15)
            tau_val = self.tau_i_default[i] if i < len(self.tau_i_default) else 1e-3
            tau_entry.insert(0, str(tau_val))
            tau_entry.grid(row=2, column=1, padx=5, pady=2)
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

    def load_master_curve(self):
        """Load master curve data from file"""
        filename = filedialog.askopenfilename(
            title="Load Master Curve Data",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            # Try to load data (expected format: freq, E', E")
            data = np.loadtxt(filename, delimiter=',', skiprows=1)

            if data.shape[1] >= 3:
                self.master_freq = data[:, 0]  # log10(f)
                self.master_E_prime = data[:, 1]  # log10(E')
                self.master_E_double = data[:, 2]  # log10(E")

                messagebox.showinfo("Success", f"Loaded {len(self.master_freq)} data points")
                self.update_modulus_plot()
            else:
                messagebox.showerror("Error", "File must have at least 3 columns: log10(f), log10(E'), log10(E\")")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def fit_to_master_curve(self):
        """Fit model parameters to master curve data"""
        if self.master_freq is None:
            messagebox.showerror("Error", "Please load master curve data first")
            return

        try:
            # Get current number of elements
            n_elem = self.n_elements

            # Define objective function
            def objective(params):
                E0 = params[0]
                E_i = params[1:n_elem+1]
                tau_i = params[n_elem+1:2*n_elem+1]

                model = ViscoelasticModeler(E0, E_i, tau_i)

                freq = 10**self.master_freq
                omega = 2 * np.pi * freq

                E_prime_pred = np.array([model.storage_modulus(w) for w in omega])
                E_double_pred = np.array([model.loss_modulus(w) for w in omega])

                E_prime_target = 10**self.master_E_prime
                E_double_target = 10**self.master_E_double

                # Mean squared error in log scale
                error = np.sum((np.log10(E_prime_pred + 1e-10) - self.master_E_prime)**2) + \
                       np.sum((np.log10(E_double_pred + 1e-10) - self.master_E_double)**2)

                return error

            # Set bounds for parameters
            E_target_max = 10**np.max(self.master_E_prime)
            bounds = [(1, E_target_max)]  # E0

            for i in range(n_elem):
                bounds.append((10, E_target_max))  # E_i

            for i in range(n_elem):
                bounds.append((1e-10, 1e5))  # tau_i

            messagebox.showinfo("Fitting", "Fitting in progress... This may take a while.")

            # Use differential evolution for global optimization
            result = differential_evolution(objective, bounds, maxiter=100, popsize=15, seed=42)

            # Extract fitted parameters
            E0_fit = result.x[0]
            E_i_fit = result.x[1:n_elem+1]
            tau_i_fit = result.x[n_elem+1:2*n_elem+1]

            # Update GUI with fitted parameters
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

        # Create model
        model = ViscoelasticModeler(E0, E_i, tau_i)

        # Generate frequency range
        freq_log = np.linspace(freq_min, freq_max, 1000)
        freq = 10**freq_log
        omega = 2 * np.pi * freq

        # Calculate moduli
        E_prime = np.array([model.storage_modulus(w) for w in omega])
        E_double_prime = np.array([model.loss_modulus(w) for w in omega])

        # Plot
        self.plot_ax.clear()
        self.plot_ax.plot(freq_log, np.log10(E_prime), 'r-', linewidth=2.5, label="Storage modulus E'")
        self.plot_ax.plot(freq_log, np.log10(E_double_prime), 'g-', linewidth=2.5, label='Loss modulus E"')

        # Plot master curve data if available
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
