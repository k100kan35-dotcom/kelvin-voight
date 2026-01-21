#!/usr/bin/env python3
"""
Viscoelastic Rubber Compound Modeler
Generalized Maxwell Model for Complex Modulus Calculation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, Frame, Label, Entry, Button


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
        self.E_i = np.array(E_i)
        self.tau_i = np.array(tau_i)

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


class ViscoelasticGUI:
    def __init__(self, master):
        self.master = master
        master.title("고무 컴파운드 점탄성 모델링 (Generalized Maxwell Model)")
        master.geometry("1400x800")

        # Default parameters (초기값)
        self.n_elements = 4  # Number of Maxwell elements
        self.E0_default = 10.0  # MPa
        self.E_i_default = [1000.0, 5000.0, 10000.0, 15000.0]  # MPa
        self.tau_i_default = [1e-6, 1e-4, 1e-2, 1.0]  # seconds

        self.freq_min = -10  # log10(Hz)
        self.freq_max = 20   # log10(Hz)

        # Create GUI layout
        self.create_widgets()
        self.update_plot()

    def create_widgets(self):
        # Main container
        main_frame = Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Parameters
        left_panel = Frame(main_frame, width=400, relief=tk.RIDGE, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

        # Title
        title_label = Label(left_panel, text="모델 파라미터", font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)

        # Scrollable frame for parameters
        canvas = tk.Canvas(left_panel)
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # E0 parameter
        Label(scrollable_frame, text="평형 탄성계수 E₀ (MPa):", font=('Arial', 11, 'bold')).pack(pady=(10, 5))
        self.E0_entry = Entry(scrollable_frame, width=20)
        self.E0_entry.insert(0, str(self.E0_default))
        self.E0_entry.pack()

        # Frequency range
        Label(scrollable_frame, text="주파수 범위 (log₁₀ Hz):", font=('Arial', 11, 'bold')).pack(pady=(20, 5))

        freq_frame = Frame(scrollable_frame)
        freq_frame.pack()
        Label(freq_frame, text="최소:").grid(row=0, column=0, padx=5)
        self.freq_min_entry = Entry(freq_frame, width=10)
        self.freq_min_entry.insert(0, str(self.freq_min))
        self.freq_min_entry.grid(row=0, column=1, padx=5)

        Label(freq_frame, text="최대:").grid(row=0, column=2, padx=5)
        self.freq_max_entry = Entry(freq_frame, width=10)
        self.freq_max_entry.insert(0, str(self.freq_max))
        self.freq_max_entry.grid(row=0, column=3, padx=5)

        # Maxwell elements
        Label(scrollable_frame, text="Maxwell 요소 파라미터:", font=('Arial', 11, 'bold')).pack(pady=(20, 10))

        self.E_entries = []
        self.tau_entries = []

        for i in range(self.n_elements):
            element_frame = Frame(scrollable_frame, relief=tk.GROOVE, borderwidth=2)
            element_frame.pack(fill=tk.X, padx=10, pady=5)

            Label(element_frame, text=f"요소 {i+1}", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2, pady=5)

            Label(element_frame, text=f"E{i+1} (MPa):").grid(row=1, column=0, sticky='e', padx=5)
            E_entry = Entry(element_frame, width=15)
            E_entry.insert(0, str(self.E_i_default[i]))
            E_entry.grid(row=1, column=1, padx=5, pady=2)
            self.E_entries.append(E_entry)

            Label(element_frame, text=f"τ{i+1} (s):").grid(row=2, column=0, sticky='e', padx=5)
            tau_entry = Entry(element_frame, width=15)
            tau_entry.insert(0, str(self.tau_i_default[i]))
            tau_entry.grid(row=2, column=1, padx=5, pady=2)
            self.tau_entries.append(tau_entry)

        # Update button
        update_btn = Button(scrollable_frame, text="그래프 업데이트", command=self.update_plot,
                           bg='#4CAF50', fg='white', font=('Arial', 12, 'bold'), pady=10)
        update_btn.pack(pady=20, fill=tk.X, padx=20)

        # Reset button
        reset_btn = Button(scrollable_frame, text="초기값으로 리셋", command=self.reset_parameters,
                          bg='#ff9800', fg='white', font=('Arial', 10), pady=5)
        reset_btn.pack(pady=(0, 20), fill=tk.X, padx=20)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Right panel - Plot
        right_panel = Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
            print(f"Error reading parameters: {e}")
            return None

    def reset_parameters(self):
        """Reset all parameters to default values"""
        self.E0_entry.delete(0, tk.END)
        self.E0_entry.insert(0, str(self.E0_default))

        for i, (E_entry, tau_entry) in enumerate(zip(self.E_entries, self.tau_entries)):
            E_entry.delete(0, tk.END)
            E_entry.insert(0, str(self.E_i_default[i]))

            tau_entry.delete(0, tk.END)
            tau_entry.insert(0, str(self.tau_i_default[i]))

        self.freq_min_entry.delete(0, tk.END)
        self.freq_min_entry.insert(0, str(self.freq_min))

        self.freq_max_entry.delete(0, tk.END)
        self.freq_max_entry.insert(0, str(self.freq_max))

        self.update_plot()

    def update_plot(self):
        """Update the plot with current parameters"""
        params = self.get_parameters()
        if params is None:
            return

        E0, E_i, tau_i, freq_min, freq_max = params

        # Create model
        model = ViscoelasticModeler(E0, E_i, tau_i)

        # Generate frequency range (log scale)
        freq_log = np.linspace(freq_min, freq_max, 1000)
        freq = 10**freq_log  # Hz
        omega = 2 * np.pi * freq  # rad/s

        # Calculate moduli
        E_prime = np.array([model.storage_modulus(w) for w in omega])
        E_double_prime = np.array([model.loss_modulus(w) for w in omega])

        # Plot
        self.ax.clear()
        self.ax.plot(freq_log, np.log10(E_prime), 'r-', linewidth=2, label="Storage modulus E'")
        self.ax.plot(freq_log, np.log10(E_double_prime), 'g-', linewidth=2, label='Loss modulus E"')

        self.ax.set_xlabel('log₁₀ f (Hz)', fontsize=12)
        self.ax.set_ylabel('log₁₀ E (MPa)', fontsize=12)
        self.ax.set_title('Complex Modulus of Viscoelasticity\n(복소 탄성계수)', fontsize=14, fontweight='bold')
        self.ax.legend(fontsize=11)
        self.ax.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()


def main():
    root = tk.Tk()
    app = ViscoelasticGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
