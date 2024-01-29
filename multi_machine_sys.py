"""
This module calculates the oscillation modes for the IEEE-WSCC multimachine System
for a 9 nodes system.

Author: Daniel Zapata Yarce - Cód. 1004965048
Electric Systems Stability - Technological University
April 2023
"""

import numpy as np
from y_bus import Ybus
import pandas as pd
import matplotlib.pyplot as plt


class MultiMachine:
    def __init__(self, num_nodes, ybus, powerFlow, gens_info, load_nodes, freq=60):
        """

        :param num_nodes: total system nodes
        :param ybus: Ybus system matriz
        :param powerFlow: array with power flow information (voltages, currents and powers)
        :param gens_info: Array with generator number, transitory reactance and H value
        :param load_nodes: list with every load node
        :param freq: System frequency. 60 Hz by default
        """
        self.freq = freq
        self.load_nodes = load_nodes
        self.n_nodes = num_nodes
        self.ybus = ybus
        self.gens_info = gens_info
        self.gen_N = self.gens_info[:, 0]
        self.gen_imp = self.gens_info[:, 1]
        self.gen_volts = np.zeros((len(self.gen_N), 1),
                                  dtype=np.complex_)  # Contains internal tensions for every generator (E_k)
        self.voltages = powerFlow[:, 0]
        self.currents = powerFlow[:, 1]
        self.powers = powerFlow[:, 2]
        self.z_loads = np.zeros((self.n_nodes, 1), dtype=np.complex_)
        self.get_zloads()
        self.get_machine_model()
        self.loads_to_Ybus()
        self.addxprim_toYbus()
        self.get_kron()

    def get_kron(self):
        """

        :return:
        """
        y_gg = self.ybus[self.n_nodes:, self.n_nodes:]
        y_gn = self.ybus[self.n_nodes:, :self.n_nodes]
        y_nn = np.linalg.inv(self.ybus[: self.n_nodes, :self.n_nodes])
        y_ng = self.ybus[:self.n_nodes, self.n_nodes:]
        self.ybus = y_gg - y_gn @ y_nn @ y_ng

    def addxprim_toYbus(self):
        """

        :return:
        """
        for k in self.gen_N:
            k = int(k)
            # Expand Ybus in one zeros row and column
            zeros_row = np.zeros((len(self.ybus[0, :]), 1))
            self.ybus = np.append(self.ybus, zeros_row, axis=1)
            zeros_col = np.zeros((1, len(self.ybus[0, :])))
            self.ybus = np.append(self.ybus, zeros_col, axis=0)
            # Add out diagonal elements
            self.ybus[k - 1, -1] = -1 / (1j * self.gen_imp[k - 1])  # Add in last column
            self.ybus[-1, k - 1] = -1 / (1j * self.gen_imp[k - 1])  # Add in last row
            # Add diagonal element
            self.ybus[-1, -1] = 1 / (1j * self.gen_imp[k - 1])

            # Update Ybus diagonal
            self.ybus[k - 1, k - 1] += 1 / (1j * self.gen_imp[k - 1])

    def get_machine_model(self):
        """
        E_k = V_k + jX'I_k
        This module assume that generator nodes are first nodes in the system and calculates its internal voltage.
        :return: None
        """
        self.gen_volts = self.voltages[:len(self.gen_N)] + 1j * self.gen_imp * self.currents[:len(self.gen_N)]

    def get_zloads(self):
        """
        Calculates loads impedances using:
            Z_k = V_k ** 2 / (P_k - jQ_k)
        :return: None
        """
        for k in self.load_nodes:
            self.z_loads[k - 1] = abs(self.voltages[k - 1]) ** 2 / (
                    abs(np.real(self.powers[k - 1])) - 1j * abs(np.imag(self.powers[k - 1])))

    def get_Mmatrix(self):
        w0 = 2 * np.pi * self.freq
        vector_H = self.gens_info[:, 2]
        diagonal_M = 2 * vector_H / w0
        M_matrix = np.diag(diagonal_M)

        return M_matrix

    def get_Ximatrix(self):
        xi_vector = [1, 1, 1]
        xi_matrix = np.diag(xi_vector)

        return xi_matrix

    def get_Tmatrix(self):
        t_vector = [1, 1, 1]
        t_matrix = np.diag(t_vector)

        return t_matrix

    def get_Kmatrix(self):
        k1 = 1
        k2 = 1
        k3 = 1
        k_vector = [k1, k2, k3]
        k_matrix = np.diag(k_vector)

        return k_matrix

    def get_Amatrx(self):
        """
        TODO: Take K, T and Xi from gen info
        :return:
        """
        M = self.get_Mmatrix()
        Xi = self.get_Ximatrix()
        T = self.get_Tmatrix()
        K = self.get_Kmatrix()
        jac = self.get_jacobian()

        A_firstR = np.concatenate((-np.linalg.inv(M) @ Xi, -np.linalg.inv(M) @ jac, np.linalg.inv(M)), axis=1)
        A_secondR = np.concatenate((np.eye(len(self.gen_N)), np.zeros((len(self.gen_N), len(self.gen_N))),
                                    np.zeros((len(self.gen_N), len(self.gen_N)))), axis=1)
        A_thirdR = np.concatenate(
            (-np.linalg.inv(T) @ K, np.zeros((len(self.gen_N), len(self.gen_N))), -np.linalg.inv(T)), axis=1)

        A_matrix = np.concatenate((A_firstR, A_secondR, A_thirdR), axis=0)

        return A_matrix

    def get_osscilation_modes(self):
        """

        :return:
        """
        A_matrix = self.get_Amatrx()
        eig_values, _ = np.linalg.eig(A_matrix)

        return eig_values

    def loads_to_Ybus(self):
        """
        Expand Ybus diagonal adding system impedance load
        :return: None
        """
        for k in self.load_nodes:
            self.ybus[k - 1, k - 1] += 1 / self.z_loads[k - 1]

    def get_jacobian(self):
        """

        :return:
        """
        jacobian = np.zeros((len(self.gen_N), len(self.gen_N)))
        # Calculate out diagonal termns
        for i in range(len(self.gen_N)):
            for j in range(len(self.gen_N)):
                if i == j:
                    continue
                jacobian[i, j] = abs(self.gen_volts[i]) * abs(self.gen_volts[j]) * (-np.imag(self.ybus[i, j]) * np.cos(
                    np.angle(self.gen_volts[i]) - np.angle(self.gen_volts[j])) + np.real(self.ybus[i, j]) * np.sin(
                    np.angle(self.gen_volts[i]) - np.angle(self.gen_volts[j])))

        # Calculate diagonal elements
        for i in range(len(self.gen_N)):
            for j in range(len(self.gen_N)):
                if i == j:
                    continue
                jacobian[i, i] += abs(self.gen_volts[i]) * abs(self.gen_volts[j]) * (np.imag(self.ybus[i, j]) * np.cos(
                    np.angle(self.gen_volts[i]) - np.angle(self.gen_volts[j])) - np.real(self.ybus[i, j]) * np.sin(
                    np.angle(self.gen_volts[i]) - np.angle(self.gen_volts[j])))

        return jacobian

    def plot_oscillations(self):
        t = np.linspace(0, 20, 100)
        landa = self.get_osscilation_modes()
        for i in range(len(landa)):
            plt.plot(t, np.exp(np.real(landa[i]) * t), label='$\lambda_{}$'.format(i + 1))
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == '__main__':
    path = 'nodes.xlsx'
    nodes = pd.read_excel(path, sheet_name='lines')
    n_nodes = max(nodes.NodoRec.max(), nodes.NodoEnv.max())
    nodes_array = nodes.to_numpy()
    # Get Ybus matrix
    y_bus = Ybus(nodes_array, n_nodes).get_Ybus()
    # Get power flow and convert it to numpy array
    power_flow = pd.read_excel(path, sheet_name='PowerFlow', dtype=np.complex_, index_col=0)
    power_flow = power_flow.to_numpy(dtype=np.complex_)
    loads_nodes = [5, 6, 8]  # Define system load nodes
    gen_info = pd.read_excel(path, sheet_name="GenInfo")
    gen_info = gen_info.to_numpy()
    mmachine_sys = MultiMachine(n_nodes,y_bus.copy(), power_flow, gen_info, loads_nodes)
    mmachine_sys.plot_oscillations()
    print("Oscillation Modes: \n", mmachine_sys.get_osscilation_modes().reshape((9, 1)))
