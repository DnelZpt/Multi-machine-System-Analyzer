"""
This module get the power-flow for a given Electrical Power System

Author: Daniel Zapata Yarce - Cód. 1004965048
Electrical Systems Stability - Technological University
April 2023
"""
import pandas as pd
import numpy as np
from y_bus import Ybus


class PowerFlow:
    def __init__(self, system_imp, N_nodes, powers, genr_nodes, vi):
        """
        Creates a Power Flow for a given system.
        :param system_imp: array system lines information
        :param N_nodes: int Number of nodes
        :param powers: vector with nodal system powers (P + jQ)
        :param genr_nodes: int Number of generators nodes including Slack node.
        :param vi: list with the magnitude voltage values for Slack an PV nodes
        """
        self.N_nodes = N_nodes
        self.gen_Nodes = genr_nodes
        # Set initial values
        # First column are magnitudes and second colum are angles
        self.delta_values = 0
        self.nodes_voltages = np.zeros((self.N_nodes, 2))
        self.nodes_voltages[:, 0] = 1
        self.nodes_voltages[:self.gen_Nodes, 0] = vi
        self.Ybus = Ybus(system_imp, self.N_nodes).get_Ybus()
        self.nodes_kind = self.set_nodes()
        self.esp_powers = np.concatenate((np.real(powers), np.imag(powers)), axis=0)
        self.esp_powers = np.delete(self.esp_powers, self.get_delPos(), axis=0)
        self.calc_powers = np.zeros((self.N_nodes, 1), dtype=np.complex_)
        self.newton_raphson()

    def newton_raphson(self):
        """
        Iterative method that solves Power Flow problem
        :return: None
        """
        n = 0
        iter_lim = 1000
        tolerance = 1E-6
        while n < iter_lim:
            # print('Current Iteration:', n)
            delta_powers = self.get_deltaPowers()
            error = float(max(np.abs(delta_powers)))
            # Decides if is moment to break the while loop
            if error <= tolerance:
                break
            jacobian = self.getJacobian()
            self.delta_values = np.linalg.inv(jacobian) @ delta_powers
            # Update phase values
            for i in range(1, self.N_nodes):
                self.nodes_voltages[i, 1] += self.delta_values[i - 1, 0]
                # TODO: generalize it to any Slack node
            # Update voltage magnitude values
            for i in range(len(self.delta_values) - (self.N_nodes - 1)):
                # Rember that we have deltaV/V, so we need to clear delta V
                delta_vi = self.delta_values[self.N_nodes - 1 + i, 0] * self.nodes_voltages[self.gen_Nodes + i, 0]
                self.nodes_voltages[self.gen_Nodes + i, 0] += delta_vi
            n += 1

    def get_deltaPowers(self):
        """
        Calculates delta powers
        :return: row vector with all delta powers
        """
        delta_powers = self.esp_powers - self.get_calcPowersCorr()

        return delta_powers

    def get_calcPowersCorr(self):
        """
        Organize actives and reactives powers in one vector and
        elimates items in non-necessary positions
        :return: array with calc powers to get deltas
        """
        active_p = self.calc_activePower()
        reactive_p = self.calc_reactivePower()
        calc_powers = np.concatenate((active_p, reactive_p), axis=0)
        calc_powers = np.delete(calc_powers, self.get_delPos(), axis=0)

        return calc_powers

    def calc_activePower(self):
        """
        Calculates active power.
        :return: vector with active powers
        """
        for i in range(self.N_nodes):
            p_i = self.nodes_voltages[i, 0] ** 2 * np.real(self.Ybus[i, i])
            for j in range(self.N_nodes):
                if i == j:
                    continue
                p_i += abs(self.Ybus[i, j]) * self.nodes_voltages[i, 0] * self.nodes_voltages[j, 0] * np.cos(
                    np.angle(self.Ybus[i, j]) + self.nodes_voltages[j, 1] - self.nodes_voltages[i, 1])

            self.calc_powers.real[i] = p_i

        return np.real(self.calc_powers)

    def calc_reactivePower(self):
        """
         Calculates reactive power.
        :return: vector with reactive powers
        """
        for i in range(self.N_nodes):
            q_i = - self.nodes_voltages[i, 0] ** 2 * np.imag(self.Ybus[i, i])
            for j in range(self.N_nodes):
                if i == j:
                    continue
                q_i -= abs(self.Ybus[i, j]) * self.nodes_voltages[i, 0] * self.nodes_voltages[j, 0] * np.sin(
                    np.angle(self.Ybus[i, j]) + self.nodes_voltages[j, 1] - self.nodes_voltages[i, 1])

            self.calc_powers.imag[i] = q_i

        return np.imag(self.calc_powers)

    def get_delPos(self):
        """
        Get positions for no necessary nodes in Jacobian and delta Powers arrays
        :return: a list with positions to delete
        """
        slack_node = list(self.nodes_kind)[list(self.nodes_kind.values()).index('Slack')]
        positions = [slack_node - 1, slack_node + self.N_nodes - 1]  # Contains nodes positions to delete in J
        for i in list(self.nodes_kind.keys()):
            if self.nodes_kind[i] == 'PV':
                positions.append(self.N_nodes + i - 1)

        return positions

    def getJacobian(self):
        """
        Calculates Jacobian Matrix
            | H   M'|
        J = |       |
            | N   K'|
        :return: Jacobian
        """
        H = self.get_Hjac()
        M_prim = self.get_Mprimjac()
        N = self.get_Njac()
        K_prim = self.getg_Kprimjac()
        first_row = np.concatenate((H, M_prim), axis=1)
        second_row = np.concatenate((N, K_prim), axis=1)
        jacobian = np.concatenate((first_row, second_row), axis=0)
        # Delete values associated to Slack and PV nodes
        del_nodes = self.get_delPos()
        jacobian = np.delete(jacobian, del_nodes, axis=1)
        jacobian = np.delete(jacobian, del_nodes, axis=0)

        return jacobian

    def get_Hjac(self):
        """
        Calculates H matrix for Jacobian
        :return: H matrix
        """
        H = np.zeros((self.N_nodes, self.N_nodes))
        Q = self.calc_reactivePower()
        # For Diagonal Elements
        for i in range(self.N_nodes):
            H[i, i] = - Q[i, 0] - self.nodes_voltages[i, 0] ** 2 * np.imag(self.Ybus[i, i])

        # For non-diagonal Elements
        for i in range(self.N_nodes):
            for j in range(self.N_nodes):
                if i == j:
                    continue  # Jump to next iteration if we are evaluating the same node
                H[i, j] = -self.nodes_voltages[i, 0] * self.nodes_voltages[j, 0] * (
                        np.imag(self.Ybus[i, j]) * np.cos(self.nodes_voltages[i, 1] - self.nodes_voltages[j, 1])
                        - np.real(self.Ybus[i, j]) * np.sin(self.nodes_voltages[i, 1] - self.nodes_voltages[j, 1])
                )

        return H

    def get_Mprimjac(self):
        """
        Get M' matrix for Jacobian
        :return: M' matrix
        """

        M_prim = np.zeros((self.N_nodes, self.N_nodes))
        P = self.calc_activePower()
        # For Diagonal Elements
        for i in range(self.N_nodes):
            M_prim[i, i] = P[i, 0] + self.nodes_voltages[i, 0] ** 2 * np.real(self.Ybus[i, i])

        # For no-diagonal Elements
        for i in range(self.N_nodes):
            for j in range(self.N_nodes):
                if i == j:
                    continue  # Jump to next iteration if we are evaluating the same node
                M_prim[i, j] = self.nodes_voltages[i, 0] * self.nodes_voltages[j, 0] * (
                        np.real(self.Ybus[i, j]) * np.cos(self.nodes_voltages[i, 1] - self.nodes_voltages[j, 1])
                        + np.imag(self.Ybus[i, j]) * np.sin(self.nodes_voltages[i, 1] - self.nodes_voltages[j, 1])
                )

        return M_prim

    def get_Njac(self):
        """
        Calculates N matrix for Jacobian
        :return: N matrix
        """
        N = np.zeros((self.N_nodes, self.N_nodes))
        P = self.calc_activePower()
        # For Diagonal Elements
        for i in range(self.N_nodes):
            N[i, i] = P[i, 0] - self.nodes_voltages[i, 0] ** 2 * np.real(self.Ybus[i, i])

        # For non-diagonal Elements
        for i in range(self.N_nodes):
            for j in range(self.N_nodes):
                if i == j:
                    continue  # Jump to next iteration if we are evaluating the same node
                N[i, j] = -self.nodes_voltages[i, 0] * self.nodes_voltages[j, 0] * (
                        np.real(self.Ybus[i, j]) * np.cos(self.nodes_voltages[i, 1] - self.nodes_voltages[j, 1])
                        + np.imag(self.Ybus[i, j]) * np.sin(self.nodes_voltages[i, 1] - self.nodes_voltages[j, 1])
                )

        return N

    def getg_Kprimjac(self):
        """
        Calculates K' matrix for Jacobian
        :return: K' matrix
        """
        K_prim = np.zeros((self.N_nodes, self.N_nodes))
        Q = self.calc_reactivePower()
        # For Diagonal Elements
        for i in range(self.N_nodes):
            K_prim[i, i] = Q[i, 0] - self.nodes_voltages[i, 0] ** 2 * np.imag(self.Ybus[i, i])

        # For no-diagonal Elements
        for i in range(self.N_nodes):
            for j in range(self.N_nodes):
                if i == j:
                    continue  # Jump to next iteration if we are evaluating the same node
                K_prim[i, j] = self.nodes_voltages[i, 0] * self.nodes_voltages[j, 0] * (
                        -np.imag(self.Ybus[i, j]) * np.cos(self.nodes_voltages[i, 1] - self.nodes_voltages[j, 1])
                        + np.real(self.Ybus[i, j]) * np.sin(self.nodes_voltages[i, 1] - self.nodes_voltages[j, 1])
                )

        return K_prim

    def set_nodes(self):
        """
        Sets every node as Slack, Generator(PV) or Load (PQ).
        Slack node usually is 1, PV are number of generator nodes and
        load nodes are: N - N_G - 1
        :return: None, is internal
        """
        # load_nodes = self.N_nodes - self.gen_Nodes - 1
        nodesKind = {1: 'Slack'}
        # Add PV nodes to dictionary
        if self.gen_Nodes != 0:  # View if there is generators nodes
            for i in range(2, self.gen_Nodes + 1):
                nodesKind.update({i: "PV"})

        # Add PQ nodes
        for i in range(1 + self.gen_Nodes, self.N_nodes + 1):
            nodesKind.update({i: "PQ"})

        return nodesKind

    def rect_nodalVolt(self):
        """
        Get a nodal voltages vector but with rectangular complex (for operations)
        :return: Vector with nodal voltages in cartesian form
        """
        nVoltagesRect = self.nodes_voltages[:, 0] * np.exp(1j * self.nodes_voltages[:, 1])

        return nVoltagesRect.reshape((len(nVoltagesRect), 1))

    def export_data(self, xlsx_path, sheet_name='PowerFlow'):
        """
        Exports Power Flow data in an Excel Sheet to give it a better presentation
        :param xlsx_path: xlsx file to export the DF
        :param sheet_name: default 'PowerFlow'. Sheet name
        :return: None
        """
        list_nodes = list(range(1, self.N_nodes + 1))
        data = {"Voltages [p.u.]": self.rect_nodalVolt()[:, 0],
                "Currents [p.u.]": self.get_currents()[:, 0],
                "Powers [p.u.]": self.calc_powers[:, 0]}
        # Convert the last dictionary in a pandas Data Frame
        data_frame = pd.DataFrame(data, index=list_nodes, dtype=np.complex_)
        # Add information to an existing Excel file
        with pd.ExcelWriter(xlsx_path, mode='a', if_sheet_exists='replace') as writer:
            data_frame.to_excel(writer, sheet_name=sheet_name)

    def get_currents(self):
        """
        Calculates de Ibus vector
        :return: Ibus
        """
        v_bus = self.rect_nodalVolt()
        i_bus = self.Ybus @ v_bus

        return i_bus


if __name__ == '__main__':
    path = 'nodes.xlsx'
    lines = pd.read_excel(path)
    n_nodes = max(lines.NodoRec.max(), lines.NodoEnv.max())
    impedances = lines.to_numpy()  # Let's remember that 2 first columns are send and receives nodes
    nodes = pd.read_excel(path, sheet_name='power')
    powers_array = nodes.to_numpy(dtype=np.complex_)[:, 1].reshape((len(nodes['Node']), 1))
    gen_nodes = 3
    initial_voltages = [1.04, 1.025, 1.025]
    power_flow = PowerFlow(impedances, n_nodes, powers_array, gen_nodes, initial_voltages)
    power_flow.export_data(path)
