"""
This module creates the Ybus matrix from a given numpy array that contains the PS information

Author: Daniel Zapata Yarce - Cód. 1004965048
Electrical Systems Stability - Technological University
April 2023
"""

import numpy as np
import pandas as pd


# Consider the next expressions for Ybus creation if 'yik' is primitive admitance value:
# Yii = sum(yik)
# Yik = -yik
class Ybus:
    def __init__(self, data, amntNodes):
        """
        Build a Ybus matrix from a numpy array with lines system information
        :param data: array with system information
        :param amntNodes: int. Nodes amount
        """
        self.data = data
        self.amntNodes = amntNodes
        # Initialize with zeros the Ybus matrix
        self.ybus = np.zeros((self.amntNodes, self.amntNodes), dtype=np.complex_)
        self.__out_diag__()
        self.__into_diag__()

    def __out_diag__(self):
        """
        Creat all out diagonal Ybus values.
        It supposes that 2 first columns are send and receive nodes.
        Third and fourth columns are resistance and impedance, respectively
        :return: None. All is internal
        """
        counter = 0  # Counter to get impedances positions
        # m is send node and k is receive node
        for m, k in zip(self.data[:, 0], self.data[:, 1]):
            m, k = int(m), int(k)  # We need nodes as integers?

            # Calculates  values:
            resistance = self.data[counter, 2]  # Resistance value
            reactance = (self.data[counter, 3] * 1j)  # Reactance value
            # Y = G + jB = 1 / (R + jX)  ==> Admitance Eq.
            y_primitive = 1 / (resistance + reactance)

            # Add to superior diagonal positions the admitances
            self.ybus[m - 1, k - 1] = -y_primitive
            # Add to inferior diagonal positions the admitances
            self.ybus[k - 1, m - 1] = -y_primitive
            counter += 1

    def __into_diag__(self):
        """
        Calculates diagonal values for Ybus matrix
        :return: None. Is internal
        """
        # We need to iterate every node in our system
        for i in range(self.amntNodes):
            b_shunts_env = 0
            b_shunts_rec = 0
            positions_env = list(np.where(self.data[:, 0] == i + 1))[0]
            positions_rec = list(np.where(self.data[:, 1] == i + 1))[0]

            for e in positions_env:
                b_shunts_env += self.data[e, -1] * 1j

            for r in positions_rec:
                b_shunts_rec += self.data[r, -1] * 1j

            prim_sums = np.sum(-self.ybus[i, :]) + b_shunts_env + b_shunts_rec
            self.ybus[i, i] = prim_sums

    def get_Ybus(self):
        """
        Get Ybus array
        :return: matrix with Ybus values
        """
        return self.ybus


if __name__ == '__main__':
    path = 'nodes.xlsx'
    nodes = pd.read_excel(path)
    n_nodes = max(nodes.NodoRec.max(), nodes.NodoEnv.max())
    impedances = nodes.to_numpy()  # Let's remember that 2 first columns are send and receives nodes
    # y_out = out_diag(impedances, n_nodes)
    y_bus = Ybus(impedances, n_nodes).get_Ybus()
