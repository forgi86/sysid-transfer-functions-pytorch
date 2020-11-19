import torch
from torchid_nb.module.lti import MimoLinearDynamicalOperator, SisoLinearDynamicalOperator
from torchid_nb.module.static import MimoStaticNonLinearity, MimoChannelWiseNonLinearity


class ParallelWHNet(torch.nn.Module):
    def __init__(self):
        super(ParallelWHNet, self).__init__()
        self.nb_1 = 12
        self.na_1 = 12
        self.nb_2 = 13
        self.na_2 = 12
        self.G1 = MimoLinearDynamicalOperator(1, 2, n_b=self.nb_1, n_a=self.na_1, n_k=1)
        self.F_nl = MimoChannelWiseNonLinearity(2, n_hidden=10)
        #self.F_nl = MimoStaticNonLinearity(2, 2, n_hidden=10)
        self.G2 = MimoLinearDynamicalOperator(2, 1, n_b=self.nb_2, n_a=self.na_2, n_k=0)
        #self.G3 = SisoLinearDynamicalOperator(n_b=3, n_a=3, n_k=1)

    def forward(self, u):
        y1_lin = self.G1(u)
        y1_nl = self.F_nl(y1_lin)  # B, T, C1
        y2_lin = self.G2(y1_nl)  # B, T, C2

        return y2_lin #+ self.G3(u)


class ParallelWHNetVar(torch.nn.Module):
    def __init__(self):
        super(ParallelWHNetVar, self).__init__()
        self.nb_1 = 3
        self.na_1 = 3
        self.nb_2 = 3
        self.na_2 = 3
        self.G1 = MimoLinearDynamicalOperator(1, 16, n_b=self.nb_1, n_a=self.na_1, n_k=1)
        self.F_nl = MimoStaticNonLinearity(16, 16) #MimoChannelWiseNonLinearity(16, n_hidden=10)
        self.G2 = MimoLinearDynamicalOperator(16, 1, n_b=self.nb_2, n_a=self.na_2, n_k=1)

    def forward(self, u):
        y1_lin = self.G1(u)
        y1_nl = self.F_nl(y1_lin)  # B, T, C1
        y2_lin = self.G2(y1_nl)  # B, T, C2

        return y2_lin


class ParallelWHResNet(torch.nn.Module):
    def __init__(self):
        super(ParallelWHResNet, self).__init__()
        self.nb_1 = 4
        self.na_1 = 4
        self.nb_2 = 4
        self.na_2 = 4
        self.G1 = MimoLinearDynamicalOperator(1, 2, n_b=self.nb_1, n_a=self.na_1, n_k=1)
        self.F_nl = MimoChannelWiseNonLinearity(2, n_hidden=10)
        self.G2 = MimoLinearDynamicalOperator(2, 1, n_b=self.nb_2, n_a=self.na_2, n_k=1)
        self.G3 = SisoLinearDynamicalOperator(n_b=6, n_a=6, n_k=1)

    def forward(self, u):
        y1_lin = self.G1(u)
        y1_nl = self.F_nl(y1_lin)  # B, T, C1
        y2_lin = self.G2(y1_nl)  # B, T, C2

        return y2_lin + self.G3(u)