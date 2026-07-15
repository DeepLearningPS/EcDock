class molchecksetting:
    def __init__(self):
        self.max_atoms=60
        self.max_mw=600
        self.allowed_atoms=[6,7,8,9,15,16,17,35,53]
        self.allowed_fcs=[0]
        self.max_ring_num_per_node=5
        self.max_single_ring_size=8
        self.avoid_substructs=[
                "[*;r8]",
                "[*;r9]",
                "[*;r10]",
                "[*;r11]",
                "[*;r12]",
                "[*;r13]",
                "[*;r14]",
                "[*;r15]",
                "[*;r16]",
                "[*;r17]",
                "[#8][#8]",
                "[#6;+]",
                "[#16][#16]",
                "[#7;!n][S;!$(S(=O)=O)]",
                "[#7;!n][#7;!n]",
                "C#C",
                "C(=[O,S])[O,S]",
                "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
                "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
                "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
                "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
                "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
                "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]"
            ]
        self.ring_cover_rate = 0.99