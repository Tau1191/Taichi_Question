# The flow and reaction was solved by Lattice Boltzmann Method using Taichi language
# Auther: Tau (Lizt1191@gmail.com)
# Time: March 2023
import math

import taichi as ti
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import time

ti.init(arch=ti.gpu)


@ti.data_oriented
class Reaction_Flow:
    def __init__(self, Re):
        nx = 101
        ny = 101
        self.nx = nx
        self.ny = ny

        self.Co2 = 0
        self.CaO = 1
        self.CaCo3 = 2

        self.rho0 = 1.98  # 单位:kg/m3
        self.C_s0 = 45.0  # 单位:mol/m3
        self.C_s_CaO0 = 1.0
        self.C_s_CaCo30 = 1e-8
        self.T_Init = 930
        self.Lx = 1e-3  # 单位:m
        self.Ly = 1e-3
        self.D = 1.7e-4  # 100微米
        self.Particle_Num_X = 4
        self.Particle_Num_Y = 2
        self.Particle_Step_x = self.Lx / (self.Particle_Num_X + 1)
        self.Particle_Step_y = self.Ly / (self.Particle_Num_Y + 1)
        self.Cy_x = [self.Particle_Step_x, 2.0 * self.Particle_Step_x, 3.0 * self.Particle_Step_x,
                     4.0 * self.Particle_Step_x]
        self.Cy_y = [self.Particle_Step_y, 2 * self.Particle_Step_y]
        self.Re = Re

        self.niu_Co2 = 1.52e-5  # 单位:m2/s
        self.Ds_Co2 = 1.53e-5  # 单位:m2/s
        self.Lamda_Co2 = 0.0163  # 单位:W/m.k
        self.Lamda_CaCo3 = 1.2  # 单位:W/m.k
        self.rho_CaCo3 = 2200.0
        self.Cp_Co2 = 37.1  # 单位:j/mol.k
        self.Cp_CaO = 39.8
        self.Cp_CaCo3 = 83.1
        self.DeltaH = 1.86e5  # 单位J·mol^-1

        self.Porous_Epsilon = 0.2
        self.Fluid_Epsilon = 0.9999
        self.Epsilon0 = self.Porous_Epsilon
        self.dp = 0.01

        self.U0 = self.Re * self.D / self.niu_Co2
        self.tau_f = 0.8
        self.dx = self.Ly / self.ny
        self.dt = (self.tau_f - 0.5) * self.dx * self.dx / (3 * self.niu_Co2)
        self.C = self.dx / self.dt
        self.Cs = self.C / np.sqrt(3)
        self.Rc = 1 / self.C
        self.Rcc = self.Rc ** 2
        self.C2 = self.C ** 2
        self.Cs2 = self.Cs ** 2

        # self.U0 = self.Re * self.D / self.niu_Co2
        # self.C = 1.0
        # self.Cs = self.C / np.sqrt(3)
        # self.dx = self.Ly / self.ny
        # self.dt = self.dx / self.C
        # self.Rc = 1 / self.C
        # self.Rcc = self.Rc ** 2
        # self.C2 = self.C ** 2
        # self.Cs2 = self.Cs ** 2
        # self.tau_f = self.niu_Co2 / (self.Cs2 * self.dt) + 0.5

        self.Vel_Temp = self.C_s_Temp = None
        self.Start_Time = self.End_Time = None
        self.CaCo3_Total = self.Solid_Total = 0.0

        self.alpha = ti.field(dtype=ti.f64, shape=())

        self.rho = ti.field(dtype=ti.f64, shape=(nx, ny))
        self.C_s = ti.Vector.field(3, dtype=ti.f64, shape=(nx, ny))
        self.T = ti.field(dtype=ti.f64, shape=(nx, ny))
        self.vel = ti.Vector.field(2, dtype=ti.f64, shape=(nx, ny))
        self.Epsilon = ti.field(dtype=ti.f64, shape=2)
        self.K = ti.field(dtype=ti.f64, shape=2)

        self.fg = ti.Vector.field(9, dtype=ti.f64, shape=(nx, ny))
        self.f = ti.Vector.field(9, dtype=ti.f64, shape=(nx, ny))
        self.gg = ti.Vector.field(9, dtype=ti.f64, shape=(nx, ny))
        self.g = ti.Vector.field(9, dtype=ti.f64, shape=(nx, ny))
        self.hg = ti.Vector.field(9, dtype=ti.f64, shape=(nx, ny))
        self.h = ti.Vector.field(9, dtype=ti.f64, shape=(nx, ny))
        self.kr = ti.field(dtype=ti.f64, shape=(nx, ny))
        self.Phi_gi = ti.Vector.field(3, dtype=ti.f64, shape=(nx, ny))
        self.Phi_gc = ti.Vector.field(3, dtype=ti.f64, shape=(nx, ny))
        self.Phi_hi = ti.field(dtype=ti.f64, shape=(nx, ny))
        self.Phi_hc = ti.field(dtype=ti.f64, shape=(nx, ny))

        self.omega_i = ti.field(dtype=ti.f64, shape=9)
        self.rek = ti.field(dtype=ti.f64, shape=9)
        self.e = ti.field(dtype=ti.i32, shape=(9, 2))
        self.mask = ti.field(dtype=ti.i32, shape=(nx, ny))

        arr = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
        self.rek.from_numpy(arr)
        arr = np.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0,
                        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0], dtype=np.float32)
        self.omega_i.from_numpy(arr)
        arr = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1],
                        [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)
        self.e.from_numpy(arr)

    @ti.func
    def f_eq(self, i, j, k):
        eu = (ti.f64(self.e[k, 0]) * self.vel[i, j][0]
              + ti.f64(self.e[k, 1]) * self.vel[i, j][1]) * self.Rc
        # vel[i,j][0]表示横向速度，vel[i,j][1]表示纵向速度
        uv = (self.vel[i, j][0] ** 2.0 + self.vel[i, j][1] ** 2.0) * self.Rcc
        return self.omega_i[k] * self.rho[i, j] * \
            (1.0 + 3.0 * eu + 4.5 * eu ** 2 / self.Epsilon[self.mask[i, j]] - 1.5 * uv / self.Epsilon[self.mask[i, j]])

    @ti.func
    def f_m(self, i, j, l):
        Temp = -self.Epsilon[self.mask[i, j]] * self.niu_Co2 / self.K[self.mask[i, j]] * self.vel[i, j][l]
        return Temp

    @ti.func
    def f_i(self, i, j, k, UFm, VFm):
        eu = (ti.f64(self.e[k, 0]) * self.vel[i, j][0]
              + ti.f64(self.e[k, 1]) * self.vel[i, j][1]) * self.Rc
        ef = (ti.f64(self.e[k, 0]) * UFm + ti.f64(self.e[k, 1]) * VFm) * self.Rc
        uf = (self.vel[i, j][0] * self.vel[i, j][0]
              + self.vel[i, j][1] * self.vel[i, j][1]) * self.Rcc
        return self.omega_i[k] * self.rho[i, j] * (1 - 0.5 / self.tau_f) * \
            (3.0 * ef + 9 * eu * ef / self.Epsilon[self.mask[i, j]] - 3 * uf / self.Epsilon[self.mask[i, j]])

    @ti.func
    def g_eq(self, i, j, k):
        eu = (ti.f64(self.e[k, 0]) * self.vel[i, j][0]
              + ti.f64(self.e[k, 1]) * self.vel[i, j][1]) * self.Rc
        # vel[i,j][0]表示横向速度，vel[i,j][1]表示纵向速度
        geq = self.omega_i[k] * self.C_s[i, j][self.Co2] * (self.Epsilon0 + eu / self.Cs2)
        if k == 0:
            geq = self.C_s[i, j][self.Co2] * (self.Epsilon[self.mask[i, j]] - self.Epsilon0) + \
                  self.omega_i[k] * self.C_s[i, j][self.Co2] * (self.Epsilon0 + eu / self.Cs2)
        return geq
        #先定义geq=None,再用k来分类对geq赋值并在最后return，不被taichi语言所支持

    @ti.func
    def h_eq(self, i, j, k):
        eu = (ti.f64(self.e[k, 0]) * self.vel[i, j][0]
              + ti.f64(self.e[k, 1]) * self.vel[i, j][1]) * self.Rc
        # vel[i,j][0]表示横向速度，vel[i,j][1]表示纵向速度
        heq = self.omega_i[k] * self.T[i, j] * (self.Epsilon0 + eu / self.Cs2)
        if k == 0:
            heq = self.T[i, j] * (self.Epsilon[self.mask[i, j]] - self.Epsilon0) + \
                  self.omega_i[k] * self.T[i, j] * (self.Epsilon0 + eu / self.Cs2)
        return heq

    def para_init(self):
        self.Epsilon[0] = self.Fluid_Epsilon
        self.Epsilon[1] = self.Porous_Epsilon
        self.K[0] = self.dp**2*self.Epsilon[0]**3/(180*(1-self.Epsilon[0])**2)
        self.K[1] = self.dp**2*self.Epsilon[1]**3/(180*(1-self.Epsilon[1])**2)

    @ti.kernel
    def field_init(self):
        for i, j, m in ti.ndrange(self.nx, self.ny, 3):
            self.vel[i, j][0] = 0.0
            self.vel[i, j][1] = 0.0
            self.rho[i, j] = self.rho0
            self.C_s[i, j][m] = 0.0
            self.T[i, j] = self.T_Init
            self.mask[i, j] = 0

            for l in ti.static(range(4)):
                for n in ti.static(range(2)):
                    if ((ti.f64(i) * self.dx - self.Cy_x[l]) ** 2.0
                        + (ti.f64(j) * self.dx - self.Cy_y[n]) ** 2.0) < (self.D * 0.5) ** 2.0:
                        self.mask[i, j] = 1
                        self.C_s[i, j][self.CaO] = self.C_s_CaO0
                        self.C_s[i, j][self.CaCo3] = self.C_s_CaCo30
            self.C_s[0, j][self.Co2] = self.C_s0

            for k in ti.static(range(9)):
                self.f[i, j][k] = self.f_eq(i, j, k)
                self.g[i, j][k] = self.g_eq(i, j, k)
                self.h[i, j][k] = self.h_eq(i, j, k)

    @ti.kernel
    def collide(self):
        for i, j, k in ti.ndrange(self.nx, self.ny, 9):
            w_f = 1.0 / self.tau_f
            De = self.Ds_Co2 * self.Epsilon[self.mask[i, j]] / ti.pow(self.Epsilon[self.mask[i, j]], 0.5)
            tau_s = De / (self.Cs2 * self.dt * self.Epsilon0) + 0.5
            w_s = 1.0 / tau_s
            Lamda_Eff = self.Epsilon[self.mask[i,j]] * self.Lamda_Co2 + \
                        (1 - self.Epsilon[self.mask[i,j]]) * self.Lamda_CaCo3
            Rho_Cp_Eff = self.Epsilon[self.mask[i,j]] * self.rho0 * self.Cp_Co2 + \
                         (1 - self.Epsilon[self.mask[i,j]]) * self.rho_CaCo3 * self.Cp_CaCo3
            Sigma = Rho_Cp_Eff / (self.rho0 * self.Cp_Co2)
            tau_t = Lamda_Eff / (self.Cs2 * self.dt * self.rho0 * self.Cp_Co2 * Sigma) + 0.5
            w_t = 1.0 / tau_t
            self.Phi_gi[i, j][self.Co2] = self.omega_i[k] * (1 - 0.5 / tau_s) * self.Phi_gc[i, j][self.Co2]
            self.Phi_hi[i, j] = self.omega_i[k] * (1 - 0.5 / tau_t) * self.Phi_hc[i, j]
            self.fg[i, j][k] = self.f[i, j][k] - (self.f[i, j][k] - self.f_eq(i, j, k)) * w_f + \
                               self.omega_i[k] * self.dt * self.f_i(i, j, k, self.f_m(i, j, 0), self.f_m(i, j, 1))
            self.gg[i, j][k] = self.g[i, j][k] - (self.g[i, j][k] - self.g_eq(i, j, k)) * w_s + \
                               self.dt * self.Phi_gi[i, j][self.Co2]
            self.hg[i, j][k] = self.h[i, j][k] - (self.h[i, j][k] - self.h_eq(i, j, k)) * w_t + \
                               self.dt * self.Phi_hi[i, j]

    @ti.kernel
    def stream(self):
        for i, j, k in ti.ndrange((1, self.nx - 1), (1, self.ny - 1), 9):
            id = i - self.e[k, 0]
            jd = j - self.e[k, 1]

            self.f[i, j][k] = self.fg[id, jd][k]
            self.g[i, j][k] = self.gg[id, jd][k]
            self.h[i, j][k] = self.hg[id, jd][k]

    @ti.kernel
    def update_macro_var(self):  # compute rho u v
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.rho[i, j] = self.vel[i, j][0] = self.vel[i, j][1] = self.C_s[i, j][self.Co2] = self.T[i, j] = 0.0
            Rho_Cp_Eff = self.Epsilon[self.mask[i,j]] * self.rho0 * self.Cp_Co2 + \
                         (1 - self.Epsilon[self.mask[i,j]]) * self.rho_CaCo3 * self.Cp_CaCo3
            Sigma = Rho_Cp_Eff / (self.rho0 * self.Cp_Co2)
            ## 此处为何不用Taichi内置的.fill函数而采用逐个赋值的方法:那样会降低近两倍的效率
            for k in ti.static(range(9)):
                self.rho[i, j] += self.f[i, j][k]
                self.vel[i, j][0] += self.C * (ti.f64(self.e[k, 0]) *
                                               self.f[i, j][k])
                self.vel[i, j][1] += self.C * (ti.f64(self.e[k, 1]) *
                                               self.f[i, j][k])
                self.C_s[i, j][self.Co2] += (self.g[i, j][k] / self.Epsilon[self.mask[i, j]])
                self.T[i, j] += (self.h[i, j][k] / Sigma)
            if self.rho[i, j] >= 1e-9:
                self.vel[i, j][0] /= (self.rho[i, j] + (0.5 * self.dt * self.Epsilon[self.mask[i, j]] * self.niu_Co2 /
                                                        self.K[self.mask[i, j]] * self.rho[i, j]))
                self.vel[i, j][1] /= (self.rho[i, j] + (0.5 * self.dt * self.Epsilon[self.mask[i, j]] * self.niu_Co2 /
                                                        self.K[self.mask[i, j]] * self.rho[i, j]))
                self.C_s[i, j][self.Co2] += (
                        0.5 * self.dt * self.Phi_gc[i, j][self.Co2] / self.Epsilon[self.mask[i, j]])
                self.T[i, j] += (0.5 * self.dt * self.Phi_hc[i, j] / Sigma)
            else:
                self.vel[i, j][0] = 0.0
                self.vel[i, j][1] = 0.0
                self.C_s[i, j][self.Co2] = 0.0
                self.T[i, j] = 0.0
            if self.C_s[i, j][self.Co2] <= 1e-9:
                self.C_s[i, j][self.Co2] = 0.0

    @ti.kernel
    def Alpha_CaCo3_Cal(self) -> float:  # impose boundary conditions
        CaCo3_Total = ti.f64(0.0)
        Solid_Total = ti.f64(0.0)
        for i, j in ti.ndrange((0, self.nx), (1, self.ny - 1)):
            if self.mask[i, j] == 1:
                CaCo3_Total += self.C_s[i, j][self.CaCo3]
                Solid_Total += self.C_s_CaO0
        return CaCo3_Total / Solid_Total

    @ti.kernel
    def wall_bc(self):  # impose boundary conditions
        u_max = self.vel[self.nx - 1, 0][0]
        for j in ti.ndrange(self.ny):
            if u_max < self.vel[self.nx - 1, j][0]:
                u_max = self.vel[self.nx - 1, j][0]

        for j in ti.ndrange((1, self.ny - 1)):
            self.vel[0, j][0] = self.U0
            self.vel[0, j][1] = 0.0
            self.rho[0, j] = self.rho0  # self.rho[1, j]
            self.C_s[0, j][self.Co2] = self.C_s0
            self.T[0, j] = self.T_Init

            # self.vel[self.nx-1, j][0] = self.vel[self.nx - 2, j][0]
            # self.vel[self.nx-1, j][1] = self.vel[self.nx - 2, j][1]
            # self.rho[self.nx-1, j] = self.rho[self.nx - 2, j]
            self.vel[self.nx - 1, j][0] = self.vel[self.nx - 1, j][0] - self.dt / self.dx * u_max * (
                    self.vel[self.nx - 1, j][0] - self.vel[self.nx - 2, j][0])  # self.vel[self.nx - 1, j][0] #
            self.vel[self.nx - 1, j][1] = self.vel[self.nx - 1, j][1] - self.dt / self.dx * u_max * (
                    self.vel[self.nx - 1, j][1] - self.vel[self.nx - 2, j][1])  # self.vel[self.nx - 1, j][1] #
            self.rho[self.nx - 1, j] = self.rho[self.nx - 1, j] - self.dt / self.dx * u_max * (
                    self.rho[self.nx - 1, j] - self.rho[self.nx - 2, j])  # self.rho[self.nx - 1, j] #
            self.C_s[self.nx - 1, j][self.Co2] = self.C_s[self.nx - 2, j][self.Co2]
            self.T[self.nx - 1, j] = self.T[self.nx - 2, j]

            for k in ti.static(range(9)):
                self.f[0, j][k] = self.f_eq(0, j, k) + (self.f[1, j][k] - self.f_eq(1, j, k))
                self.f[self.nx - 1, j][k] = self.f_eq(self.nx - 1, j, k) + (self.f[self.nx - 2, j][k] - self.f_eq(self.nx - 2, j, k))
                self.g[0, j][k] = self.g_eq(0, j, k) + (self.g[1, j][k] - self.g_eq(1, j, k))
                self.g[self.nx - 1, j][k] = self.g_eq(self.nx - 1, j, k) + (self.g[self.nx - 2, j][k] - self.g_eq(self.nx - 2, j, k))
                self.h[0, j][k] = self.h_eq(0, j, k) + ( self.h[1, j][k] - self.h_eq(1, j, k))
                self.h[self.nx - 1, j][k] = self.h_eq(self.nx - 1, j, k) + (self.h[self.nx - 2, j][k] - self.h_eq(self.nx - 2, j, k))

        for i in ti.ndrange((0, self.nx)):
            self.vel[i, self.ny - 1][0] = self.vel[i, self.ny - 2][0]
            self.vel[i, self.ny - 1][1] = 0.0
            self.rho[i, self.ny - 1] = self.rho[i, self.ny - 2]
            self.C_s[i, self.ny - 1][self.Co2] = self.C_s[i, self.ny - 2][self.Co2]
            self.T[i, self.ny - 1] = self.T[i, self.ny - 2]

            self.vel[i, 0][0] = self.vel[i, 1][0]
            self.vel[i, 0][1] = 0.0
            self.rho[i, 0] = self.rho[i, 1]
            self.C_s[i, 0][self.Co2] = self.C_s[i, 1][self.Co2]
            self.T[i, 0] = self.T[i, 1]
            for k in ti.static(range(9)):
                self.f[i, self.ny - 1][k] = self.f_eq(i, self.ny - 1, k) + ( self.f[i, self.ny - 2][k] - self.f_eq(i, self.ny - 2, k))
                self.f[i, 0][k] = self.f_eq(i, 0, k) + ( self.f[i, 1][k] - self.f_eq(i, 1, k))
                self.g[i, self.ny - 1][k] = self.g_eq(i, self.ny - 1, k) + ( self.g[i, self.ny - 2][k] - self.g_eq(i, self.ny - 2, k))
                self.g[i, 0][k] = self.g_eq(i, 0, k) + ( self.g[i, 1][k] - self.g_eq(i, 1, k))
                self.h[i, self.ny - 1][k] = self.h_eq(i, self.ny - 1, k) + ( self.h[i, self.ny - 2][k] - self.h_eq(i, self.ny - 2, k))
                self.h[i, 0][k] = self.h_eq(i, 0, k) + ( self.h[i, 1][k] - self.h_eq(i, 1, k))

    @ti.kernel
    def source_cal(self, alpha: float):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            if self.mask[i, j] == 1 and self.C_s[i, j][self.Co2] / self.C_s0 >= 0.001 \
                    and self.C_s[i, j][self.CaO] >= 0.001:
                Teq = 12845 / (16.508 - ti.log((self.C_s[i, j][self.Co2] * 8.314 * self.T[i, j]) / 100000))
                A = 1.87e12  # 单位s^-1
                E = 1.87e5  # 单位J·mol^-1
                self.kr[i, j] = A * ti.exp(-E / 8.314 / self.T[i, j]) * (self.T[i, j] / Teq - 1)
            else:
                self.kr[i, j] = 0.0
            self.Phi_gc[i, j][self.Co2] = self.kr[i, j] * self.C_s[i, j][self.CaO]
            self.Phi_gc[i, j][self.CaO] = self.kr[i, j] * self.C_s[i, j][self.CaO]
            self.Phi_gc[i, j][self.CaCo3] = -self.kr[i, j] * self.C_s[i, j][self.CaO]
            self.Phi_hc[i, j] = -self.kr[i, j] * self.C_s[i, j][self.CaO] * self.DeltaH / (self.rho[i, j] * self.Cp_Co2)
            self.C_s[i, j][self.CaO] = self.C_s[i, j][self.CaO] + self.Phi_gc[i, j][self.CaO] * self.dt
            self.C_s[i, j][self.CaCo3] = self.C_s[i, j][self.CaCo3] + self.Phi_gc[i, j][self.CaCo3] * self.dt

    def solve(self):
        gui = ti.GUI('Reaction_Flow', (self.nx, 5 * self.ny))
        self.Start_Time = time.time()
        i = 0
        self.para_init()
        self.field_init()
        print("C= %f,Ma= %f, tau_f=%f" % (
            self.C, self.U0 / self.Cs, self.tau_f))
        while 1:
            i = i + 1
            if i % 1000 == 0:
                self.vel_Temp = self.vel.to_numpy()
                self.C_s_Temp = self.C_s.to_numpy()
            self.collide()
            self.stream()
            self.update_macro_var()
            self.wall_bc()
            self.source_cal(self.Alpha_CaCo3_Cal())
            if (i % 1000 == 0):
                vel = self.vel.to_numpy()
                dens = self.C_s.to_numpy()
                T = self.T.to_numpy()
                vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5
                dens_mag_Co2 = dens[:, :, self.Co2]
                dens_mag_CaO = dens[:, :, self.CaO]
                dens_mag_CaCo3 = dens[:, :, self.CaCo3]
                T_mag = T[:, :]
                vel_img = cm.plasma(vel_mag / 0.015)
                dens_img_Co2 = cm.plasma(dens_mag_Co2)
                dens_img_CaO = cm.plasma(dens_mag_CaO)
                dens_img_CaCo3 = cm.plasma(dens_mag_CaCo3)
                T_img = cm.plasma(T_mag)
                img = np.concatenate((T_img, dens_img_CaCo3, dens_img_CaO, dens_img_Co2, vel_img), axis=1)
                gui.set_image(img)
                gui.show()
            if i % 1000 == 0:
                self.End_Time = time.time()
                Nx_Temp = int(self.nx / 2)
                Ny_Temp = int(self.ny / 2)
                DD_Temp = int(self.D / self.dx / 2)
                Time = int(self.End_Time - self.Start_Time)
                All_Time = Time
                hou = Time / 3600
                Time = Time % 3600
                mint = Time / 60
                Time = Time % 60
                sec = Time
                U_Err_Temp = ((self.vel.to_numpy()[:, :, 0] - self.vel_Temp[:, :, 0]) ** 2 + (
                        self.vel.to_numpy()[:, :, 1] - self.vel_Temp[:, :, 1]) ** 2) ** 0.5
                U_Temp2 = (self.vel.to_numpy()[:, :, 0] ** 2 + self.vel.to_numpy()[:, :, 1] ** 2) ** 0.5
                C_s_Err_Temp = self.C_s.to_numpy()[:, :, self.Co2] - self.C_s_Temp[:, :, self.Co2]
                C_s_Temp2 = self.C_s.to_numpy()[:, :, self.Co2]
                U_Err = U_Err_Temp.sum() / U_Temp2.sum()
                C_s_Err = C_s_Err_Temp.sum() / C_s_Temp2.sum()
                print(
                    "U_center=%e,C_center=%e,U_Err=%e,C_Err=%e,alpha=%f,Step:%d,T_phy=%.3fs,Time: %02d %02d:%02d:%02d" % (
                        self.vel[Nx_Temp, Ny_Temp][0],
                        self.C_s[Nx_Temp, Ny_Temp - DD_Temp + 1][self.Co2],
                        U_Err,
                        C_s_Err,
                        self.Alpha_CaCo3_Cal(),
                        i,
                        float(i) * self.dt,
                        All_Time / i * 1000,
                        hou, mint, sec))
                if i % 10000 == 0:
                    print(self.C_s[11, 33][self.CaCo3], self.C_s[13, 33][self.CaCo3], self.C_s[15, 33][self.CaCo3],
                          self.C_s[17, 33][self.CaCo3], self.C_s[19, 33][self.CaCo3], self.C_s[21, 33][self.CaCo3],
                          self.C_s[51, 66][self.CaCo3], self.C_s[53, 66][self.CaCo3], self.C_s[55, 66][self.CaCo3],
                          self.C_s[57, 66][self.CaCo3], self.C_s[59, 66][self.CaCo3])
                    print(self.C_s[11, 33][self.CaO], self.C_s[13, 33][self.CaO], self.C_s[15, 33][self.CaO],
                          self.C_s[17, 33][self.CaO], self.C_s[19, 33][self.CaO], self.C_s[21, 33][self.CaO],
                          self.C_s[51, 66][self.CaO], self.C_s[53, 66][self.CaO], self.C_s[55, 66][self.CaO],
                          self.C_s[57, 66][self.CaO], self.C_s[59, 66][self.CaO])
                    print(self.C_s[11, 33][self.Co2], self.C_s[13, 33][self.Co2], self.C_s[15, 33][self.Co2],
                          self.C_s[17, 33][self.Co2], self.C_s[19, 33][self.Co2], self.C_s[21, 33][self.Co2],
                          self.C_s[51, 66][self.Co2], self.C_s[53, 66][self.Co2], self.C_s[55, 66][self.Co2],
                          self.C_s[57, 66][self.Co2], self.C_s[59, 66][self.Co2])
                    print(self.T[11, 33], self.T[13, 33], self.T[15, 33], self.T[17, 33], self.T[19, 33],
                          self.T[21, 33],
                          self.T[51, 66], self.T[53, 66], self.T[55, 66], self.T[57, 66], self.T[59, 66])
                    print(self.kr[11, 33], self.kr[13, 33], self.kr[15, 33], self.kr[17, 33], self.kr[19, 33],
                          self.kr[21, 33],
                          self.kr[51, 66], self.kr[53, 66], self.kr[55, 66], self.kr[57, 66], self.kr[59, 66])
                # if U_Err <= 1.0e-6 and C_s_Err < 1.0e-6:
                #     global C_s_End, Rho_End, Vel_End
                #     C_s_End = self.C_s.to_numpy()[:, :, self.Co2]
                #     Rho_End = self.rho.to_numpy()[:, :]
                #     Vel_End = self.vel.to_numpy()[:, :, :]
                #
                #     fig, Comp2 = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=200)
                #     x_lbm = np.linspace(10, 320.0, 31)
                #     x_analyse = np.linspace(1, 320, 320)
                #     Comp2.plot(x_lbm, (C_s_End()[10:320:10, 2] - C_s_End()[10:320:10, 1]) * 320,
                #                'b^',
                #                label='LBM Solution')
                #     Comp2.plot(x_analyse, 0.854 * (0.06 * 320 * 320 / x_analyse * 6) ** (1 / 3),
                #                'r-',
                #                label='Analysis Solution')
                #     Comp2.set_xlabel(r'Y/Ly')
                #     Comp2.set_ylabel(r'U')
                #     Comp2.legend()
                #     plt.tight_layout()
                #     plt.show()
                #     break


if __name__ == '__main__':
    # Re_Init = 0.001
    Re_Init = 0.3
    # while 1:
    #     Reaction_Flow(Re_Init).solve()
    #     Re_Init += 1
    #     # RF_Solve = Reaction_Flow(Re_Init)
    #     # RF_Solve.solve()
    #     Average=np.average(C_s_End)
    #     if Average > 1:
    #         break
    Reaction_Flow(Re_Init).solve()
