#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOVING CYLINDERS VORTEX STREET OPTIMIZATION
@author: Philipp Krah
"""

###############################################################################
# %% IMPORTED MODULES
###############################################################################
import sys

sys.path.append("./../src/WABBIT-opt/LIB")
sys.path.append("../src/sPOD/lib")

import numpy as np
from numpy import exp, mod, meshgrid, pi, sin, size, cos
import matplotlib.pyplot as plt
from utils import *
import glob
from IO import read_ACM_dat
from transforms import transforms
from tabulate import tabulate
from sPOD_tools import (
    frame,
    reconstruction_error,
    shifted_POD_J2,
    shifted_rPCA,
    sPOD_Param,
    shifted_POD_BFB,
    build_all_frames,
    load_frames,
    save_frames,
    give_interpolation_error,
)
from farge_colormaps import farge_colormap_multi
from plot_utils import show_animation, save_fig
from IO import load_trajectories
import matplotlib
from os.path import expanduser
import os
from farge_colormaps import farge_colormap_multi
from IPython.display import HTML

fc = farge_colormap_multi(etalement_du_zero=0.2, limite_faible_fort=0.5)

ROOT_DIR = os.path.dirname(os.path.abspath("README.md"))
home = expanduser("~")

font = {"family": "normal", "weight": "bold", "size": 24}

matplotlib.rc("font", **font)
###############################################################################
cm = farge_colormap_multi(etalement_du_zero=0.2, limite_faible_fort=0.5)


###############################################################################
def path(mu_vec, time, freq):
    return mu_vec[0] * cos(2 * pi * freq * time)


def dpath(mu_vec, time, freq):
    return pi * freq * (-2 * mu_vec[0] * sin(2 * pi * freq * time))


def give_shift(time, x, mu_vec, freq):
    shift = np.zeros([len(x), len(time)])
    for it, t in enumerate(time):
        shift[..., it] = path(mu_vec, np.heaviside(x, 0) * (x) - t, freq)
    return shift


def my_interpolated_state(
    sPOD_frames, frame_amplitude_list, mu_points, Ngrid, Nt, mu_vec
):
    from scipy.interpolate import griddata

    print(mu_vec)

    shift1 = np.zeros([2, Nt])
    shift2 = np.zeros([2, Nt])
    shift2[1, :] = -path(mu_vec)
    shiftsnew = [shift1, shift2]

    qtilde = 0
    for shift, frame, amplitudes in zip(shiftsnew, sPOD_frames, frame_amplitude_list):

        Modes = frame.modal_system["U"]
        VT = []
        for k in range(frame.Nmodes):
            a = griddata(mu_points.T, amplitudes[k].T, mu_vec)
            VT.append(np.squeeze(a))
        VT = np.asarray(VT)
        Q = Modes[:, : frame.Nmodes] @ VT
        qframe = np.reshape(Q, [*Ngrid, 1, Nt])
        trafo = transforms(
            [*Ngrid, 1, Nt],
            frame.trafo.domain_size,
            shifts=shift,
            dx=frame.trafo.dx,
            use_scipy_transform=True,
        )
        qtilde += trafo.apply(qframe)

    return qtilde


def compute_offline_errors(
    q,
    trafos,
    methods=["srPCA"],
    maxmodes=None,
    Niter=40,
    eps=1e-13,
    skip_existing=True,
    mu0=None,
    lambd0=None,
    lambda_s=1,
    case="vortex_street",
):

    [N, M] = np.shape(q)
    # lambd0 = mu0*5e2
    if mu0 is None:
        mu0 = N * M / (4 * np.sum(np.abs(q)))
    if lambd0 is None:
        lambd0 = 1 / np.sqrt(np.maximum(M, N))

    err_dict = {}
    for method in methods:
        if method == "srPCA":
            sPOD_log_dir = ddir + "/srPCA/" + case + "_mu0_%.3e" % mu0 + "/"
            if skip_existing and os.path.exists(sPOD_log_dir):
                Nframes = len(trafos)
                qframes, E = load_frames(
                    sPOD_log_dir + "frame.npy", Nframes, load_ErrMat=True
                )
                qtilde = build_all_frames(qframes)
            else:
                ret = shifted_rPCA(
                    q,
                    trafos,
                    nmodes_max=np.max(nmodes) + 100,
                    eps=eps,
                    Niter=Niter,
                    use_rSVD=True,
                    lambd=lambd0,
                    mu=mu0,
                    isError=True,
                )
                os.makedirs(
                    sPOD_log_dir, exist_ok=True
                )  # succeeds even if directory exists.
                qframes, qtilde, rel_err_list = (
                    ret.frames[:],
                    np.reshape(ret.data_approx, data_shape),
                    ret.rel_err_hist,
                )
                E = np.reshape(ret.error_matrix, data_shape)
                save_frames(sPOD_log_dir + "frame.npy", qframes, error_matrix=E)

            log_save = sPOD_log_dir + "err_mat_srPCA.out"
            nModes = np.asarray([frame.Nmodes for frame in qframes])
            if maxmodes is None:
                nmodes_max = nModes
            else:
                nmodes_max = np.asarray([maxmodes for frame in qframes])
            # err_mat = reconstruction_error(q, qframes, trafos, max_ranks=nModes)
            # ret = force_constraint(qframes.copy(), trafos, q, Niter=1, alphas = [1,0])
            err_mat_rPCA = reconstruction_error(
                q, ret.frames, trafos, max_ranks=nmodes_max
            )

            err_dict[method] = err_mat_rPCA
            np.savetxt(log_save, err_mat_rPCA, delimiter=",")

        if method == "sPOD":
            if maxmodes is None:
                nmodes_max = int(np.max(nModes))
            else:
                nmodes_max = maxmodes
            norm_q = np.linalg.norm(q, ord="fro")
            max_dof = np.sum(np.asarray(nmodes_max) * 2 + 1)
            err_mat = 2 * np.ones_like(err_mat_rPCA)
            for k in range(0, np.shape(err_mat_rPCA)[0]):
                nModes = np.asarray(err_mat_rPCA[k, :2], dtype=np.int32)
                print("mode combi: [", *nModes, "]\n")
                rank_text = "_".join(map(str, nModes))
                sPOD_log_dir = ddir + "/sPOD/" + case + "_ranks_" + rank_text + "/"
                if skip_existing and os.path.exists(sPOD_log_dir):
                    Nframes = len(trafos)
                    qframes = load_frames(
                        sPOD_log_dir + "frame.npy", Nframes, load_ErrMat=False
                    )
                    qtilde = build_all_frames(qframes)
                else:
                    ret = shifted_POD_J2(
                        q,
                        trafos,
                        nmodes=nModes,
                        eps=1e-13,
                        Niter=Niter,
                        use_rSVD=False,
                        dtol=1e-3,
                    )
                    print(
                        "\n\nmodes: [",
                        *nModes,
                        "] error = %4.4e \n\n" % ret.rel_err_hist[-1],
                    )
                    os.makedirs(
                        sPOD_log_dir, exist_ok=True
                    )  # succeeds even if directory exists.
                    qframes, qtilde, rel_err_list = (
                        ret.frames,
                        np.reshape(ret.data_approx, data_shape),
                        ret.rel_err_hist,
                    )
                    save_frames(sPOD_log_dir + "frame.npy", qframes)

                rel_err = (
                    np.linalg.norm(np.reshape(qtilde, np.shape(q)) - q, ord="fro")
                    / norm_q
                )
                dof = np.sum(nModes)
                if rel_err < err_mat[dof, -1]:
                    err_mat[dof, -1] = rel_err
                    for ir, r in enumerate(nModes):
                        err_mat[dof, ir] = r

            log_save = sPOD_log_dir + "err_mat_sPOD.out"

            err_dict[method] = err_mat
            np.savetxt(log_save, err_mat, delimiter=",")

        if method == "BFB":
            if method == "BFB":
                BFB_log_dir = ddir + "/BFB/" + case + "_lambda_s_%.3e" % lambda_s + "/"
            if skip_existing and os.path.exists(BFB_log_dir):
                Nframes = len(trafos)
                qframes, E = load_frames(
                    BFB_log_dir + "frame.npy", Nframes, load_ErrMat=True
                )
                qtilde = build_all_frames(qframes)
            else:
                myparams = sPOD_Param(maxit=Niter, lambda_s=lambda_s)
                ret = shifted_POD_BFB(q, trafos, nmodes, myparams)
                os.makedirs(
                    BFB_log_dir, exist_ok=True
                )  # succeeds even if directory exists.
                qframes, qtilde, rel_err_list = (
                    ret.frames[:],
                    np.reshape(ret.data_approx, data_shape),
                    ret.rel_err_hist,
                )

            log_save = BFB_log_dir + "err_mat_srPCA.out"
            nModes = np.asarray([frame.Nmodes for frame in qframes])
            if maxmodes is None:
                nmodes_max = nModes
            else:
                nmodes_max = np.asarray([maxmodes for frame in qframes])
            # err_mat = reconstruction_error(q, qframes, trafos, max_ranks=nModes)
            # ret = force_constraint(qframes.copy(), trafos, q, Niter=1, alphas = [1,0])
            err_mat_BFB = reconstruction_error(
                q, ret.frames, trafos, max_ranks=nmodes_max
            )

            err_dict[method] = err_mat_BFB
            np.savetxt(log_save, err_mat_BFB, delimiter=",")

    return err_dict


##########################################
# %%% Define your DATA:
##########################################
plt.close("all")
# ddir = ROOT_DIR+"/../data"
# ddir = "../data/1params_opt/"
ddir = "../../sPOD-data/WABBIT/data/2cylinder"
idir = "../images/vortex"
case = "vortex_street"
shift_type = "general"
# shift_type = "naive"
skip_existing = False  # True #False
frac = 1  # fraction of grid points to use
time_frac = 4
use_general_shift = True

time_sum = []
ux_list = []
uy_list = []
mu_vec_list = [[16]]
ux, uy, mask, p, time, Ngrid, dx, L = read_ACM_dat(
    ddir + "/ALL_2cyls_mu16.mat", sample_fraction=4, time_sample_fraction=2
)
time_sum = time
# %%

Nt = np.size(ux, -1)  # Number of time intervalls
Nvar = 1  # data.shape[0]                    # Number of variables
nmodes = [40, 40]  # reduction of singular values
Ngrid = np.shape(ux[..., 0])

# number of grid points in x
data_shape = [*Ngrid, Nvar, Nt]
# size of time intervall
freq0 = 0.01 / 5
Radius = 1
T = time[-1]
C_eta = 2.5e-3
x, y = (np.linspace(0, L[i] - dx[i], Ngrid[i]) for i in range(2))
dX = (x[1] - x[0], y[1] - y[0])
dt = time[1] - time[0]
[Y, X] = meshgrid(y, x)
fd = finite_diffs(Ngrid, dX)

if True:
    vort = np.asarray(
        [
            fd.rot(ux[..., nt], uy[..., nt]) for nt in range(np.size(ux, 2))
        ]  # ux as shifsted truncated
    )
    vort = np.moveaxis(vort, 0, -1)
if False:
    show_animation(
        vort.swapaxes(0, 1),
        Xgrid=[x, y],
        use_html=False,
        vmin=-1,
        vmax=1,
        save_path="images",
    )

# %% general_shift with wake correction:
if shift_type == "general":
    shift1 = np.zeros([2, np.prod(Ngrid), Nt])
    shift2_general = np.zeros([2, np.prod(Ngrid), Nt])
    # shift1 = np.zeros([2,Nsnapshots])
    # shift2 = np.zeros([2,Nsnapshots])

    # shift1[0,:] = 0 * time_joint                      # frame 1, shift in x
    # shift1[1,:] = 0 * time_joint                      # frame 1, shift in y
    # shift2[0,:] = 0 * time_joint                      # frame 2, shift in x
    y_shifts = []
    dy_shifts = []
    for mu_vec in mu_vec_list:
        print("mu = " + str(mu_vec))
        y_shifts.append(
            -give_shift(time, X.flatten() - L[0] / 2 - Radius, mu_vec, freq0)
        )
        # y_shifts.append(-path(mu_vec, time, freq0)) # frame 2, shift in y
        dy_shifts.append(dpath(mu_vec, time, freq0))  # frame 2, shift in y
    dy_shifts = np.concatenate([*dy_shifts], axis=-1)
    shift2_general[1, ...] = np.concatenate([*y_shifts], axis=-1)

    shift_trafo_1 = transforms(
        data_shape,
        L,
        shifts=shift1,
        trafo_type="identity",
        dx=dX,
        use_scipy_transform=False,
    )
    shift_trafo_2 = transforms(
        data_shape,
        L,
        shifts=shift2_general,
        dx=dX,
        use_scipy_transform=False,
        interp_order=5,
    )
    trafos = [shift_trafo_1, shift_trafo_2]

else:
    # %% naive shift
    shift1 = np.zeros([2, Nt])
    shift2 = np.zeros([2, Nt])
    shift1[0, :] = 0 * time_sum  # frame 1, shift in x
    shift1[1, :] = 0 * time_sum  # frame 1, shift in y
    shift2[0, :] = 0 * time_sum  # frame 2, shift in x
    y_shifts = []
    dy_shifts = []
    for mu_vec in mu_vec_list:
        y_shifts.append(-path(mu_vec, time, freq0))  # frame 2, shift in y
        dy_shifts.append(dpath(mu_vec, time, freq0))  # frame 2, shift in y
    dy_shifts = np.concatenate(dy_shifts, axis=0)
    shift2[1, :] = np.concatenate(y_shifts, axis=0)

    shift_trafo_1 = transforms(
        data_shape,
        L,
        shifts=shift1,
        trafo_type="identity",
        dx=dX,
        use_scipy_transform=False,
    )
    shift_trafo_2 = transforms(
        data_shape, L, shifts=shift2, dx=dX, use_scipy_transform=False, interp_order=5
    )
    trafos = [shift_trafo_1, shift_trafo_2]
# %% interp err

# interp_err_list_naive = []
# for q in [ux,uy,p,mask]:
#     qmat = np.reshape(q,[-1,Nt])
#     err = give_interpolation_error(qmat,trafos_naive[1])
#     interp_err_list_naive.append(err)

# interp_err_list_wake = []
# for q in [ux, uy, p, mask]:
#     qmat = np.reshape(q, [-1, Nt])
#     err = give_interpolation_error(qmat, trafos_wake[1])
#     interp_err_list_wake.append(err)

# table = [["interp. err. naive", *interp_err_list_naive],["interp. err. wake corr.", *interp_err_list_wake]]
# table_string = tabulate(table,headers=["","$u_1$","$u_2$","$p$", "$\chi$"], tablefmt='latex_booktabs',floatfmt=("", "1.1e", "1.1e", "1.1e","1.1e"), numalign="center")
# print(table_string)


vort_shift = shift_trafo_2.reverse(vort)
# %%

# show_animation(vort_shift.swapaxes(0,1),Xgrid = [x,y],use_html=False,vmin=-1,vmax=1,save_path="../images/")
# %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                        velocity ux
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
q_ux = np.reshape(ux, [-1, Nt])
q_uy = np.reshape(uy, [-1, Nt])
Niter = 200
[N, M] = np.shape(q_ux)
# mu0 = N * M / (4 * np.sum(np.abs(q))) * 5.0e-03  # * 0.001
lambd0 = 1 / np.sqrt(np.maximum(M, N)) * 20
lambda_s = 1  # 0.1

############################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# q_ux = np.reshape(ux, [-1, Nt])
# ret_vort_ux = shifted_rPCA(
#     q_ux,
#     trafos,
#     nmodes_max=np.max(nmodes) + 500,
#     eps=1e-13,
#     Niter=Niter,
#     use_rSVD=True,
#     lambd=lambd0,
#     mu=N * M / (4 * np.sum(np.abs(q_ux))) * 5.0e-03,
#     isError=True,
# )

# qframes_ux = ret_vort_ux.frames

# qtilde_ux = np.zeros_like(q)
# qk_ux = []
# ranks_ux = [1, 1]
# for k, (trafo, frame) in enumerate(zip(trafos, qframes_ux)):
#     qtilde_ux += trafo.apply(frame.build_field(ranks_ux[k]))
#     qk_ux.append(
#         np.reshape(trafo.apply(frame.build_field(ranks_ux[k])), (128, 128, 251))
#     )

# qtilde_ux = np.reshape(qtilde_ux, (128, 128, 251))

# q_uy = np.reshape(uy, [-1, Nt])
# ret_vort_uy = shifted_rPCA(
#     q_uy,
#     trafos,
#     nmodes_max=np.max(nmodes) + 500,
#     eps=1e-13,
#     Niter=Niter,
#     use_rSVD=True,
#     lambd=lambd0,
#     mu=N * M / (4 * np.sum(np.abs(q_uy))) * 4.0e-04,
#     isError=True,
# )

# qframes_uy = ret_vort_uy.frames

# qtilde_uy = np.zeros_like(q)
# qk_uy = []
# ranks_uy = [1, 1]
# for k, (trafo, frame) in enumerate(zip(trafos, qframes_uy)):
#     qtilde_uy += trafo.apply(frame.build_field(ranks_uy[k]))
#     qk_uy.append(
#         np.reshape(trafo.apply(frame.build_field(ranks_uy[k])), (128, 128, 251))
#     )

# qtilde_uy = np.reshape(qtilde_uy, (128, 128, 251))


# vort = np.asarray(
#     [
#         fd.rot(qk_ux[1][..., nt], qk_uy[1][..., nt])
#         for nt in range(np.size(qtilde_ux, 2))
#     ]  # ux as shifsted truncated
# )
# vort = np.moveaxis(vort, 0, -1)


# show_animation(
#     vort.swapaxes(0, 1),
#     Xgrid=[x, y],
#     use_html=False,
#     vmin=-1,
#     vmax=1,
#     save_path="images",
# )

#############################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


################################################################################################################
# ret_list = []
# linestyles = ["--", "-.", ":", "-", "-."]
# plot_list = []
# fig, ax = plt.subplots(num=87)
# factors = [0.01, 0.1, 1, 10, 100]
# for ip, fac in enumerate(factors):  # ,400, 800]):#,800,1000]: #
#     lambd_sig = lambda_s * fac
#     print("START WITH LAMBDA", lambd_sig)
#     myparams = sPOD_Param(maxit=Niter, lambda_s=lambd_sig)
#     ret = shifted_POD_BFB(q, trafos, nmodes, myparams)
#     ret_list.append(ret)

#     h = ax.semilogy(
#         np.arange(0, np.size(ret.rel_err_hist)),
#         ret.rel_err_hist,
#         linestyles[ip],
#         label="sPOD-BFB $\lambda_{\sigma}=10^{%d}\lambda^0_{\sigma}$"
#         % int(np.log10(fac)),
#     )

#     plt.text(
#         Niter,
#         ret.rel_err_hist[-1],
#         "$(r_1,r_2)=(%d,%d)$" % (ret.ranks[0], ret.ranks[1]),
#         transform=ax.transData,
#         va="bottom",
#         ha="right",
#     )
# plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower right")
# ax.set_xlim(-5, ax.get_xlim()[-1])
# plt.ylabel(r"relative error")
# plt.xlabel(r"iteration")
# # fig.set_tight_layout(True)
# plt.tight_layout()
# plt.show()


###################################################################################################################

rel_err_dict_ux = compute_offline_errors(
    q_ux,
    trafos,
    methods=["srPCA", "sPOD"],  # ["srPCA", "sPOD"],
    eps=1e-9,
    Niter=Niter,
    skip_existing=skip_existing,
    mu0=N * M / (4 * np.sum(np.abs(q_ux))) * 5.0e-03,
    lambd0=lambd0,
    lambda_s=lambda_s,
    case="ux",
)
rel_err_dict_uy = compute_offline_errors(
    q_uy,
    trafos,
    methods=["srPCA", "sPOD"],  # ["srPCA", "sPOD"],
    eps=1e-9,
    Niter=Niter,
    skip_existing=skip_existing,
    mu0=N * M / (4 * np.sum(np.abs(q_uy))) * 4.0e-04,
    lambd0=lambd0,
    lambda_s=lambda_s,
    case="uy",
)
err_mat_srPCA_ux = rel_err_dict_ux["srPCA"]
err_mat_sPOD_ux = rel_err_dict_ux["sPOD"]

err_mat_srPCA_uy = rel_err_dict_uy["srPCA"]
err_mat_sPOD_uy = rel_err_dict_uy["sPOD"]
# err_mat_BFB = rel_err_dict["BFB"]
# %%
nModes_ux = err_mat_srPCA_ux[-1, :2]
maxrank_ux = int(np.sum(nModes_ux))
nModes_uy = err_mat_srPCA_uy[-1, :2]
maxrank_uy = int(np.sum(nModes_uy))
# [_, ss_ux, _] = np.linalg.svd(q_ux, full_matrices=False)
# err_POD = list(np.sqrt(1 - (np.cumsum(ss_ux[:-1] ** 2)) / np.sum(ss_ux[:-1] ** 2)))
# err_POD.insert(0, 1)
# plot it
# fig, ax = plt.subplots(num=201)
# ax.semilogy(err_mat_srPCA[:-1, -1], "*", label="sPOD-$\mathcal{J}_1$")
# ax.semilogy(err_mat_srPCA2[:-1,-1], '*', label="sPOD-$\mathcal{J}_1$")
# ax.semilogy(err_mat_sPOD[:-1, -1], "+", label="sPOD-$\mathcal{J}_2$")
# ax.semilogy(err_POD[: maxrank - 1], "x", label="POD")
# ax.semilogy(err_mat_BFB[:-1, -1], "o", label="BFB")
# plt.legend(loc=1)
# plt.show()

nmodes_max_show = 60

err_mat_srPCA_ux = rel_err_dict_ux["srPCA"]
err_mat_sPOD_ux = rel_err_dict_ux["sPOD"]
err_mat_srPCA_uy = rel_err_dict_uy["srPCA"]
err_mat_sPOD_uy = rel_err_dict_uy["sPOD"]
# err_mat_BFB = rel_err_dict_ux["BFB"]
# print("AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH, matrices")
# print("PCA")
# print(err_mat_srPCA)
# print("sPOD")
# print(err_mat_sPOD)
# print("BFB")
# print(err_mat_BFB)
# nModes = (np.max(err_mat_sPOD[:, :2]) + 1) * 2
# maxrank = int(np.sum(nModes))
# [_, ss, _] = np.linalg.svd(q, full_matrices=False)
# err_POD = list(np.sqrt(1 - (np.cumsum(ss[:-1] ** 2)) / np.sum(ss[:-1] ** 2)))
# err_POD.insert(0, 1)
# fig, ax = plt.subplots(num=201)
# err_mat_srPCA[1, -1] = err_mat_sPOD[1, -1]
# ax.semilogy(err_mat_srPCA[:-1, -1], "*", label="ADM")
# # ax.semilogy(err_mat_srPCA2[:-1,-1], '*', label="sPOD-$\mathcal{J}_1$")
# ax.semilogy(err_mat_sPOD[:-1, -1], "+", label="J2")
# ax.semilogy(err_POD[: maxrank - 1], "x", label="POD")
# ax.semilogy(err_mat_BFB[:-1, -1], "o", label="BFB")
# plt.legend(loc=1)
# plt.xlim([-1, nmodes_max_show])
# plt.ylim([1e-2, 1.1])
# plt.grid(which="both", linestyle="--")
# save_fig("images/error_uy_two_cycl.png", fig)
# plt.show()

#####################################################################################

matching_tuples = []
for row1 in err_mat_srPCA_ux:
    for row2 in err_mat_srPCA_uy:
        if row1[2] == row2[2]:
            matching_tuples.append(
                ((int(row1[0]), int(row1[1])), (int(row2[0]), int(row2[1])))
            )

print("AHHHHHHHHHHHHHHHHHHHHHHHHHHHHH", matching_tuples)
with open("matching_tuples.txt", "w") as file:
    for pair in matching_tuples:
        file.write(f"{pair}\n")


# ret_vort_ux = shifted_rPCA(
#     q_ux,
#     trafos,
#     nmodes_max=np.max(nmodes) + 500,
#     eps=1e-13,
#     Niter=Niter,
#     use_rSVD=True,
#     lambd=lambd0,
#     mu=N * M / (4 * np.sum(np.abs(q_ux))) * 5.0e-03,
#     isError=True,
# )

# qframes_ux = ret_vort_ux.frames

# qtilde_ux = np.zeros_like(q_ux)
# qk_ux = []
# ranks_ux = [1, 1]
# for k, (trafo, frame) in enumerate(zip(trafos, qframes_ux)):
#     qtilde_ux += trafo.apply(frame.build_field(ranks_ux[k]))
#     qk_ux.append(
#         np.reshape(trafo.apply(frame.build_field(ranks_ux[k])), (128, 128, 251))
#     )

# qtilde_ux = np.reshape(qtilde_ux, (128, 128, 251))

# ret_vort_uy = shifted_rPCA(
#     q_uy,
#     trafos,
#     nmodes_max=np.max(nmodes) + 500,
#     eps=1e-13,
#     Niter=Niter,
#     use_rSVD=True,
#     lambd=lambd0,
#     mu=N * M / (4 * np.sum(np.abs(q_uy))) * 4.0e-04,
#     isError=True,
# )

# qframes_uy = ret_vort_uy.frames

# qtilde_uy = np.zeros_like(q)
# qk_uy = []
# ranks_uy = [1, 1]
# for k, (trafo, frame) in enumerate(zip(trafos, qframes_uy)):
#     qtilde_uy += trafo.apply(frame.build_field(ranks_uy[k]))
#     qk_uy.append(
#         np.reshape(trafo.apply(frame.build_field(ranks_uy[k])), (128, 128, 251))
#     )

# qtilde_uy = np.reshape(qtilde_uy, (128, 128, 251))


# vort = np.asarray(
#     [
#         fd.rot(qk_ux[1][..., nt], qk_uy[1][..., nt])
#         for nt in range(np.size(qtilde_ux, 2))
#     ]  # ux as shifsted truncated
# )
# vort = np.moveaxis(vort, 0, -1)


# show_animation(
#     vort.swapaxes(0, 1),
#     Xgrid=[x, y],
#     use_html=False,
#     vmin=-1,
#     vmax=1,
#     save_path="images",
# )


#####################################################################################


DOFs = np.arange(0, nmodes_max_show - 1)

idx_show = [
    1,
    2,
    3,
    8,
    9,
    10,
    15,
    20,
    30,
    40,
]

dofs_show = [DOFs[i] for i in idx_show]
ePOD = [err_POD[i] for i in idx_show]
eBFB = [err_mat_BFB[i, -1] for i in idx_show]
esPOD = [err_mat_sPOD[i, -1] for i in idx_show]
esrPCA = [err_mat_srPCA[i, -1] for i in idx_show]

table = np.stack((dofs_show, esPOD, esrPCA, eBFB, ePOD), axis=1)
table_string = tabulate(
    table,
    headers=["DOF", "shifted POD", "shifted srPCA", "BFB", "POD"],
    tablefmt="latex_booktabs",
    floatfmt=(".0f", ".3f", ".3f", ".3f"),
)
print(table_string)

with open("tabl/error-uy-two-cyls-BFB-3000.tabl.tex", "w") as text_file:
    text_file.write(table_string)
