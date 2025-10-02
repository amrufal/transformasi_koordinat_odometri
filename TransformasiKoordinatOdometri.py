import math
import time
import numpy as np
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ====================================================
# ================== KONFIGURASI =====================
# ====================================================
# Path objek di scene CoppeliaSim
BASE_PATH              = "/PioneerP3DX"
RIGHT_JOINT_PATH       = "/rightMotor"
LEFT_JOINT_PATH        = "/leftMotor"
RIGHT_WHEEL_SHAPE_PATH = "/rightMotor/rightWheel"

# Mode stepping (True: Python memanggil sim.step())
STEPPING = True

# ---- PARAMETER EKSPERIMEN 
ROT_DEG = 90.0        # Sudut rotasi (derajat), default 90° CCW
TX      = 2.0         # Translasi sumbu X (meter)
TY      = 3.0         # Translasi sumbu Y (meter)
# ==============================================


# ====================================================
# ============ UTILITAS ODOMETRI & SCENE =============
# ====================================================

def wrap_pi(a: float) -> float:
    """Bungkus sudut ke rentang [-pi, pi)."""
    return math.atan2(math.sin(a), math.cos(a))

def read_params_from_scene(sim):
    """Ambil R roda, L_half, dt scene, dan handle objek dari scene."""
    base_h  = sim.getObject(BASE_PATH)
    rJ      = sim.getObject(RIGHT_JOINT_PATH)
    lJ      = sim.getObject(LEFT_JOINT_PATH)
    rWheel  = sim.getObject(RIGHT_WHEEL_SHAPE_PATH)

    dt_scene = sim.getSimulationTimeStep()

    # L_half dari selisih koordinat y kedua joint pada frame base
    pL = sim.getObjectPosition(lJ, base_h)  # [x_B, y_B, z_B]
    pR = sim.getObjectPosition(rJ, base_h)
    L_half = 0.5 * abs(pR[1] - pL[1])

    # R dari diameter primitive cylinder roda: dims[0] = diameter
    result, pureType, dims = sim.getShapeGeomInfo(rWheel)
    if not isinstance(dims, (list, tuple)) or len(dims) < 1:
        raise RuntimeError("getShapeGeomInfo tidak mengembalikan dims yang valid.")
    R = 0.5 * float(dims[0])

    return R, L_half, dt_scene, rJ, lJ, base_h

def get_gt_pose2d(sim, base_h):
    """Pose Ground-Truth (world): (x,y,yaw)."""
    x, y, _ = sim.getObjectPosition(base_h, -1)
    ax, ay, az = sim.getObjectOrientation(base_h, -1)  # yaw = az
    return x, y, wrap_pi(az)

def run_until_stop(sim, rJ, lJ, base_h, R, L_half, log_to_sim=True, stepping=True):
    """
    Loop odometri mengikuti status simulasi; berhenti saat simulation_stopped.
    """
    # ODO lokal
    x_o = y_o = th_o = 0.0

    # Pose awal GT untuk merelatifkan GT → frame lokal awal
    x0, y0, th0 = get_gt_pose2d(sim, base_h)
    c0, s0 = math.cos(th0), math.sin(th0)

    # Histori
    t_hist=[]; x_o_hist=[]; y_o_hist=[]; yaw_o_hist=[]
    x_gt_hist=[]; y_gt_hist=[]; yaw_gt_hist=[]
    ex_hist=[];  ey_hist=[];  eth_hist=[]

    # Waktu simulasi awal
    t_start = sim.getSimulationTime()
    t_prev  = t_start

    if log_to_sim:
        sim.addLog(1, f"[ODO] start (stepping={stepping}, ROT_DEG={ROT_DEG}, TX={TX}, TY={TY})")

    while True:
        state = sim.getSimulationState()

        # STOP
        if state == sim.simulation_stopped:
            if log_to_sim: sim.addLog(1, "[ODO] sim stopped → plotting")
            break

        # PAUSE
        if (state & sim.simulation_paused) != 0:
            time.sleep(0.02)
            continue

        # RUN
        if stepping:
            sim.step()
            t_now = sim.getSimulationTime()
            dt_k  = t_now - t_prev
            if dt_k <= 0.0:
                continue
        else:
            t_now = sim.getSimulationTime()
            dt_k  = t_now - t_prev
            if dt_k <= 0.0:
                time.sleep(0.01)
                continue

        # ---- ODOMETRI (differential drive) ----
        wr = sim.getJointVelocity(rJ)  # rad/s
        wl = sim.getJointVelocity(lJ)  # rad/s
        vr = wr * R
        vl = wl * R
        v  = 0.5 * (vr + vl)
        omega = (vr - vl) / (2.0 * L_half)

        x_o  += v * math.cos(th_o) * dt_k
        y_o  += v * math.sin(th_o) * dt_k
        th_o  = wrap_pi(th_o + omega * dt_k)

        # ---- GT relatif frame awal ----
        xw, yw, thw = get_gt_pose2d(sim, base_h)
        dx, dy = xw - x0, yw - y0
        x_gt =  c0*dx + s0*dy
        y_gt = -s0*dx + c0*dy
        th_gt = wrap_pi(thw - th0)

        # ---- Error ----
        ex = x_o - x_gt
        ey = y_o - y_gt
        eth = wrap_pi(th_o - th_gt)

        # ---- Simpan histori ----
        t_hist.append(t_now - t_start)
        x_o_hist.append(x_o);   y_o_hist.append(y_o);   yaw_o_hist.append(math.degrees(th_o))
        x_gt_hist.append(x_gt); y_gt_hist.append(y_gt); yaw_gt_hist.append(math.degrees(th_gt))
        ex_hist.append(ex);     ey_hist.append(ey);     eth_hist.append(math.degrees(eth))

        t_prev = t_now

    return (t_hist,
            x_o_hist, y_o_hist, yaw_o_hist,
            x_gt_hist, y_gt_hist, yaw_gt_hist,
            ex_hist,  ey_hist,  eth_hist)


# ====================================================
# ============ UTILITAS TRANSFORMASI HOMOGEN =========
# ====================================================

def Rz_deg(gamma_deg: float) -> np.ndarray:
    """Rotasi yaw (Z) dalam derajat sebagai matriks homogen 4×4."""
    g = math.radians(gamma_deg)
    cg, sg = math.cos(g), math.sin(g)
    return np.array([
        [cg, -sg, 0, 0],
        [sg,  cg, 0, 0],
        [ 0,   0, 1, 0],
        [ 0,   0, 0, 1],
    ], dtype=float)

def from_h(p4: np.ndarray) -> np.ndarray:
    """(x,y,z,w) → (x,y,z) dengan pembagian terhadap w (jika w!=0)."""
    w = p4[3] if p4[3] != 0 else 1.0
    return p4[:3] / w


# ====================================================
# ============ EKSPERIMEN TRANSFORMASI LINTASAN ======
# ====================================================

def experiment1_rotation(xs, ys):
    """Rotasi ROT_DEG derajat sekitar z-axis."""
    R = Rz_deg(ROT_DEG)
    pts = []
    for x, y in zip(xs, ys):
        p = np.array([x, y, 0.0, 1.0])   # homogen
        pts.append(from_h(R @ p))        # rotasi
    P = np.array(pts)                    # (N,3)
    return P[:, 0], P[:, 1]              # x', y'

def experiment2_translation_after_rotation(xs, ys):
    """Rotasi → Translasi (pakai ROT_DEG, lalu +TX, +TY)."""
    x_rot, y_rot = experiment1_rotation(xs, ys)
    return x_rot + TX, y_rot + TY

def experiment3_translation_before_rotation(xs, ys):
    """Translasi → Rotasi (pakai +TX, +TY lalu ROT_DEG)."""
    R = Rz_deg(ROT_DEG)
    pts = []
    for x, y in zip(xs, ys):
        p = np.array([x + TX, y + TY, 0.0, 1.0])  # translasi dulu
        pts.append(from_h(R @ p))                 # lalu rotasi
    P = np.array(pts)                             # (N,3)
    return P[:, 0], P[:, 1]


# ====================================================
# ======================= MAIN =======================
# ====================================================

def main():
    print("[INFO] connect to CoppeliaSim...")
    print(f"[CFG] ROT_DEG={ROT_DEG} deg | TX={TX} m | TY={TY} m | STEPPING={STEPPING}")

    client = RemoteAPIClient()
    sim = client.require('sim')
    sim.setStepping(STEPPING)
    sim.startSimulation()
    try:
        R, L_half, dt_scene, rJ, lJ, base_h = read_params_from_scene(sim)
        print(f"[PARAM] R={R:.4f} m | L(half)={L_half:.4f} m | dt(scene)={dt_scene:.4f} s")

        (t,
         x_o, y_o, yaw_o,
         x_gt, y_gt, yaw_gt,
         ex, ey, eth_deg) = run_until_stop(sim, rJ, lJ, base_h, R, L_half,
                                           log_to_sim=True, stepping=STEPPING)
    finally:
        sim.stopSimulation()

    # ---- Terapkan tiga eksperimen pada lintasan ODO ----
    x_e1, y_e1 = experiment1_rotation(x_o, y_o)
    x_e2, y_e2 = experiment2_translation_after_rotation(x_o, y_o)
    x_e3, y_e3 = experiment3_translation_before_rotation(x_o, y_o)

    # ====================================================
    # =============== JENDELA 1: ODO vs GT ===============
    # ====================================================
    fig1, axs1 = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # x(t)
    ax = axs1[0, 0]
    ax.plot(t, x_o, label="x_odo")
    ax.plot(t, x_gt, label="x_gt")
    ax.set_xlabel("t [s]"); ax.set_ylabel("x [m]")
    ax.set_title("x(t) — ODO vs GT")
    ax.grid(True, alpha=0.3); ax.legend()

    # y(t)
    ax = axs1[0, 1]
    ax.plot(t, y_o, label="y_odo")
    ax.plot(t, y_gt, label="y_gt")
    ax.set_xlabel("t [s]"); ax.set_ylabel("y [m]")
    ax.set_title("y(t) — ODO vs GT")
    ax.grid(True, alpha=0.3); ax.legend()

    # yaw(t) [deg]
    ax = axs1[1, 0]
    ax.plot(t, yaw_o, label="yaw_odo [deg]")
    ax.plot(t, yaw_gt, label="yaw_gt [deg]")
    ax.set_xlabel("t [s]"); ax.set_ylabel("yaw [deg]")
    ax.set_title("yaw(t) — ODO vs GT")
    ax.grid(True, alpha=0.3); ax.legend()

    # lintasan x vs y
    ax = axs1[1, 1]
    ax.plot(x_o,  y_o,  label="ODO")
    ax.plot(x_gt, y_gt, label="GT")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Trajectory — ODO vs GT")
    ax.grid(True, alpha=0.3); ax.legend()
    ax.set_aspect('equal', 'box')

    # ====================================================
    # =============== JENDELA 2: ERROR ==================
    # ====================================================
    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    ax = axs2[0]
    ax.plot(t, ex)
    ax.set_ylabel("e_x [m]")
    ax.set_title("Error x(t)")
    ax.grid(True, alpha=0.3)

    ax = axs2[1]
    ax.plot(t, ey)
    ax.set_ylabel("e_y [m]")
    ax.set_title("Error y(t)")
    ax.grid(True, alpha=0.3)

    ax = axs2[2]
    ax.plot(t, eth_deg)
    ax.set_xlabel("t [s]"); ax.set_ylabel("e_yaw [deg]")
    ax.set_title("Error yaw(t)")
    ax.grid(True, alpha=0.3)

    # ====================================================
    # =========== JENDELA 3: XP LINTASAN (XY) ============
    # ====================================================
    # Tiga subplot: E1 (Rotasi), E2 (Rotasi→Translasi), E3 (Translasi→Rotasi)
    fig3, axs3 = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    def _setup_xy(ax):
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.3); ax.set_aspect('equal', 'box'); ax.legend()

    # E1: Rotasi
    ax = axs3[0]
    ax.plot(x_o, y_o, linestyle='--', linewidth=1.0, label="ODO (asli)")
    ax.plot(x_e1, y_e1, label=f"E1: Rotasi {ROT_DEG:.0f}°")
    ax.set_title(f"E1: Rotasi {ROT_DEG:.0f}° vs ODO")
    _setup_xy(ax)

    # E2: Rotasi → Translasi
    ax = axs3[1]
    ax.plot(x_o, y_o, linestyle='--', linewidth=1.0, label="ODO (asli)")
    ax.plot(x_e2, y_e2, label=f"E2: Rotasi → Translasi (+{TX}, +{TY})")
    ax.set_title(f"E2: Rot → Trans (+{TX}, +{TY}) vs ODO")
    _setup_xy(ax)

    # E3: Translasi → Rotasi
    ax = axs3[2]
    ax.plot(x_o, y_o, linestyle='--', linewidth=1.0, label="ODO (asli)")
    ax.plot(x_e3, y_e3, label=f"E3: Trans (+{TX}, +{TY}) → Rot")
    ax.set_title(f"E3: Trans (+{TX}, +{TY}) → Rot vs ODO")
    _setup_xy(ax)


    # ====================================================
    # ========== JENDELA 4: OVERLAY SEMUA TRAJ ===========
    # ====================================================
    fig4, ax4 = plt.subplots(1, 1, figsize=(8.5, 7), constrained_layout=True)
    ax4.plot(x_o,  y_o,  label="ODO")
    ax4.plot(x_gt, y_gt, label="GT")
    ax4.plot(x_e1, y_e1, label=f"E1: Rot {ROT_DEG:.0f}°")
    ax4.plot(x_e2, y_e2, label=f"E2: Rot→Trans (+{TX},{TY})")
    ax4.plot(x_e3, y_e3, label=f"E3: Trans(+{TX},{TY})→Rot")
    ax4.set_xlabel("x [m]"); ax4.set_ylabel("y [m]")
    ax4.set_title("Semua Lintasan (Overlay)")
    ax4.grid(True, alpha=0.3); ax4.legend()
    ax4.set_aspect('equal', 'box')

    plt.show()


if __name__ == "__main__":
    main()
