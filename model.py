from pathlib import Path
from collections import namedtuple
import math
import numpy as np
from numba import njit

COIL_TURNS = 600.0  # Coil turns
OIL_DENSITY = 850.0  # Oil density (kg/m^3)
OIL_VISCOSITY = 0.0391  # Dynamic viscosity (Pa s)
PIPE_DIAMETER = 0.012  # Equivalent line diameter (m)
SPOOL_LIMIT = 0.002
SPOOL_STOP_K = 2e6
SPOOL_STOP_B = 100.0
CD0, CD_MIN = 0.65, 0.30
RE_REFERENCE = 2000.0
DP_REFERENCE = 6.89e6
LVDT_M_PER_V = 5.480299070432863e-05  # LVDT calibration (m/V)
DP_REGULARIZATION_PA = 1e4
FEM_I_MIN, FEM_I_MAX = 0.8, 2.85

DEFAULTS = {
    'flow_exponent': 0.7034469699031658,
    'coil_R_ohm': 3.0,  # Coil resistance (ohm)
    'area_limit_cap_m2': 1.1302013641422352e-05,
    'area_limit_rod_m2': 1.1302013641422352e-05,
    'i_bias': 1.5515044526819255,
    'spool_B_Ns_m': 2.462023670527653,
    'spool_Fc_N': 0.0011661727212274832,
    'spool_Stribeck_N': 0.006092490160811126,
    'spool_vS_m_s': 0.01047903698258034,
    'spool_vSmooth_m_s': 0.0024274649141581895,
    'cylinder_B_Ns_m': 8956.789630486723,
    'cylinder_B_asym_Ns_m': 6683.119852293386,
    'cylinder_Fc_asym_N': 702.0451457522181,
    'cylinder_Fc_N': 746.3561668599426,
    'cylinder_Stribeck_N': 97.48344565740167,
    'cylinder_vS_m_s': 0.007287687501203132,
    'cylinder_vSmooth_m_s': 0.008513307538937904,
    'leak_C_m3_s_Pa': 8.486407255728318e-13,
    'width_SA_m': 0.013864406895979975,
    'width_AT_m': 0.011586181648852157,
    'width_SB_m': 0.006916680015159871,
    'width_BT_m': 0.01131384421317631,
    'overlap_A_m': 7.3806816561279925e-06,
    'overlap_B_m': 6.453970386329047e-06,
    'Rf_cap': 0.5680988774333183,
    'Rf_rod': 1.0757229472014265,
    'bulk_Pa': 1238405649.7235205,
    'dead_cap_m3': 0.0016243732364581679,
    'dead_rod_m3': 2.928164757039289e-05,
    'Cd_opening_factor': -0.39672181998890604,
    'Cd_Re_factor': 0.1691130861713004,
    'hydraulic_force_c1_N': 0.037753272141171655,
    'hydraulic_force_c2_N': 0.0567835557413933,
    'hydraulic_force_c3_N': -0.025272067474705066,
    'supply_Pa': 6890000.0,
    'return_Pa': 0.0,
    'Ap_m2': 0.0031172453105244723,  # Cap-end piston area (m^2)
    'Ar_m2': 0.002155132560362598,  # Rod-end piston area (m^2)
    'stroke_m': 0.8,  # Actuator stroke (m)
    'cylinder_mass_kg': 12.0,  # Actuator moving mass (kg)
    'spool_mass_kg': 0.15,  # Spool moving mass (kg)
    'spring_K_N_m': 2000.0,  # Spool spring stiffness (N/m)
    'spring_preload_N': 5.0,  # Spring preload (N)
    'stop_K_N_m': 149999999.98556015,
    'stop_B_Ns_m': 99457.2776908534,
    'Lf_min_s': 1e-05,
    'Lf_reference_cap_s': 1.00e-2,  # B-chamber time scale (s)
    'Lf_reference_rod_s': 1.00e-4,  # A-chamber time scale (s)
    'fem_position_offset_m': 0.0002124158477853974,
    'valve_opening_limit_m': 0.0004917725257650272,
}
Parameters = namedtuple('Parameters', DEFAULTS.keys())
P = Parameters(**DEFAULTS)

THETA_L = np.array([22.86525865700471, -3.834245428891024e-06, -4.584538780713605e-06, 2702.3156544552044, 174.21440663300186, -485.8062032117837, -16.44430260976157, -11.160558265464257, 1.0540289439092709e-06, 5.852925341626475e-07, 633.8106980033684, 481.8463304770338, -1897.994277252738, -3.141184501576184],dtype=float)

@njit
def fem(i, x):

    a = i*(i*(i*(i*(i*(i*(i*(i*(1.3392898107410826e-6 - 1.1407362591778521e-7*i) - 6.3197417885318624e-6) + 1.5333032379103403e-5) - 2.1250838607854709e-5) + 2.1665948474054305e-5) - 2.4164716743137437e-5) + 1.555876326453812e-5) + 2.8461096877950534e-5) - 2.9998780693164051e-8
    b = i*(i*(i*(i*(i*(i*(i*(1.6385678177136009e-7*i - 1.8576796459061211e-6) + 7.2481227473776773e-6) - 9.7677103151136904e-6) - 3.3355929755768183e-6) + 1.2063311963558343e-5) + 8.4737740787764415e-6) - 1.8640143766827901e-5) - 7.607471861159975e-8
    c = i*(i*(i**2*(i**2*(0.0003230458671809637*i - 0.0016217545849003354) + 0.0081905528699418358) - 0.030755197303321501) + 0.029044819993620741) + 0.00016122604269944499
    phi = a+x*b+x*x*c
    ai = i*(i*(i*(i*(i*(i*(i*(1.0714318485928661e-5 - 1.0266626332600669e-6*i) - 4.4238192519723037e-5) + 9.1998194274620421e-5) - 0.00010625419303927355) + 8.6663793896217219e-5) - 7.2494150229412312e-5) + 3.111752652907624e-5) + 2.8461096877950534e-5
    bi = i*(i*(i*(i*(i*(i*(1.3108542541708807e-6*i - 1.3003757521342848e-5) + 4.3488736484266064e-5) - 4.8838551575568452e-5) - 1.3342371902307273e-5) + 3.618993589067503e-5) + 1.6947548157552883e-5) - 1.8640143766827901e-5
    ci = i*(i**2*(i**2*(0.0022613210702667459*i - 0.0097305275094020124) + 0.032762211479767343) - 0.061510394606643003) + 0.029044819993620741
    force = -0.064088422083926302*i**7 + 19017.17480573773*i**6*phi + 0.19485176182654562*i**6 - 324014755.48996437*i**5*phi**2 - 117271.44101814588*i**5*phi - 1.2360882925978292*i**5 + 1885121511.4108825*i**4*phi**2 + 217618.12022726136*i**4*phi + 18.243200797442839*i**4 - 708815.45315050043*i**3*phi - 68.743200217826342*i**3 - 11484462325.422945*i**2*phi**2 + 3443220.1150077479*i**2*phi + 71.658673251473118*i**2 - 5840592.1601466369*i*phi + 47.937721465199644*i + 47575538320.823135*phi**2 + 162329.37022281275*phi - 13.388662529994097
    return force, COIL_TURNS*(ai+x*bi+x*x*ci), -COIL_TURNS*(b+2*x*c)

@njit
def softplus(a):return max(a,0.)+math.log1p(math.exp(-abs(a)))

@njit
def flow_coefficients(z,p,theta):

    features=np.array([z[3],z[4],z[1],z[7],z[2],z[8]])
    cap=theta[0];rod=theta[7]
    for j in range(6):
        cap+=theta[j+1]*features[j]
        rod+=theta[j+8]*features[j]
    return p.Lf_min_s+p.Lf_reference_cap_s*softplus(cap),p.Lf_min_s+p.Lf_reference_rod_s*softplus(rod)

@njit
def pressure_root(dp,nq):

    r=dp/DP_REFERENCE;e=DP_REGULARIZATION_PA/DP_REFERENCE
    return math.sqrt(2*DP_REFERENCE/OIL_DENSITY)*r*(r*r+e*e)**(nq/4-.5)

@njit
def hydraulic_force_map(aa,at,ba,bt,pc,pr,xs,p):
    d=(pr-pc)/DP_REFERENCE
    s=(pr+pc)/DP_REFERENCE
    return p.hydraulic_force_c1_N*d+p.hydraulic_force_c2_N*s+p.hydraulic_force_c3_N*d*s

@njit
def metering_area(width, opening, area_max, opening_max):

    return min(area_max,width*min(opening_max,max(opening,0.)))

@njit
def metering_areas(xs,p):
    x=xs
    return (metering_area(p.width_SA_m,x-p.overlap_A_m,p.area_limit_rod_m2,p.valve_opening_limit_m),
            metering_area(p.width_AT_m,-x-p.overlap_A_m,p.area_limit_rod_m2,p.valve_opening_limit_m),
            metering_area(p.width_SB_m,-x-p.overlap_B_m,p.area_limit_cap_m2,p.valve_opening_limit_m),
            metering_area(p.width_BT_m,x-p.overlap_B_m,p.area_limit_cap_m2,p.valve_opening_limit_m))

@njit
def rates(z,terminal_voltage,p,theta):
    i,xs,vs,pc,pr,qc,qr,x,v=z
    fem_coordinate=p.fem_position_offset_m-xs
    bounded_coordinate=min(.0004,max(0.,fem_coordinate))
    force,li,lx=fem(min(FEM_I_MAX,max(FEM_I_MIN,i)),bounded_coordinate)
    if fem_coordinate<=0. or fem_coordinate>=.0004:lx=0.
    if li<=0:return np.full(9,np.nan)
    di=(terminal_voltage-p.coil_R_ohm*i-lx*vs)/li
    aa,at,ba,bt=metering_areas(xs,p)

    re_cap=OIL_DENSITY*abs(qc)*PIPE_DIAMETER/(OIL_VISCOSITY*math.pi*PIPE_DIAMETER**2/4)
    re_rod=OIL_DENSITY*abs(qr)*PIPE_DIAMETER/(OIL_VISCOSITY*math.pi*PIPE_DIAMETER**2/4)
    cd_cap=CD0*(1+p.Cd_opening_factor*min(abs(xs)/(p.valve_opening_limit_m),1.)+p.Cd_Re_factor*re_cap/(RE_REFERENCE+re_cap))
    cd_rod=CD0*(1+p.Cd_opening_factor*min(abs(xs)/(p.valve_opening_limit_m),1.)+p.Cd_Re_factor*re_rod/(RE_REFERENCE+re_rod))
    cd_cap=min(CD0,max(CD_MIN,cd_cap));cd_rod=min(CD0,max(CD_MIN,cd_rod))
    fqcap=cd_cap*(ba*pressure_root(p.supply_Pa-pc,p.flow_exponent)-bt*pressure_root(pc-p.return_Pa,p.flow_exponent))
    fqrod=cd_rod*(aa*pressure_root(p.supply_Pa-pr,p.flow_exponent)-at*pressure_root(pr-p.return_Pa,p.flow_exponent))
    lf_c,lf_r=flow_coefficients(z,p,theta)
    dqc=(fqcap-p.Rf_cap*qc)/lf_c;dqr=(fqrod-p.Rf_rod*qr)/lf_r
    dp=pc-pr
    leak=p.leak_C_m3_s_Pa*dp
    va=max(.25*p.dead_cap_m3,p.dead_cap_m3+p.Ap_m2*x)
    vb=max(.25*p.dead_rod_m3,p.dead_rod_m3+p.Ar_m2*(p.stroke_m-x))
    dpc=p.bulk_Pa/va*(qc-p.Ap_m2*v-leak)
    dpr=p.bulk_Pa/vb*(qr+p.Ar_m2*v+leak)
    if pc<=0 and dpc<0:dpc=0.
    if pr<=0 and dpr<0:dpr=0.
    flowforce=hydraulic_force_map(aa,at,ba,bt,pc,pr,xs,p)
    fs=(p.spool_Fc_N+p.spool_Stribeck_N*math.exp(-(vs/p.spool_vS_m_s)**2))*math.tanh(vs/p.spool_vSmooth_m_s)+p.spool_B_Ns_m*vs
    stop_s=-SPOOL_STOP_K*math.copysign(max(abs(xs)-SPOOL_LIMIT,0.),xs)
    if abs(xs)>SPOOL_LIMIT:stop_s-=SPOOL_STOP_B*vs
    dvs=(force-p.spring_preload_N-p.spring_K_N_m*xs-fs-flowforce+stop_s)/p.spool_mass_kg
    fc=(p.cylinder_Fc_N+p.cylinder_Stribeck_N*math.exp(-(v/p.cylinder_vS_m_s)**2))*math.tanh(v/p.cylinder_vSmooth_m_s)+p.cylinder_B_Ns_m*v
    stop=0.
    if x<0:stop=-p.stop_K_N_m*x-p.stop_B_Ns_m*min(v,0.)
    elif x>p.stroke_m:stop=-p.stop_K_N_m*(x-p.stroke_m)-p.stop_B_Ns_m*max(v,0.)
    tv=math.tanh(v/p.cylinder_vSmooth_m_s)
    fc+=p.cylinder_B_asym_Ns_m*v*tv+p.cylinder_Fc_asym_N*tv*tv
    dv=(p.Ap_m2*pc-p.Ar_m2*pr-fc+stop)/p.cylinder_mass_kg
    return np.array([di,vs,dvs,dpc,dpr,dqc,dqr,v,dv])

@njit(nogil=True)
def _integrate(voltage,p,theta,dt,internal_step):
    z=np.zeros(9);z[0]=p.i_bias;z[4]=p.supply_Pa
    state=np.empty((len(voltage)+1,9));state[0]=z;diag=np.zeros(4)
    for k in range(len(voltage)):
        remaining=dt
        while remaining>1e-13:
            la,lb=flow_coefficients(z,p,theta)
            h=min(internal_step,.35*min(la/p.Rf_cap,lb/p.Rf_rod),remaining)
            accepted=False
            for _ in range(12):
                aa=rates(z,voltage[k],p,theta)
                z2=z+.5*h*aa
                lc,lr=flow_coefficients(z2,p,theta)
                if h>.5*min(lc/p.Rf_cap,lr/p.Rf_rod):h*=.5;continue
                bb=rates(z2,voltage[k],p,theta)
                z3=z+.5*h*bb
                lc,lr=flow_coefficients(z3,p,theta)
                if h>.5*min(lc/p.Rf_cap,lr/p.Rf_rod):h*=.5;continue
                cc=rates(z3,voltage[k],p,theta)
                z4=z+h*cc
                lc,lr=flow_coefficients(z4,p,theta)
                if h>.5*min(lc/p.Rf_cap,lr/p.Rf_rod):h*=.5;continue
                dd=rates(z4,voltage[k],p,theta)
                zn=z+h*(aa+2*bb+2*cc+dd)/6
                if np.isfinite(zn).all():accepted=True;break
                h*=.5
            if not accepted or h<1e-11 or diag[3]>2e8:
                state[k+1:]=np.nan;return state,diag
            z=zn;remaining-=h
            for j in (3,4):
                if z[j]<0:z[j]=0.;diag[0]+=1
            if z[0]<.8 or z[0]>2.85:diag[1]+=1
            if abs(z[1])>SPOOL_LIMIT:diag[2]+=1
            diag[3]+=1
            if abs(z[1])>.01 or abs(z[7])>2 or max(z[3],z[4])>2e8:
                state[k+1:]=np.nan;return state,diag
        state[k+1]=z
    return state,diag


def simulate(voltage):
    voltage = np.asarray(voltage, dtype=float)
    if voltage.ndim != 1 or len(voltage) < 2 or not np.isfinite(voltage).all():
        raise ValueError('Voltage must be a finite one-dimensional array.')
    state, _ = _integrate(voltage, P, THETA_L, 0.001, 0.00005)
    if not np.isfinite(state).all():
        raise RuntimeError('Integration failed. Check the voltage input.')
    state = state[:-1]
    return np.column_stack((np.arange(len(voltage))*0.001,
                            state[:,0], state[:,1], state[:,4],
                            state[:,3], state[:,7]))


def plot_outputs(data, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif',
                         'font.serif':['Times New Roman','DejaVu Serif'],
                         'font.size':9})
    t = data[:,0]
    series = (data[:,1], data[:,2]*1000, data[:,3]/1e5,
              data[:,4]/1e5, data[:,5]*100)
    labels = (r'$i$ (A)', r'$x_s$ (mm)', r'$P_A$ (bar)',
              r'$P_B$ (bar)', r'$x_c$ (cm)')
    fig, axes = plt.subplots(5, 2, figsize=(7.2,7.5), sharex='col',
                             constrained_layout=True)
    for row, (y, label) in enumerate(zip(series, labels)):
        for col in range(2):
            ax = axes[row,col]
            ax.plot(t, y, color='#245fa8', linewidth=0.9)
            ax.set_xlim((t[0],t[-1]) if col == 0 else
                        (max(t[0], t[-1]-5),t[-1]))
            ax.tick_params(direction='in')
            if col == 0:
                ax.set_ylabel(label)
    axes[0,0].set_title('Full simulation')
    axes[0,1].set_title('Final 5 s')
    for ax in axes[-1]:
        ax.set_xlabel('Time (s)')
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent
    data = np.genfromtxt(root/'input_voltage.csv', delimiter=',',
                         names=True, encoding='utf-8-sig')
    if data.dtype.names != ('time_s','voltage_V') or data.ndim != 1 or len(data) < 2:
        raise ValueError('Expected time_s,voltage_V columns and at least two rows.')
    expected_time = np.arange(len(data))*0.001
    if not np.allclose(data['time_s'], expected_time, rtol=0, atol=1e-10):
        raise ValueError('Time must start at zero and have a 0.001 s interval.')
    result = simulate(data['voltage_V'])
    plot_outputs(result, root/'five_outputs.png')
    print('Saved five_outputs.png (1000 Hz).')


if __name__ == '__main__':
    main()
