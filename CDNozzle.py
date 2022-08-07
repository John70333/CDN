import math as m
import sympy as s
import numpy as n
import matplotlib.pyplot as plt
from numpy import array as a
from scipy.optimize import fsolve as fs, brentq as bq
from time import process_time as time
t0 = time()
x = s.Symbol('x', real='True', positive='True')


def c_gc(ax, bx, cx, kx):
    def c_dc(pt, qt, kt):
        if pt == 0:
            return 0
        elif 4*pt**3+27*qt**2 < 0:
            return 2*(-pt/3)**.5*m.cos(m.acos(1.5*qt/pt*(-3/pt)**.5)/3-2*kt*m.pi/3)
        elif 4*pt**3+27*qt**2 == 0:
            return 3*qt/pt if kt == 0 else -1.5*qt/pt
        else:
            return (-.5*qt+((qt/2)**2+(pt/3)**3)**.5)**(1/3)+(-.5*qt-((qt/2)**2+(pt/3)**3)**.5)**(1/3)
    return c_dc(bx-ax**2/3, 2*(ax/3)**3-ax*bx/3+cx, kx)-ax/3


tf = lambda ti, mi, mf: ti*(1+.5*(g-1)*mi**2)/(1+.5*(g-1)*mf**2)
mf_t = lambda mi, tmi, tmf: ((2/(g-1))*((tf(1, mi, 0)*tmi/tmf)-1))**.5
pf = lambda pi, mi, mf: pi*tf(1, mi, mf)**(g/(g-1))
mf_p = lambda mi, pmi, pmf: mf_t(mi, pmi**((g-1)/g), pmf**((g-1)/g))
af = lambda ai, mi, mf: ai*(mi/mf)*tf(1, mi, mf)**(-.5*(g+1)/(g-1))
mfr_p0 = lambda ap, mp, p0p, t0p: ap*mp*p0p*(g/(r*t0p))**.5*tf(1, mp, 0)**(-.5*(g+1)/(g-1))
mf_ns = lambda mi: ((1+.5*(g-1)*mi**2)/(g*mi**2-.5*(g-1)))**.5
pr_ns = lambda mi: 1+(2*g/(g+1))*(mi**2-1)
p0r_ns = lambda mi: pf(pr_ns(mi), mf_ns(mi), mi)
mi_pr_ns = lambda pr: (1+.5*(g+1)/g*(pr-1))**.5
sab_os = lambda mi: m.asin((.25*(g+1)/g-1/(g*mi**2) +
                           ((.25*(g+1)/g-1/(g*mi**2))**2+((g+1)*mi**2+2)/(2*g*mi**4))**.5)**.5)
da_sa_os = lambda mi, sa: m.atan((2/m.tan(sa))*((mi*m.sin(sa))**2-1)/(mi**2*(g+m.cos(2*sa))+2))
mf_sa_os = lambda mi, sa: mf_ns(mi*m.sin(sa))/m.sin(sa-da_sa_os(mi, sa))
v_ef = lambda mi: ((g+1)/(g-1))**.5*m.atan(((mi**2-1)*(g-1)/(g+1))**.5)-m.atan((mi**2-1)**.5)


def cdn(p):
    ar1t, ar2t, prb0, lrxd = p
    ss = lambda fn, vl, vr, k: a(s.solve(s.Eq(fn, vl), vr))[k]
    faxp = lambda k, xk1, xk2: n.where((xk1 <= k) & (k <= xk2), 1+(ar2t-1)*(3-2*k)*k**2, 100*max(ar1t, ar2t))
    fxpa = lambda k: c_gc(-1.5, 0, .5*(abs(k)-1)/(ar2t-1), 1 if k >= 0 else (0 if abs(k) == ar2t else 2))
    m2_sb, m2_sp = ss(af(1, 1, x), ar2t, x, 0), ss(af(1, 1, x), ar2t, x, 1)
    prb0_cr1, prb0_cr3 = pf(1, 0, m2_sb), pf(1, 0, m2_sp)
    prb0_cr2, prb0_crmxd = prb0_cr3*pr_ns(m2_sp), prb0_cr3*pr_ns(m2_sp*m.sin(sab_os(m2_sp)))
    prb0_crsnc = prb0_cr3*fs(lambda k: mf_sa_os(m2_sp,  m.asin(mi_pr_ns(k)/m2_sp))-1, a(1))[0]
    lr1d, lr2d, mfrc = fxpa(-ar1t), fxpa(ar2t), n.zeros(len(prb0))
    arxt, prx0, mx = faxp(lrxd, lr1d, lr2d), n.zeros((len(prb0), len(lrxd))), n.zeros((len(prb0), len(lrxd)))
    txt = "\nInput:\nar1t = "+str(ar1t)+"\nar2t = "+str(ar2t) +\
          "\n\nMach nos. at exit for undisturbed flow:\nm2_sb = "+str(m2_sb)+"\nm2_sp = "+str(m2_sp) +\
          "\n\nCritical back pressure ratios:\nprb0_cr1 = "+str(prb0_cr1)+"\nprb0_cr2 = "+str(prb0_cr2) +\
          "\nprb0_crmxd = "+str(prb0_crmxd)+"\nprb0_crsnc = "+str(prb0_crsnc)+"\nprb0_cr3 = "+str(prb0_cr3)+"\n"
    print("Time taken until prb0 loop = ", time()-t0)
    for i in range(len(prb0)):
        if 1 < ar1t and 1 < ar2t and 0 < prb0[i] < 1:
            mt = 1.0 if prb0[i] <= prb0_cr1 else ss(af(ar2t, mf_p(0, 1, prb0[i]), x), 1, x, 0)
            m1_sb, mfrc[i], m2, arwt = ss(af(1, mt, x), ar1t, x, 0), mfr_p0(1, mt, 1, 1), 0, 0
            if prb0_cr1 <= prb0[i]:
                m2, arwt = mf_p(0, 1, prb0[i]), 1
                txt += "\nfor prb0 = "+str(prb0[i])+"\n"
            elif prb0_cr2 <= prb0[i] < prb0_cr1:
                mxy_ns = lambda k: af(af(1, 1, k), mf_ns(k), mf_p(0, p0r_ns(k), prb0[i]))-ar2t
                mxy2g = abs(fs(lambda k: pf(pr_ns(k), mf_ns(k), k)-prb0[i], a(1))[0])
                trnc = lambda num, prcn: m.floor(num*10**prcn)/10**prcn
                ms1_ns = bq(mxy_ns, 1, trnc(mxy2g, 12))
                ms2_ns, m2, arwt = mf_ns(ms1_ns), mf_p(0, p0r_ns(ms1_ns), prb0[i]), af(1, 1, ms1_ns)
                txt += "\nfor prb0 = "+str(prb0[i])+"\nms1_ns = "+str(ms1_ns)+"\nms2_ns = "+str(ms2_ns)+"\n"
            elif prb0_cr3 <= prb0[i] < prb0_cr2:
                sa_os = m.asin(mi_pr_ns(prb0[i]/prb0_cr3)/m2_sp)
                da_os, m2, arwt = da_sa_os(m2_sp, sa_os), mf_sa_os(m2_sp, sa_os), ar2t
                txt += "\nfor prb0 = "+str(prb0[i])+"\nsa_os = "+str(sa_os)+"\nda_os = "+str(da_os)+"\n"
            elif prb0[i] < prb0_cr3:
                m2, arwt = mf_p(0, 1, prb0[i]), ar2t
                fa_ef = v_ef(m2)-v_ef(m2_sp)
                txt += "\nfor prb0 = "+str(prb0[i])+"\nfa_ef = "+str(fa_ef)+"\n"
            p02 = pf(prb0[i], m2, 0) if prb0_cr3 < prb0[i] < prb0_cr1 else 1
            lrwd, s12 = fxpa(arwt), -r*m.log(p02)
            txt += "\nmt = "+str(mt)+"\nm1 = "+str(m1_sb)+"\nm2 = "+str(m2) +\
                   "\narwt = "+str(arwt)+"\nmfrc = "+str(mfrc[i])+"\ns12 = "+str(s12)+"\n"
            for j in range(len(lrxd)):
                if lrxd[j] < lr1d:
                    mx[i][j], prx0[i][j] = 0, 1
                elif lr1d <= lrxd[j] <= 0:
                    mx[i][j] = ss(af(1, mt, x), arxt[j], x, 0)
                    prx0[i][j] = pf(1, 0, mx[i][j])
                elif 0 < lrxd[j] <= lrwd:
                    mx[i][j] = ss(af(1, mt, x), arxt[j], x, 1)
                    prx0[i][j] = pf(1, 0, mx[i][j])
                elif lrwd < lrxd[j] <= lr2d:
                    mx[i][j] = ss(af(ar2t, m2, x), arxt[j], x, 0 if m2 <= 1 else 1)
                    prx0[i][j] = pf(p02, 0, mx[i][j])
                elif lr2d < lrxd[j]:
                    mx[i][j], prx0[i][j] = mf_p(0, p02, prb0[i]), prb0[i]
        print("Time until after prb0 [", i, "] = ", time()-t0)
    return arxt, mx, prx0, txt, mfrc, a([prb0_cr1, prb0_cr2, prb0_crmxd, prb0_crsnc, prb0_cr3])


# pbcr_ref = [5957.39251652314, 47250.3239045630, 48506.0024250006, 59089.9613254698, 197022.137500427]
g, r, a1s, ats, a2s, lns, p0s, t0s = 1.4, 287.057, 4, 1, 4, 20, 200000, 300
pbs = a([198000, 197022.137500427, 101325, 59089.9613254698, 47250.3239045630, 5957.39251652314, 3000])
xs = n.arange(-11, 23, 1)
axr, mxr, pxr, txtr, mfrr, pbcrr = cdn((a1s/ats, a2s/ats, pbs/p0s, xs/lns))
print("Time taken to execute cdn() = ", time()-t0)
txr = a([[tf(1, 0, emxr) for emxr in mxr[rmxr]] for rmxr in n.arange(len(mxr))])
if n.all(pxr == 0):
    print("Error! Check input values for conditions: 1 < ar1t, 1 < ar2t, 0 < prb0[i] < 1")
else:
    print("\n\n", txtr, "\nPosition values:\n", xs, "\n\nMach no. values:\n", mxr, "\n\nPressure values:\n", pxr,
          "\n\nMass flow rates:\n", mfrr*p0s*ats/t0s**0.5, "\n\nCritical back pressures:\n", pbcrr*p0s, "\n")
    fig, axis = plt.subplots(2, 2, figsize=(18, 10))
    axrp, mxrp, pxrp, txrp = axis[0, 0], axis[0, 1], axis[1, 0], axis[1, 1]
    axrp.plot(xs, 0.5*ats*axr, 'k-', xs, -0.5*ats*axr, 'k-')
    axrp.set_title('Convergent - Divergent Nozzle', fontsize='10')
    axrp.set_ylim([-0.6*max(a1s, a2s), 0.6*max(a1s, a2s)])
    mxrp.plot(xs, mxr.transpose(), '-')
    mxrp.set_title('Mach Number Distribution', fontsize='10')
    mxrp.set_ylim([-0.1*n.amax(mxr), 1.1*n.amax(mxr)])
    pxrp.plot(xs, p0s*pxr.transpose(), '-')
    pxrp.set_title('Pressure Distribution', fontsize='10')
    pxrp.set_ylim([(n.amin(pxr)-0.1)*p0s, 1.1*p0s])
    txrp.plot(xs, t0s*txr.transpose(), '-')
    txrp.set_title('Temperature Distribution', fontsize='10')
    txrp.set_ylim([(n.amin(txr)-0.1)*t0s, 1.1*t0s])
    for axs in axis.flat:
        axs.set_xlim([xs[0]-0.1*lns, xs[-1]+0.1*lns])
        axs.grid()
    for axs in axis.flat[1:]:
        axs.legend(pbs, fontsize='7')
    plt.tight_layout()
    plt.savefig('CDN.png')
    print("Total time taken = ", time()-t0)
    plt.show()
