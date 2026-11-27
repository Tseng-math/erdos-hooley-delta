"""Exact arithmetic for the sixteen-type Hooley-delta lower bound.

Run with Python 3.9+ without -O. Only the standard library is needed.
The sixteen output rows correspond to node types 0 through 15.
"""
from fractions import Fraction as F
from functools import lru_cache

if not __debug__:
    raise RuntimeError("Run with assertions enabled to check the certificate.")

one = F(1)
zero = F(0)
cutoff = F(".00025")
left = (F("0.17008802076989"), F("0.17445386687652"), F("0.16936398844577"), F("0.17388142632538"), F("0.17009530835310"), F("0.17438951025087"), F("0.16945123639388"), F("0.17388691615311"), F("0.17008831268608"), F("0.17445384071325"), F("0.16936298668175"), F("0.17388152290720"), F("0.17009683577282"), F("0.17438937491771"), F("0.16945134069206"), F("0.17388690617701"))
ratio = (F("0.68401572231443"), F("0.66471379404345"), F("0.68921875368253"), F("0.66892994497063"), F("0.68393223772060"), F("0.66497954537033"), F("0.68877984539362"), F("0.66890195988505"))
right = tuple(v*ratio[i%8] for i,v in enumerate(left))
phase = tuple(ratio[i//2] if i%2==0 else left[i-1]/ratio[i//2] for i in range(16))
q = (
    (F("0.367906599960"), F("0.378382774055"), F("0.378609457829"), F("0.382868703940")),
    (F("0.367906988749"), F("0.370803262020"), F("0.371785782457"), F("0.371946413457")),
    (F("0.367995493671"), F("0.379097176841"), F("0.379119849971"), F("0.383989643006")),
    (F("0.367881403353"), F("0.371076969007"), F("0.372210929174"), F("0.372308612632")),
    (F("0.367879728139"), F("0.378327161781"), F("0.378614497836"), F("0.382778568605")),
    (F("0.368044463559"), F("0.370926681912"), F("0.371881452912"), F("0.372061709237")),
    (F("0.367880216220"), F("0.378897274167"), F("0.378922330993"), F("0.383794414115")),
    (F("0.367881830045"), F("0.371054243031"), F("0.372198860777"), F("0.372294368723")),
)
hshift = (F("0.005753434443"), F("0.004550810468"), F("0.006430082793"), F("0.004990980423"), F("0.005690650612"), F("0.004521571727"), F("0.006384472359"), F("0.004974131778"))
delta = (F("0.003668869499"), F("0.004076944109"), F("0.003631054748"), F("0.004049115897"))

q=q*2
hshift=hshift*2
delta=delta*2

def snap(x, up):
    n=10**22
    return F(-(-x*n//1) if up else x*n//1,n)

@lru_cache(None)
def ln_grid(x, n=40):
    z=(x-1)/(x+1)
    assert abs(z)<1
    u=z; partial=zero
    for j in range(n):
        partial+=2*u/(2*j+1)
        u*=z*z
    err=2*abs(u)/((2*n+1)*(1-z*z))
    return snap(partial-err,False),snap(partial+err,True)

def ln(x,n=40):
    return ln_grid(snap(x,False),n)[0],ln_grid(snap(x,True),n)[1]

def entropy(p):
    return -p*ln(p)[1]-(1-p)*ln((1-p)/2)[1]

@lru_cache(None)
def information(p,r,offset):
    upper=zero
    for i,x in enumerate((p,1-p)):
        for j,y in enumerate((r,1-r)):
            base=x*y
            mass=base+(offset if i!=j else -offset)
            assert 0<mass<1
            upper+=mass*ln(mass/base)[1]
    return upper

def state(i,t):
    return sum(t<s for s in (phase[i],left[i],right[i]))

@lru_cache(None)
def pattern(i,t):
    if t>1:
        return one,zero
    c=2*(i%8)
    x,l=pattern(c,t/left[i])
    y,r=pattern(c+1,t/right[i])
    p=q[c][state(c,t/left[i])]
    s=q[c+1][state(c+1,t/right[i])]
    p1=p-hshift[c]*(1-r)
    s1=s-hshift[c+1]*(1-l)
    z=x*y-l*r*(p*s-p1*s1+delta[i%8])
    assert max(zero,x+y-1)<=z<=min(x,y)
    return q[i][state(i,t)]*z,z

def points(i,height):
    if height<cutoff:
        return set()
    c=2*(i%8)
    return ({height, height*phase[i]} | points(c,height*left[i]) |
            points(c+1,height*right[i]))

def budget(i):
    older=i%2==0
    p=i//2
    c=2*(i%8)
    pts=points(i,one)
    if older:
        pts|=points(i+1,ratio[p])
    else:
        pts|=points(i-1,1/ratio[p])
    pts=sorted({u for u in pts if cutoff<u<=1}|{cutoff})
    total=zero
    for lo,hi in zip(pts,pts[1:]):
        t=(lo+hi)/2
        x,l=pattern(c,t/left[i])
        y,r=pattern(c+1,t/right[i])
        v,z=pattern(i,t)
        prob=q[i][state(i,t)]
        sib=i+1 if older else i-1
        ts=t/ratio[p] if older else t*ratio[p]
        _,z1=pattern(sib,ts)
        p1=prob-hshift[i]*(1-z1)
        p0=prob+hshift[i]*z1
        assert F(".35")<=p1<=p0<=F(".40")
        h=z*((1-z1)*entropy(p0)+z1*entropy(p1))+(x+y-2*z)*ln(F(2))[0]
        if older and t<ratio[p]:
            s1=q[sib][state(sib,ts)]-hshift[sib]*(1-z)
            h-=z*z1*information(p1,s1,delta[p])
        total+=(hi-lo)*h
    return len(pts)-1,(total-1)*10**12//1

def expneg(u):
    """Enclose exp(-u) using the alternating tail from term 34 onward."""
    assert 0<=u<=2
    term=total=one
    for j in range(1,35):
        term*=-u/j
        total+=term
    return total-u*term/35,total

def power(x,d):
    low,high=ln(1/x,120)
    return expneg(d*high)[0],expneg(d*low)[1]

def perron_side(d):
    # Exact Collatz-Wielandt residual evaluated on a fixed positive rational
    # vector shared by the two endpoints. A strictly one-sided residual
    # places the spectral radius on that same side of one.
    D=(F("1.00000000000000"), F("0.99507721338637"), F("1.00122719927539"), F("0.99616212670301"), F("0.99998002103143"), F("0.99514192367964"), F("1.00112266506735"), F("0.99615533827001"))
    lo=[];hi=[]
    for i in range(8):
        a=power(left[2*i],d);b=power(ratio[i]*left[2*i+1],d)
        j=2*i%8
        lo.append(a[0]*D[j]+b[0]*D[j+1]-D[i])
        hi.append(a[1]*D[j]+b[1]*D[j+1]-D[i])
    return min(lo),max(hi)

if __name__=="__main__":
    assert all(1>phase[i]>left[i]>right[i]>cutoff for i in range(16))
    assert min(min(row) for row in q)-max(hshift)>F(".35")
    assert max(max(row) for row in q)+max(hshift)<F(".40")
    assert max(left)<min(ratio) and min(left)*min(ratio)>0
    # MI <= chi^2 = delta^2 / (p(1-p)r(1-r)). G is concave.
    assert min(entropy(F(".35")),entropy(F(".40")))-(max(delta)/(F(".35")*F(".60")))**2>1
    assert (1+max(max(row) for row in q))*ln(F(2))[1]<1
    expected_surpluses = (29, 29, 29, 30, 30, 30, 29, 29)
    for i in range(16):
        cells,surplus=budget(i)
        print(f"type {i}, cells={cells}, floor(10^12 * surplus)={surplus}",flush=True)
        assert cells == (56 if i % 2 == 0 else 60)
        assert surplus == expected_surpluses[i % 8]
    least_lower, _ = perron_side(F(".35553850187"))
    _, greatest_upper = perron_side(F(".35553850189"))
    assert least_lower > F("2.08e-11")
    assert greatest_upper < -F("1.78e-11")
    print("0.35553850187 < d < 0.35553850189")
