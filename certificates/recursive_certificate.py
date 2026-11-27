#!/usr/bin/env python3
"""Verify the recursive Bellman bound for Hooley's divisor concentration exponent.

Python 3.9 or later and NumPy are required. Run without arguments for the
complete certificate. Arguments from 0 to 3 select individual scale intervals.
All coefficients, subdivision bounds and numerical comparisons use integers
or exact rational numbers.
"""
import argparse
from fractions import Fraction as Q
from functools import lru_cache
from itertools import product
from math import comb, lcm

import numpy as np


if not __debug__:
    raise RuntimeError("Run with assertions enabled to check the certificate.")


class Polynomial:
    """A sparse polynomial in (s, x, y) over the rational numbers."""

    def __init__(self, value):
        if isinstance(value, Polynomial):
            self.terms = value.terms
        elif isinstance(value, dict):
            self.terms = value
        else:
            self.terms = {(0, 0, 0): Q(value)} if value else {}

    def __add__(self, other):
        total = self.terms.copy()
        for exponent, coefficient in Polynomial(other).terms.items():
            new_coefficient = total.get(exponent, 0) + coefficient
            if new_coefficient:
                total[exponent] = new_coefficient
            else:
                total.pop(exponent, None)
        return Polynomial(total)

    __radd__ = __add__

    def __neg__(self):
        return Polynomial({power: -coef for power, coef in self.terms.items()})

    def __sub__(self, other):
        return self + (-Polynomial(other))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        total = {}
        for (i, j, k), left in self.terms.items():
            for (a, b, c), right in Polynomial(other).terms.items():
                exponent = (i + a, j + b, k + c)
                total[exponent] = total.get(exponent, 0) + left * right
        return Polynomial({power: coef for power, coef in total.items() if coef})

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        return self * (1 / Q(scalar))

    def __pow__(self, exponent):
        if exponent == 0:
            return Polynomial(1)
        if exponent == 1:
            return self
        half = self ** (exponent // 2)
        square = half * half
        return square * self if exponent % 2 else square


a = Q(355630, 1000000)
epsilon = 5 - 14 * a
p0, p1, p2 = (Q(i, 10000000) for i in (2998182, 1908507, 617448))
s, x, y = (Polynomial({tuple((int(i == axis) for i in range(3))): Q(1)}) for axis in range(3))
S = Q(76, 100) * s
u, v = (x ** 14, (x * y) ** 14)

def log_bounds(n):
    """The atanh series for log(n) with a geometric bound on the tail."""
    t = Q(n - 1, n + 1)
    N = 120
    lower = 2 * sum((t ** (2 * i + 1) / (2 * i + 1) for i in range(N)))
    return (lower, lower + 2 * t ** (2 * N + 1) / ((2 * N + 1) * (1 - t * t)))
lo2, hi2 = (Q(69314718055, 10 ** 11), Q(69314718056, 10 ** 11))
lo3, hi3 = (Q(109861228866, 10 ** 11), Q(109861228867, 10 ** 11))
bound2, bound3, bound7 = (log_bounds(n) for n in (2, 3, 7))
assert lo2 < bound2[0] <= bound2[1] < hi2
assert lo3 < bound3[0] <= bound3[1] < hi3
assert 100 * (hi3 - 1) < 65 * (lo3 - 1 + 2 - bound7[1])
assert Q(65, 100) ** 9 < Q(76, 100) ** 14
assert 0 < epsilon < 1 and Q(76, 100) * hi2 < Q(527, 1000)
assert Q(1, 27) - Q(527, 1000) / 48 > Q(26, 1000)

def potential(z):
    return p0 + p1 * z + p2 * z ** 2

def power_lower(z):
    """Lower polynomial envelope of z to the power 9 + epsilon."""
    return z ** 9 - epsilon * z ** 8 * (1 - z * z) / 2

def branch_upper(z):
    """Upper polynomial envelope of the uncharged child potential."""
    low_power = z ** 5 + epsilon * z ** 4 * (1 - z * z) / 2
    low_power += epsilon ** 2 * z ** 3 * (1 - z) ** 2 / 2
    high_power = z ** 23 * (1 - epsilon * (1 - z) - epsilon * (1 - epsilon) * (1 - z) ** 2 / 2)
    return p0 * low_power + p1 * z ** 14 + p2 * high_power

def q_lower(z):
    return Q(1, 3) + 2 * lo2 * power_lower(z) / 9 + Q(26, 1000) * lo2 ** 2 * z ** 20

def variance_upper(z):
    return Q(2, 9) + hi2 * z ** 9 / 12
L = hi3 + hi2 * S / 3 + hi2 ** 2 * S ** 2 / 9 + hi2 ** 3 * S ** 3 / 81 - lo2 ** 4 * S ** 4 / 324
base = potential(S) - L + 1 + (L - hi2) * u + lo2 * v - branch_upper(x) - branch_upper(x * y)
save_one = (2 * lo2 - L) / 3 + power_lower(x) * (lo2 ** 2 - (L - lo2) ** 2) / 9
save_one += Q(26, 1000) * x ** 20 * (lo2 ** 3 - (L - lo2) ** 3) / 3
D = 2 * lo2 - L
first = q_lower(x) * D - x ** 9 * D ** 2 * variance_upper(x) / 2
save_two = q_lower(x * y) * first - (x * y) ** 9 * first ** 2 * variance_upper(x * y) / 2
constant = [Q(i, 10000000) for i in (-549378, 518337, 3850552, 308836, 2552524)]
linear = [Q(i, 10000000) for i in (2467210, -1612715, -5000000, -763799, -5000000)]
forecast = Q(9, 40)
cuts = [Q(17, 100), Q(195, 1000), Q(225, 1000)]
ratio_cuts = [Q(67, 100), Q(8, 10)]
limit = Q(65, 100)
rate = Q(37, 1000)
margin = Q(2, 1000000)
scale = 10 ** 12
guard = sorted(cuts + [Q(135, 1000)])
corners = [(Q(0), Q(0))] + list(product(guard + [limit], [Q(0)] + ratio_cuts + [Q(1)]))
xcuts = sorted(set([Q(0), forecast, limit] + cuts))

def charge(z, g, r):
    """Lower envelope of the child hinge charge at a cell corner."""
    geom = [max(0, g - j) for j in cuts] + [g * max(0, r - j) for j in ratio_cuts]
    c = sum((a * d for a, d in zip(constant, geom)))
    k = sum((a * d for a, d in zip(linear, geom)))
    return z ** 14 * (c + k * (power_lower(z) if k >= 0 else z ** 9))

def all_expressions():
    """Construct the rational polynomials entering the Bellman slack."""
    expr = [base - margin, save_one * u, save_one * v, save_two * u, save_two * v]
    for low in xcuts[:-1]:
        ex = -sum((S * (constant[i] + linear[i] * S) * (u - cut) for i, cut in enumerate(cuts) if low >= cut))
        if low >= forecast:
            ex -= rate * S * (u - forecast)
        expr.append(ex)
    for lo in [Q(0)] + ratio_cuts:
        expr.append(-sum((S * (constant[i + len(cuts)] + linear[i + len(cuts)] * S) * (v - r * u) for i, r in enumerate(ratio_cuts) if lo >= r)))
    for z in [x, x * y]:
        expr.extend((charge(z, g, r) for g, r in corners))
    return [Polynomial(t) for t in expr]

def bernstein(poly, degrees):
    """Enclose each tensor-product Bernstein coefficient in integer units of 10^-12."""
    terms = list(poly.terms.items())
    denominator = lcm(*(c.denominator for _, c in terms))
    t = np.zeros(tuple((d + 1 for d in degrees)), dtype=object)
    for i, c in terms:
        t[i] = c.numerator * (denominator // c.denominator)
    for ax, d in enumerate(degrees):
        cc = lcm(*(comb(d, j) for j in range(d + 1)))
        denominator *= cc
        b = np.moveaxis(t, ax, 0)
        v = b.copy()
        for i in range(d + 1):
            v[i] = sum((b[j] * (comb(i, j) * (cc // comb(d, j))) for j in range(i + 1)))
        t = np.moveaxis(v, 0, ax)
    t *= scale
    low = t // denominator
    high = (t + denominator - 1) // denominator
    assert low.min() > -scale * 2 and high.max() < scale * 2
    return (low.astype(np.int64), high.astype(np.int64))

def component_tables():
    """Use a common Bernstein degree for all terms of the slack."""
    polys = all_expressions()
    nonzero = [p for p in polys if p.terms]
    deg = tuple((max((max((exponent[axis] for exponent in p.terms)) for p in nonzero)) for axis in range(3)))
    return [bernstein(p, deg) for p in polys]

def dyadic_split(t, axis):
    """Lower Bernstein tables on both halves by downward-rounded de Casteljau averages."""
    b = np.moveaxis(t, axis, 0)
    d = len(b) - 1
    left = np.empty_like(b)
    right = np.empty_like(b)
    for i in range(d + 1):
        left[i] = b[0]
        right[d - i] = b[-1]
        b = (b[:-1] + b[1:]) // 2
    return (np.moveaxis(left, 0, axis), np.moveaxis(right, 0, axis))

@lru_cache(None)
def cbrt_minorant(t):
    """Compute floor(1000 * t^(2/3)) / 1000 by an integer comparison."""
    lo = 0
    hi = 1001
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if Q(mid, 1000) ** 3 <= t * t:
            lo = mid
        else:
            hi = mid
    return Q(lo, 1000)

def marks(g, r, ix):
    """Construct the two conservative saving fractions and their three cutoff masses."""
    g0 = max([Q(0)] + [w for w in guard if w < g])
    r0 = max([Q(0)] + [w for w in ratio_cuts if w < r])
    G0 = cbrt_minorant(g0)
    H0 = cbrt_minorant(g0 * r0)
    f = Q(97, 100) if xcuts[ix + 1] <= Q(225, 1000) else Q(951, 1000)
    A = Q(1, 3) + Q(9, 100) * G0
    B = Q(1, 3) + Q(124, 1000) * H0
    p1 = Q(int(10000 * f * A), 10000)
    p2 = Q(int(10000 * f * A * B), 10000)
    assert 0 <= p2 <= p1 <= Q(36, 100)
    return [(1, 1 - p1), (max(r, forecast), p1 - p2), (forecast, p2)]

def pieces(g, rg, b, rb, ix):
    """Partition the ratio v/u into intervals of affine expected saving lengths."""
    pts = set([Q(0), Q(1)] + ratio_cuts)
    ml = marks(g, rg, ix)
    mr = marks(b, rb, ix)
    for a, p in ml:
        for k, q in mr:
            G = a * g
            H = k * b
            pts.add(G)
            if H and G < H:
                pts.add(G / H)
    pts = sorted(pts)
    out = []
    for low, high in zip(pts, pts[1:]):
        z = (low + high) / 2
        cf = [Q(0)] * 4
        for a, p in ml:
            for k, q in mr:
                G = a * g
                H = k * b
                cs = (1, -1) if z >= G else (1 - G, 0)
                ds = (0, 0) if z <= G else (-G, 1) if G >= z * H else (0, 1 - H)
                for i, v in enumerate((*cs, *ds)):
                    cf[i] += p * q * v
        if out and low not in ratio_cuts and (out[-1][2] == cf):
            out[-1] = (out[-1][0], high, cf)
        else:
            out.append((low, high, cf))
    return out

def axis_index(q, depth):
    """Enclose the fourteenth root of q between adjacent dyadic grid points."""
    lo = 0
    hi = 1 << depth
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if Q(mid, 1 << depth) ** 14 <= q:
            lo = mid
        else:
            hi = mid
    return (lo, hi)

def weighted(t, weight):
    """Use the upper coefficient for a negative multiplier and round the product down."""
    b = t[0] if weight >= 0 else t[1]
    if weight == 0:
        return 0
    n = weight.numerator
    d = weight.denominator
    return (n * b.astype(object) // d).astype(np.int64)

@lru_cache(None)
def restriction(degree, low, high):
    """Compute dyadic restriction weights by convolving two binomial distributions."""
    depth = 25
    d = 1 << depth
    lo = axis_index(low, depth)[0]
    hi = axis_index(high, depth)[1]
    scale = 1 << 15
    denom = d ** degree
    top = []
    middle = []
    bottom = []
    for i in range(degree + 1):
        k = degree - i
        p = np.array([comb(k, j) * lo ** j * (d - lo) ** (k - j) for j in range(k + 1)], dtype=object)
        q = np.array([comb(i, j) * hi ** j * (d - hi) ** (i - j) for j in range(i + 1)], dtype=object)
        vals = np.convolve(p, q)
        assert sum(vals) == denom
        floor = vals * scale ** 3 // denom
        top.append(floor // (scale * scale))
        middle.append(floor // scale % scale)
        bottom.append(floor % scale)
    return tuple((np.array(table, dtype=np.int64) for table in (top, middle, bottom)))

def localize(t, axis, low, high):
    """Subtract the least control on each line before rounding the weights down."""
    B = 1 << 15
    hi, mid, lo = restriction(t.shape[axis] - 1, low, high)
    a = np.moveaxis(t, axis, 0).copy()
    shape = a.shape
    a = a.reshape(shape[0], -1)
    floor = a.min(axis=0)
    r = a - floor
    assert int(r.max()) * (B + 1) * (len(a) + 1) < 2 ** 63
    c = floor + (hi @ r + (mid @ r + lo @ r // B) // B) // B
    return np.moveaxis(c.reshape(shape), 0, axis)

def certify(t, xlow, xhi, ylow, yhi):
    """Certify nonnegativity on a parameter box by adaptive subdivision."""
    b = localize(localize(t, 1, xlow, xhi), 2, ylow, yhi)
    stack = [(b, 0)]
    count = depth = 0
    while stack:
        b, n = stack.pop()
        count += 1
        depth = max(depth, n)
        if b.min() >= 0:
            continue
        assert n < 72, (xlow, xhi, ylow, yhi, int(b.min()))
        l, r = dyadic_split(b, (1, 2, 0)[n % 3])
        stack.extend([(l, n + 1), (r, n + 1)])
    return (count, depth)

def verify(tabs, ix):
    """Check all pairs of child cells on one interval of the parent scale."""
    nx = len(xcuts) - 1
    nc = len(corners)
    off = 5 + nx + len(ratio_cuts) + 1
    regions = boxes = dep = 0
    cache = {}
    for i, (g, rg) in enumerate(corners):
        for j, (b, rb) in enumerate(corners):
            elem = tabs[0][0] + tabs[5 + ix][0] + tabs[off + i][0] + tabs[off + nc + j][0]
            for low, high, coeff in pieces(g, rg, b, rb, ix):
                ik = sum((1 for c in ratio_cuts if c <= low))
                t = elem + tabs[5 + nx + ik][0]
                for k, w in enumerate(coeff):
                    assert -1 <= w <= 1
                    if (k, w) not in cache:
                        cache[k, w] = weighted(tabs[k + 1], w)
                    t += cache[k, w]
                n, d = certify(t, xcuts[ix], xcuts[ix + 1], low, high)
                regions += 1
                boxes += n
                dep = max(dep, d)
    return (regions, boxes, dep)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("intervals", nargs="*", type=int, metavar="N",
                        help="scale interval index, 0 through 3")
    args = parser.parse_args()
    if any(i not in range(4) for i in args.intervals):
        parser.error("interval indices must be between 0 and 3")

    expected = [(3238, 88490, 31), (3238, 6670, 9),
                (3238, 3414, 5), (3238, 4434, 6)]
    labels = ["[0,0.17]", "[0.17,0.195]", "[0.195,0.225]", "[0.225,0.65]"]
    tables = component_tables()
    print("u interval       polynomial pieces   boxes visited   greatest depth")
    for i in args.intervals or range(4):
        result = verify(tables, i)
        assert result == expected[i], (i, result, expected[i])
        pieces_count, boxes_count, depth = result
        print(f"{labels[i]:<16} {pieces_count:>16} {boxes_count:>15} {depth:>16}",
              flush=True)

    # Rational bounds for a_star = a log(2) / (log(2) - log(1 - 1/C)).
    inverse_mass = Q(10, 1499091)
    log_lower = inverse_mass + inverse_mass**2 / 2
    log_upper = log_lower + inverse_mass**3 / (3 * (1 - inverse_mass))
    exponent_lower = a / (1 + log_upper / lo2)
    exponent_upper = a / (1 + log_lower / hi2)
    assert Q(3556265775098, 10**13) < exponent_lower <= exponent_upper
    assert exponent_upper < Q(3556265775100, 10**13)
    if not args.intervals or set(args.intervals) == set(range(4)):
        print("Bellman margin >= 2/1000000; 0.3556265775098 < a_star < 0.3556265775100")


if __name__ == "__main__":
    main()
