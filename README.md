# The normal order of the Erdős–Hooley Δ-function

**Tseng**

The paper proves that $\Delta(n)=(\log\log n)^{\alpha+o(1)}$ on a set of natural density one. It characterizes $\alpha$ by two variational formulas that give rigorous bounds to arbitrary precision. The explicit bounds proved in the paper are $0.35553850187<\alpha<0.3556265775100$.

[Read the paper (PDF)](Erdos_Hooley_Delta.pdf) · [LaTeX source](Erdos_Hooley_Delta.tex)

AI Disclosure: The author selected the research topic; artificial intelligence completed all mathematical derivations, proofs, and writing.

## Reproducing the numerical bounds

Run the following commands from the repository root with Python 3.9 or later and assertions enabled. The lower-bound certificate uses the standard library. The upper-bound certificate also requires NumPy.

```sh
python3 -B certificates/sixteen_certificate.py
python3 -B certificates/recursive_certificate.py
```

The expected outputs are included for the [lower bound](certificates/sixteen_expected.txt) and the [upper bound](certificates/recursive_expected.txt). Sections 7 and 8 of the paper derive the inequalities checked by the two programs.
