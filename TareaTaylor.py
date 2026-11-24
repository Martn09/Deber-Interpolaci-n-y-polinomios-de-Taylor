import math
import matplotlib.pyplot as plt
import numpy as np

def taylor_sin(x, n):
  """Calcula el polinomio de Maclaurin de grado n (impar) para sin(x)

  actualizando cada término a partir del anterior.
  """
  x_arr = np.asarray(x, dtype=float)
  termino = x_arr.copy()
  suma = termino.copy()

  num_pasos = (n - 1) // 2
  for k in range(num_pasos):
    termino = termino * (-(x_arr**2)) / ((2 * k + 2) * (2 * k + 3))
    suma = suma + termino

  return float(suma) if np.ndim(x) == 0 else suma


print(f"P_9(pi) = {taylor_sin(np.pi, 9):.6f}")

x1 = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
plt.figure(figsize=(9, 5))
plt.plot(x1, np.sin(x1), "k-", linewidth=2, label="sin(x)")
for grado in [1, 3, 5, 7, 9]:
  plt.plot(x1, taylor_sin(x1, grado), "--", label=f"P_{grado}(x)")
plt.ylim(-2, 2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Aproximación de sin(x) con polinomios de Taylor")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("grafica1.png", dpi=150)
plt.close()

grados = list(range(1, 22, 2))
x_pi = np.linspace(-np.pi, np.pi, 2001)
x_pi4 = np.linspace(-np.pi / 4, np.pi / 4, 2001)

err_pi = [np.max(np.abs(np.sin(x_pi) - taylor_sin(x_pi, n))) for n in grados]
err_pi4 = [np.max(np.abs(np.sin(x_pi4) - taylor_sin(x_pi4, n))) for n in grados]

for n, e1, e2 in zip(grados, err_pi, err_pi4):
  print(f"n={n:2d} | max_err[-pi,pi]={e1:.3e} | max_err[-pi/4,pi/4]={e2:.3e}")

plt.figure(figsize=(8, 5))
plt.semilogy(grados, err_pi, "o-", label="Intervalo [-π, π]")
plt.semilogy(grados, err_pi4, "s-", label="Intervalo [-π/4, π/4]")
plt.axhline(1e-6, color="r", linestyle=":", label="Tolerancia 10^-6")
plt.xticks(grados)
plt.xlabel("Grado del polinomio (n)")
plt.ylabel("Error máximo absoluto (escala log)")
plt.title("Error máximo de Taylor para sin(x) según el grado")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("grafica2.png", dpi=150)
plt.close()

def serie_geometrica(x, N):
  """Calcula S_N(x) = sum_{k=0}^N (-1)^k x^(2k) actualizando el término."""
  x_arr = np.asarray(x, dtype=float)
  termino = np.ones_like(x_arr)
  suma = termino.copy()

  for _ in range(N):
    termino = termino * (-(x_arr**2))
    suma = suma + termino

  return float(suma) if np.ndim(x) == 0 else suma


print(f"S_40(0.9) = {serie_geometrica(0.9, 40):.8f}")
print(f"S_40(1.1) = {serie_geometrica(1.1, 40):.8f}")

x3 = np.linspace(-1.5, 1.5, 1000)
plt.figure(figsize=(9, 5))
plt.plot(x3, 1 / (1 + x3**2), "k-", linewidth=2, label="f(x) = 1 / (1 + x²)")
for N in [2, 5, 10, 20]:
  plt.plot(x3, serie_geometrica(x3, N), "--", label=f"S_{N}(x)")
plt.ylim(-0.5, 1.5)
plt.axvline(-1, color="gray", linestyle=":", alpha=0.7)
plt.axvline(1, color="gray", linestyle=":", alpha=0.7)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sumas parciales de Maclaurin para 1 / (1 + x²)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("grafica3.png", dpi=150)
plt.close()

def taylor_exp(x, N):
  termino = 1.0
  suma = 1.0
  max_abs_termino = 1.0
  for k in range(1, N + 1):
    termino = termino * x / k
    suma += termino
    if abs(termino) > max_abs_termino:
      max_abs_termino = abs(termino)
  return suma, max_abs_termino


p100_neg, max_term = taylor_exp(-20, 100)
p100_pos, _ = taylor_exp(20, 100)

print(f"Exacto math.exp(-20) = {math.exp(-20):.15e}")
print(f"P_100(-20)           = {p100_neg:.15e}")
print(f"Max |termino|        = {max_term:.15e}")
print(f"1 / P_100(20)        = {1.0 / p100_pos:.15e}")