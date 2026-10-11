"""Live demo for the neural SDE lecture (CMSC 838B).

Differentiates a toy neural SDE two ways and shows the gradients agree:
  1. BacksolveAdjoint          -> stochastic adjoint, O(1) memory in the number of steps
  2. RecursiveCheckpointAdjoint -> backprop through the solver steps (discretize-then-differentiate)

Install:  pip install "jax[cpu]" diffrax
Run:      python neural_sde_demo.py
"""
import jax, jax.numpy as jnp, diffrax as dfx
jax.config.update("jax_enable_x64", True)
d = 2
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
params = {"A": 0.3*jax.random.normal(k1, (d, d)), "s": jnp.array(0.2)}
z0 = jnp.ones(d)
mu    = lambda p, z, t: jnp.tanh(p["A"] @ z)          # drift network (toy)
sigma = lambda p, z, t: p["s"] * jnp.eye(d)           # diffusion, d x d
objective = lambda z: jnp.sum(z**2)

def loss(params, key, adjoint):
    drift = lambda t, z, p: mu(p, z, t)
    diff  = lambda t, z, p: sigma(p, z, t)
    bm    = dfx.VirtualBrownianTree(0., 1., tol=1e-3, shape=(d,), key=key)
    terms = dfx.MultiTerm(dfx.ODETerm(drift), dfx.ControlTerm(diff, bm))
    sol   = dfx.diffeqsolve(terms, dfx.ReversibleHeun(), 0., 1., dt0=0.01,
                            y0=z0, args=params, adjoint=adjoint)
    return objective(sol.ys[-1])

g_adj = jax.grad(loss)(params, k2, dfx.BacksolveAdjoint())
g_dto = jax.grad(loss)(params, k2, dfx.RecursiveCheckpointAdjoint())
print("adjoint  :", g_adj)
print("backprop :", g_dto)
print("max abs diff:", max(float(jnp.max(jnp.abs(g_adj[k]-g_dto[k]))) for k in g_adj))
