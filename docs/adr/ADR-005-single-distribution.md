# ADR-005 — One installable distribution, eight packages inside it

**Status:** accepted · **Date:** 2026-06-03 · **Requirement:** NFR-07

## Context
The capabilities are cleanly separated. The obvious next step — publishing each
as its own distribution with its own version — was considered and rejected.

## Options
| Option | Consequence |
|---|---|
| Eight distributions | eight version numbers, a resolver problem, and an offline install that needs eight wheels present |
| One distribution, eight packages | one version, one install, boundaries still enforced by the contract layer |

## Decision
One distribution, `leanlm`. The internal boundaries are real but enforced by
convention and by tests, not by the packaging system:

- capabilities communicate only through canonical DTOs;
- `apps/ccm.py` fails CI if a capability reaches outside its package structure;
- each capability declares its own `CapabilitySpec` and owns its stages.

## Why the trade-off falls this way
The product's promise is that it installs on a laptop with no working package
index. Eight interdependent distributions is the opposite of that promise. The
isolation that actually matters — a capability being replaceable without touching
its neighbours — is a property of the contracts, and we keep it.

## Reversibility
Splitting later is mechanical: each package already has its own contracts,
policies, tests and benchmarks. Nothing in the code assumes a shared module.
