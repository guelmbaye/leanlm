# Where to measure, and where not to

> Start with `leanlm preflight`. It reports each route below, what it is missing,
> the command that fixes it, and which routes must not be used for the submitted
> numbers whatever the local hardware happens to be.

The audit sandbox re-runs every submission and compares its numbers to yours:

| Metric | Tolerated | Fails beyond |
|---|---|---|
| `memory.peak_rss_mb` | ±15% | 50% |
| `throughput.tokens_per_second_generation` | ±25% | 50% |
| `throughput.first_token_latency_ms` | ±25% | 50% |

That makes the measuring machine part of the submission. A number produced on
hardware far from the standard profile — **4 vCPU, 8 GB RAM, integrated GPU** —
does not merely score badly. It risks a *failed comparison*, which is a
different and worse outcome.

`leanlm doctor` now states the divergence rather than leaving it to be
discovered afterwards.

## A development laptop is the right place for three things

**Functional verification.** Does it ingest, retrieve, refuse correctly, package
conformantly? None of that depends on the hardware. `make test`, `make ci`,
`leanlm accuracy`.

**Model selection.** This is where the work pays off. The score pulls in two
directions: the accuracy stage rewards a larger model, throughput and efficiency
reward a smaller one. `S_perf = min(TPS/15, 1.0)` means every token per second
below 15 costs two points of the final score, and `S_eff = (7 − peak_gb)/7`
means every gigabyte costs nearly three. The optimum is empirical, and a harness
that runs candidates under identical conditions and reports dispersion is the
tool for finding it. Relative rankings between candidates survive a change of
machine far better than absolute figures do.

**Prompt and report work.** The four prompts and `REPORT.md` are read by judges,
not measured.

## And the wrong place for one

**The `submission.json` you ship.** Produce that where the hardware resembles
the profile.

## Constraining a machine that is too generous

More RAM than the profile is not a safe place to measure an 8 GB claim: nothing
swaps, so the failure that disqualifies cannot be observed. The profiler ships a
Dockerfile precisely for this:

```bash
docker build -t adtc-profiler:latest .

docker run --rm --memory=7.5g --cpus=4 \
  -v "/path/to/submission:/submission:ro" \
  -v "/path/to/artifacts:/artifacts" \
  adtc-profiler:latest run \
  --submission /submission \
  --mode audit \
  --output /artifacts/audit.json
```

`--memory=7.5g` reproduces the constraint; `--cpus=4` reproduces the vCPU count
if the host has at least four logical processors. What a container cannot
reproduce is the speed of the cores it is given: capping a slow processor does
not make it a fast one.

## Cores are not threads

A four-thread dual-core is not a four-core machine. Hyper-threaded siblings
share execution units, and llama.cpp generation is bound by compute and memory
bandwidth — precisely the resources siblings share. Expect a two-core machine to
read well below a four-core one at the same clock, and expect that gap to show
up as variance against the audit run.

## Thermal

The 10-point penalty applies above 85 °C or on flagged throttling. Low-power
mobile processors in thin chassis reach that under sustained generation load
routinely — it is one of the likelier ways to lose points quietly.

The profiler reads temperature through Linux sensors. On Windows the field will
be unmeasured, and an unmeasured thermal is an unquantified risk of the penalty,
not an absence of one. Install `lm-sensors` on the machine that produces the
submitted run.

---

## No conforming machine? Rent one for an afternoon

This is the ordinary situation, and it has a clean answer. **The audit runs in a
cloud VM** — the profiler's own environment block distinguishes
`participant_laptop` from `audit_cloud_vm`, and the comparator expects exactly
that pair. So a cloud instance is not a workaround. It is closer to the
environment your numbers will be compared against than any laptop.

### What to provision

**4 vCPU, 8 GB RAM, Ubuntu 24.04 LTS, dedicated CPU.**

Dedicated rather than shared matters more than the price difference. A
shared-vCPU instance gives you a slice of a core alongside other tenants, and
throughput varies run to run by more than the ±25% you are allowed to differ
from the audit. On DigitalOcean the CPU-Optimized line is dedicated; the Basic
line is shared. Equivalent tiers exist elsewhere (AWS `c7i.xlarge`, Hetzner
CCX13, GCP `c3-highcpu-4`). Prices change — check before provisioning — but a
measurement campaign is a few hours, so the cost is a couple of dollars, not a
subscription.

Take the 8 GB tier rather than a larger one and then capping it: matching the
profile directly is simpler than constraining a machine that exceeds it.

```bash
git clone <your-submission-repo> && cd <repo>
curl -O https://raw.githubusercontent.com/<you>/leanlm/main/scripts/provision_ubuntu.sh
bash provision_ubuntu.sh          # llama.cpp, Python 3.11+, adtc-profiler
bash download_model.sh
adtc-profiler run --submission . --mode participant --output submission.json
```

Then destroy the instance. Nothing about the measurement needs it to persist.

### The free alternative, and its quiet advantage

**GitHub Actions.** The standard `ubuntu-latest` runner is 4 vCPU, and the
runner minutes are free on public repositories — which yours must be anyway.

Its real advantage is not the price. A measurement produced in CI is *logged,
timestamped and re-runnable by anyone*, including a judge who wonders whether
the numbers are real. That is a stronger position than a JSON file whose
provenance is your word.

Cap the memory to the profile, since the runner has more than 8 GB:

```yaml
- name: Measure
  run: |
    docker run --rm --memory=7.5g --cpus=4 \
      -v "${{ github.workspace }}:/submission:ro" \
      -v "${{ github.workspace }}/artifacts:/artifacts" \
      adtc-profiler:latest run --submission /submission \
      --mode participant --output /artifacts/submission.json
```

### What a VM cannot give you

**Thermal.** Hypervisors do not expose CPU temperature to guests, so
`core_temp_c_peak` will be `null` and `throttled` will be `false`. Reading the
profiler's source: with no samples, no penalty is applied. The audit VM has the
same limitation, so thermal is unmeasured on both sides rather than being an
advantage either way.

This is worth understanding rather than exploiting. A thermal penalty exists
because a laptop that cooks itself is a laptop that throttles, and your users
will feel it even if the scoring never sees it.

### Say where you measured

`environment.capture()` records `cpu_model`, `ram_gb`, `gpu` and `os` into the
report. The hardware you used is visible to the judges whether or not you
mention it. So mention it, in `REPORT.md`, in one line:

> Measured on a 4 vCPU / 8 GB Ubuntu 24.04 instance, chosen to match the
> standard laptop profile; our development hardware (dual-core, 16 GB) diverges
> from it in both directions.

A stated limitation reads as method. The same limitation discovered by a
reviewer reads as something else.

---

## Everything each route needs

`leanlm preflight` checks all of this and prints the shortest path that works
from where you are.

| Route | Needs | Submittable |
|---|---|---|
| This machine, development | the package installed | no, by construction |
| This machine, native profiler | Python ≥3.11, `llama-bench`, `adtc-profiler`, a built package with weights | only if the hardware matches the profile |
| This machine, in Docker | Docker running, a built package | only if the host has ≥4 logical CPUs |
| Cloud instance | `scripts/provision_ubuntu.sh` | yes |
| GitHub Actions | `.github/workflows/measure.yml` | yes |

Commands:

```bash
leanlm preflight                    # what works from here
make provision                      # Ubuntu: llama.cpp + the official profiler
make measure-docker                 # capped to 7.5 GB, 4 CPUs
gh workflow run measure.yml         # in CI, logged and re-runnable
```

The development route is marked unsuitable regardless of hardware. That is
deliberate: it is the route someone reaches for at 2 a.m. on the day of the
deadline, and the moment it becomes conditionally acceptable is the moment it
gets used.
