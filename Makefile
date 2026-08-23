.PHONY: help install install-optional install-all test lint ccm doctor ingest ask accuracy accuracy-fr score preflight provision provision-check measure-docker bench naive profile corpus baseline submission serve ci clean

PYTHON ?= python3
PROFILE ?= development

help:
	@echo "LeanLM -- offline LLM optimization layer"
	@echo
	@echo "  make install       install the package (no third-party dependency)"
	@echo "  make install-optional  pypdf, psutil, PyYAML, pdfminer (pure Python)"
	@echo "  make install-all   the above plus llama-cpp-python (needs a compiler)"
	@echo "  make doctor        check whether this machine can run LeanLM"
	@echo "  make ingest        index datasets/enterprise"
	@echo "  make ask Q='...'   ask a question"
	@echo "  make serve         start the local workspace on 127.0.0.1:8770"
	@echo "  make accuracy      measure accuracy against ground truth (50% of the score)"
	@echo "  make score         compute the ADTC score from the results"
	@echo "  make naive         measure the same model with NO LeanLM layer"
	@echo "  make bench         run the benchmark campaign"
	@echo "  make profile Q=... where the time goes in a request"
	@echo "  make corpus        regenerate the large test corpus"
	@echo "  make preflight     which routes can produce a submittable measurement"
	@echo "  make provision     install llama.cpp + adtc-profiler on Ubuntu"
	@echo "  make measure-docker  profile under a 7.5 GB cap"
	@echo "  make baseline      record the current results as the baseline"
	@echo "  make submission    build and check the competition package"
	@echo "  make test          run the test suite"
	@echo "  make ccm           verify the capability-to-code mapping"
	@echo "  make ci            everything the pipeline runs"

install:
	$(PYTHON) -m pip install -e .

install-optional:
	$(PYTHON) -m pip install -e ".[optional,dev]"

install-all:
	$(PYTHON) -m pip install -e ".[full,dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check src tests

ccm:
	$(PYTHON) -m leanlm.apps.cli ccm

doctor:
	$(PYTHON) -m leanlm.apps.cli --profile $(PROFILE) doctor

ingest:
	$(PYTHON) -m leanlm.apps.cli --profile $(PROFILE) ingest datasets/enterprise_en

ask:
	$(PYTHON) -m leanlm.apps.cli --profile $(PROFILE) ask "$(Q)"

serve:
	$(PYTHON) -m leanlm.apps.cli --profile demo serve

bench:
	$(PYTHON) -m leanlm.apps.cli --profile benchmark bench --with-profile

accuracy:
	$(PYTHON) -m leanlm.apps.cli --profile benchmark accuracy

accuracy-fr:
	$(PYTHON) -m leanlm.apps.cli --profile benchmark ingest datasets/enterprise
	$(PYTHON) -m leanlm.apps.cli --profile benchmark accuracy \
	  --evaluation datasets/enterprise/evaluation.json

score:
	$(PYTHON) -m leanlm.apps.cli score

naive:
	$(PYTHON) -m leanlm.apps.cli --profile benchmark baseline --run

profile:
	$(PYTHON) -m leanlm.apps.cli --profile benchmark profile "$(Q)"

preflight:
	$(PYTHON) -m leanlm.apps.cli preflight

measure-docker:
	bash scripts/measure_docker.sh $(SUBMISSION)

provision:
	bash scripts/provision_ubuntu.sh

provision-check:
	bash scripts/provision_ubuntu.sh --check

corpus:
	$(PYTHON) scripts/make_corpus.py --documents 40

baseline:
	$(PYTHON) -m leanlm.apps.cli --profile benchmark bench --save-baseline

submission:
	$(PYTHON) -m leanlm.apps.cli --profile competition submission --output dist/submission

ci:
	./scripts/run_ci.sh

clean:
	rm -rf .pytest_cache .leanlm dist build **/__pycache__ .coverage
