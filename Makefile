SHELL := /bin/bash

PYTHON=$(shell which python3)
ifeq ($(PYTHON),)
	PYTHON=$(shell which python)
	ifneq ($(PYTHON),)
		ifeq ($(shell $(PYTHON) -V | grep '^Python 3'),)
			PYTHON=
		endif
	endif
endif


PYTHON = ./.venv/bin/python

WORKDIR=tra_go


install:  ## Install poetry to run on local
	$(MAKE) generate_dot_env
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install poetry
	$(PYTHON) -m poetry install
	poetry run pre-commit install

run: ## run the program
	clear
	PYTHONPATH=$(WORKDIR)/ $(PYTHON) tra_go/main.py $(model) $(num) $(move)

clean:
	rm -rf __pycache__


clean_logs:
	PYTHONPATH=$(WORKDIR)/ $(PYTHON) support/clean_logs.py


new-yf-data:
	PYTHONPATH=$(WORKDIR)/ $(PYTHON) tra_go/download_data_yf.py


script_1:
	PYTHONPATH=$(WORKDIR)/ $(PYTHON) script_1.py


generate_dot_env:  ## Create .env from template if it does not exist.
	@if [[ ! -e .env ]]; then \
		cp .env.template .env; \
	fi
