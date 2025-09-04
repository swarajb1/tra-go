SHELL := /bin/bash

.PHONY: create-venv install run clean clean_logs get_clean_data new-yf-data script_1 create_data_folders generate_dot_env

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


create-venv:  ## Create Python virtual environment
	python3 -m venv .venv


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

get_clean_data:
	PYTHONPATH=$(WORKDIR)/ $(PYTHON) tra_go/data_clean.py

new-yf-data:
	PYTHONPATH=$(WORKDIR)/ $(PYTHON) tra_go/download_data_yf.py


script_1:
	PYTHONPATH=$(WORKDIR)/ $(PYTHON) script_1.py


create_data_folders:  ## Create data and training folders
	@mkdir -p data_z
	@echo "Created folder: data_z"

	@mkdir -p data_cleaned
	@echo "Created folder: data_cleaned"

	@mkdir -p data_training
	@echo "Created folder: data_training"

	@mkdir -p training
	@echo "Created folder: training"

	@mkdir -p training/graphs
	@echo "Created folder: training/graphs"
	@mkdir -p training/models
	@echo "Created folder: training/models"
	@mkdir -p training/models_saved
	@echo "Created folder: training/models_saved"
	@mkdir -p training/models_saved_double
	@echo "Created folder: training/models_saved_double"
	@mkdir -p training/models_saved_triple
	@echo "Created folder: training/models_saved_triple"
	@mkdir -p training/models_z_old
	@echo "Created folder: training/models_z_old"
	@mkdir -p training/models_zz_discarded
	@echo "Created folder: training/models_zz_discarded"


generate_dot_env:  ## Create .env from template if it does not exist.
	@if [[ ! -e .env ]]; then \
		cp .env.template .env; \
	fi
