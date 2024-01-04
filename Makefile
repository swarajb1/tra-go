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


WORKDIR=tra_go


install:  ## Install poetry to run on local
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install poetry
	$(PYTHON) -m poetry install
	poetry run pre-commit install

run: ## run the program
	PYTHONPATH=$(WORKDIR)/ python3 main.py

clean:
	rm -rf __pycache__


new-yf-data:
	PYTHONPATH=$(WORKDIR) tra_go/download_yf_data.py


gst:
	gst
