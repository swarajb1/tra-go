SHELL := /bin/bash

# Variables
PYTHON_VENV := ./.venv/bin/python
POETRY := $(PYTHON_VENV) -m poetry
WORKDIR := tra_go
LOGDIR := training/logs
PROJECT_NAME := tra-go

# Color codes for better output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
WHITE := \033[37m
RESET := \033[0m

# .PHONY targets
.PHONY: help create-venv install run clean clean-all clean-logs clean-cache clean-temp clean-discarded \
        get-clean-data new-yf-data tensorboard script-1 create-data-folders \
        generate-dot-env setup dev-setup lint format test check status check-venv verify-venv \
        train train-parallel eval-new eval-saved eval-saved-double eval-saved-triple \
        eval-old eval-discarded show-models

# Default target
.DEFAULT_GOAL := help


# Help target
help: ## Show this help message
	@printf "$(BLUE)$(PROJECT_NAME) - Machine Learning Trading System$(RESET)\n"
	@printf "$(BLUE)================================================$(RESET)\n"
	@printf "\n"
	@printf "$(GREEN)Available targets:$(RESET)\n"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\n"
	@printf "$(GREEN)Training parameters:$(RESET)\n"
	@printf "  MODEL             Model type (train, eval_new, saved, etc.)\n"
	@printf "  NUM               Number of models to evaluate (default: 6)\n"
	@printf "  MOVE              Move files after evaluation (true/false, default: false)\n"
	@printf "\n"
	@printf "$(GREEN)Examples:$(RESET)\n"
	@printf "  make train\n"
	@printf "  make eval-saved NUM=10 MOVE=true\n"
	@printf "  make setup                    # Complete project setup\n"

# Setup and Installation targets
setup: create-venv install create-data-folders ## Complete project setup
	@printf "$(GREEN)✓ Project setup completed successfully!$(RESET)\n"

dev-setup: setup ## Setup development environment with pre-commit hooks
	$(POETRY) run pre-commit install
	@printf "$(GREEN)✓ Development environment setup completed!$(RESET)\n"

create-venv: ## Create Python virtual environment
	@printf "$(BLUE)Creating Python virtual environment...$(RESET)\n"
	python3 -m venv .venv
	@printf "$(GREEN)✓ Virtual environment created at .venv$(RESET)\n"

install: generate-dot-env ## Install dependencies with Poetry
	@printf "$(BLUE)Installing dependencies...$(RESET)\n"
	@if [ ! -f .venv/bin/python ]; then \
		printf "$(RED)Virtual environment not found. Run 'make create-venv' first.$(RESET)\n"; \
		exit 1; \
	fi
	$(PYTHON_VENV) -m pip install --upgrade pip
	$(PYTHON_VENV) -m pip install poetry
	$(PYTHON_VENV) -m poetry install --no-root
	@printf "$(GREEN)✓ Dependencies installed successfully!$(RESET)\n"

# Main execution targets
run: ## Run the program with specified parameters
	@printf "$(BLUE)Running TRA-GO with MODEL=$(MODEL) NUM=$(NUM) MOVE=$(MOVE)$(RESET)\n"
	@clear
	PYTHONPATH=$(WORKDIR)/ $(PYTHON_VENV) tra_go/main.py $(MODEL) $(NUM) $(MOVE)

# Training targets
train: ## Train models on all tickers
	@$(MAKE) run MODEL=train

# Evaluation targets
eval-new: ## Evaluate newly trained models
	@$(MAKE) run MODEL=training_new

eval-trained: ## Evaluate trained models
	@$(MAKE) run MODEL=eval_trained NUM=$(NUM) MOVE=$(MOVE)

eval-saved: ## Evaluate saved models
	@$(MAKE) run MODEL=saved NUM=$(NUM) MOVE=$(MOVE)

eval-saved-double: ## Evaluate double-tier saved models
	@$(MAKE) run MODEL=saved_double NUM=$(NUM) MOVE=$(MOVE)

eval-saved-triple: ## Evaluate triple-tier saved models
	@$(MAKE) run MODEL=saved_triple NUM=$(NUM) MOVE=$(MOVE)

eval-old: ## Evaluate old models
	@$(MAKE) run MODEL=old NUM=$(NUM) MOVE=$(MOVE)

eval-discarded: ## Evaluate discarded models
	@$(MAKE) run MODEL=discarded NUM=$(NUM) MOVE=true

# Data management targets
get-clean-data: ## Clean and process market data
	@printf "$(BLUE)Cleaning and processing market data...$(RESET)\n"
	PYTHONPATH=$(WORKDIR)/ $(PYTHON_VENV) tra_go/data_clean.py
	@printf "$(GREEN)✓ Data cleaning completed!$(RESET)\n"

new-yf-data: ## Download new data from Yahoo Finance
	@printf "$(BLUE)Downloading new data from Yahoo Finance...$(RESET)\n"
	PYTHONPATH=$(WORKDIR)/ $(PYTHON_VENV) tra_go/download_data_yf.py
	@printf "$(GREEN)✓ Data download completed!$(RESET)\n"

# Monitoring and analysis targets
tensorboard: ## Launch TensorBoard to view training logs
	@printf "$(BLUE)Launching TensorBoard at http://localhost:6006$(RESET)\n"
	@if [ ! -d "$(LOGDIR)" ]; then \
		printf "$(RED)Training logs directory not found: $(LOGDIR)$(RESET)\n"; \
		printf "$(YELLOW)Run some training first or create the directory$(RESET)\n"; \
		exit 1; \
	fi
	$(PYTHON_VENV) -m tensorboard.main --logdir=$(LOGDIR)

show-models: ## Show model directory structure and counts
	@printf "$(BLUE)╔══════════════════════════════════════════════════════════════════════╗$(RESET)\n"
	@printf "$(BLUE)║                       Model Directory Overview                       ║$(RESET)\n"
	@printf "$(BLUE)╚══════════════════════════════════════════════════════════════════════╝$(RESET)\n"
	@for dir in training/models training/models_saved training/models_saved_double training/models_saved_triple training/models_z_old training/models_zz_discarded; do \
		if [ -d "$$dir" ]; then \
			count=$$(find "$$dir" -name "*.h5" -o -name "*.keras" 2>/dev/null | wc -l | tr -d ' '); \
			if [ "$$count" -gt 0 ]; then \
				printf "  $(GREEN)✓$(RESET) $(YELLOW)%-40s$(RESET) $(WHITE)%s models$(RESET)\n" "$$dir" "$$count"; \
			else \
				printf "  $(YELLOW)○$(RESET) $(WHITE)%-40s$(RESET) $(WHITE)%s models$(RESET)\n" "$$dir" "$$count"; \
			fi \
		else \
			printf "  $(RED)✗$(RESET) $(RED)%-40s$(RESET) $(RED)not found$(RESET)\n" "$$dir"; \
		fi \
	done
	@printf "\n"

status: ## Show project status and configuration
	@printf "$(BLUE)╔═══════════════════════════════════════════════════════════════╗$(RESET)\n"
	@printf "$(BLUE)║                     TRA-GO Project Status                     ║$(RESET)\n"
	@printf "$(BLUE)╚═══════════════════════════════════════════════════════════════╝$(RESET)\n"
	@printf "  $(WHITE)%-35s$(RESET) $(WHITE)%s$(RESET)\n" "Python executable:" "$(PYTHON_VENV)"
	@printf "  $(WHITE)%-35s$(RESET) $(WHITE)%s$(RESET)\n" "Working directory:" "$(WORKDIR)"
	@printf "  $(WHITE)%-35s$(RESET) " "Virtual environment:"; [ -d .venv ] && printf "$(GREEN)✓ Active$(RESET)\n" || printf "$(RED)✗ Not found$(RESET)\n"
	@printf "  $(WHITE)%-35s$(RESET) " ".env configuration:"; [ -f .env ] && printf "$(GREEN)✓ Present$(RESET)\n" || printf "$(RED)✗ Not found$(RESET)\n"
	@printf "  $(WHITE)%-35s$(RESET) " "Data directories:"; [ -d data_training ] && printf "$(GREEN)✓ Ready$(RESET)\n" || printf "$(RED)✗ Missing$(RESET)\n"
	@printf "  $(WHITE)%-35s$(RESET) " "Training directories:"; [ -d training ] && printf "$(GREEN)✓ Ready$(RESET)\n" || printf "$(RED)✗ Missing$(RESET)\n"
	@printf "\n"
	@total=0; \
	for dir in training/models*; do \
		if [ -d "$$dir" ]; then \
			count=$$(find "$$dir" -name "*.h5" -o -name "*.keras" 2>/dev/null | wc -l | tr -d ' '); \
			total=$$((total + count)); \
		fi \
	done; \
	printf "  $(YELLOW)%-35s$(RESET) $(YELLOW)%s models$(RESET)\n" "Total models available:" "$$total"
	@printf "\n"


# Utility targets
script-1: ## Run script_1.py utility
	@printf "$(BLUE)Running script_1.py...$(RESET)\n"
	PYTHONPATH=$(WORKDIR)/ $(PYTHON_VENV) scripts/script_1.py

clean-logs: ## Clean training logs
	@printf "$(BLUE)Cleaning training logs...$(RESET)\n"
	PYTHONPATH=$(WORKDIR)/ $(PYTHON_VENV) support/clean_logs.py
	@printf "$(GREEN)✓ Training logs cleaned!$(RESET)\n"

clean-cache: ## Remove Python cache files
	@printf "$(BLUE)Removing Python cache files...$(RESET)\n"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	@printf "$(GREEN)✓ Cache files removed!$(RESET)\n"

clean-temp: ## Remove temporary files from temp/ directories
	@printf "$(BLUE)Removing temporary files...$(RESET)\n"
	rm -rf temp/* 2>/dev/null || true
	@printf "$(GREEN)✓ Temporary files removed!$(RESET)\n"

clean: clean-cache ## Basic cleanup (cache files only)

clean-all: clean-cache clean-logs clean-temp ## Complete cleanup (cache + logs + temp)
	@printf "$(GREEN)✓ Complete cleanup finished!$(RESET)\n"

clean-discarded: ## Remove all models in discarded folder
	@printf "$(BLUE)Removing all models in discarded folder...$(RESET)\n"
	rm -rf training/models_zz_discarded/* 2>/dev/null || true
	@printf "$(GREEN)✓ Discarded models removed!$(RESET)\n"

# Project structure targets
create-data-folders: ## Create data and training folders
	@printf "$(BLUE)╔═══════════════════════════════════════════════════════════════╗$(RESET)\n"
	@printf "$(BLUE)║                   Creating Project Directories                ║$(RESET)\n"
	@printf "$(BLUE)╚═══════════════════════════════════════════════════════════════╝$(RESET)\n"
	@mkdir -p data_z && printf "  $(GREEN)✓$(RESET) $(WHITE)%-50s$(RESET) $(GREEN)[Created]$(RESET)\n" "data_z"
	@mkdir -p data_cleaned && printf "  $(GREEN)✓$(RESET) $(WHITE)%-50s$(RESET) $(GREEN)[Created]$(RESET)\n" "data_cleaned"
	@mkdir -p data_training && printf "  $(GREEN)✓$(RESET) $(WHITE)%-50s$(RESET) $(GREEN)[Created]$(RESET)\n" "data_training"
	@mkdir -p training && printf "  $(GREEN)✓$(RESET) $(WHITE)%-50s$(RESET) $(GREEN)[Created]$(RESET)\n" "training"
	@mkdir -p training/logs && printf "  $(GREEN)✓$(RESET) $(WHITE)%-50s$(RESET) $(GREEN)[Created]$(RESET)\n" "training/logs"
	@mkdir -p training/graphs && printf "  $(GREEN)✓$(RESET) $(WHITE)%-50s$(RESET) $(GREEN)[Created]$(RESET)\n" "training/graphs"
	@mkdir -p training/models && printf "  $(GREEN)✓$(RESET) $(WHITE)%-50s$(RESET) $(GREEN)[Created]$(RESET)\n" "training/models"
	@mkdir -p training/models_saved && printf "  $(GREEN)✓$(RESET) $(WHITE)%-50s$(RESET) $(GREEN)[Created]$(RESET)\n" "training/models_saved"
	@mkdir -p training/models_saved_double && printf "  $(GREEN)✓$(RESET) $(WHITE)%-50s$(RESET) $(GREEN)[Created]$(RESET)\n" "training/models_saved_double"
	@mkdir -p training/models_saved_triple && printf "  $(GREEN)✓$(RESET) $(WHITE)%-50s$(RESET) $(GREEN)[Created]$(RESET)\n" "training/models_saved_triple"
	@mkdir -p training/models_z_old && printf "  $(GREEN)✓$(RESET) $(WHITE)%-50s$(RESET) $(GREEN)[Created]$(RESET)\n" "training/models_z_old"
	@mkdir -p training/models_zz_discarded && printf "  $(GREEN)✓$(RESET) $(WHITE)%-50s$(RESET) $(GREEN)[Created]$(RESET)\n" "training/models_zz_discarded"
	@printf "\n$(GREEN)✓ All directories created successfully!$(RESET)\n"

generate-dot-env: ## Create .env from template if it does not exist
	@if [ ! -f .env ]; then \
		if [ -f .env.template ]; then \
			cp .env.template .env; \
			printf "$(GREEN)✓ Created .env file from template$(RESET)\n"; \
			printf "$(YELLOW)⚠ Please edit .env file with your configuration$(RESET)\n"; \
		else \
			printf "$(RED)✗ .env.template not found$(RESET)\n"; \
			exit 1; \
		fi \
	else \
		printf "$(YELLOW)⚠ .env file already exists$(RESET)\n"; \
	fi

# Development targets
lint: ## Run code linting
	@printf "$(BLUE)Running code linting...$(RESET)\n"
	$(POETRY) run flake8 $(WORKDIR)/ --count --select=E9,F63,F7,F82 --show-source --statistics
	$(POETRY) run flake8 $(WORKDIR)/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	@printf "$(GREEN)✓ Linting completed!$(RESET)\n"

format: ## Format code with black and isort
	@printf "$(BLUE)Formatting code...$(RESET)\n"
	$(POETRY) run black $(WORKDIR)/
	$(POETRY) run isort $(WORKDIR)/
	@printf "$(GREEN)✓ Code formatting completed!$(RESET)\n"

test: ## Run tests
	@printf "$(BLUE)Running tests...$(RESET)\n"
	$(POETRY) run pytest tests/ -v
	@printf "$(GREEN)✓ Tests completed!$(RESET)\n"

check: lint test ## Run all code quality checks

check-venv: ## Verify virtual environment setup and command consistency
	@printf "$(BLUE)╔═══════════════════════════════════════════════════════════════╗$(RESET)\n"
	@printf "$(BLUE)║               Virtual Environment Verification                ║$(RESET)\n"
	@printf "$(BLUE)╚═══════════════════════════════════════════════════════════════╝$(RESET)\n"
	@printf "  $(WHITE)%-35s$(RESET) " "Virtual environment:"; [ -d .venv ] && printf "$(GREEN)✓ Exists$(RESET)\n" || printf "$(RED)✗ Missing$(RESET)\n"
	@printf "  $(WHITE)%-35s$(RESET) " "Python executable:"; [ -f $(PYTHON_VENV) ] && printf "$(GREEN)✓ Available$(RESET)\n" || printf "$(RED)✗ Missing$(RESET)\n"
	@if [ -f $(PYTHON_VENV) ]; then \
		python_version=$$($(PYTHON_VENV) --version 2>/dev/null); \
		printf "  $(WHITE)%-35s$(RESET) $(GREEN)%s$(RESET)\n" "Python version:" "$$python_version"; \
	fi
	@printf "  $(WHITE)%-35s$(RESET) " "Poetry in venv:"; $(PYTHON_VENV) -c "import poetry" 2>/dev/null && printf "$(GREEN)✓ Available$(RESET)\n" || printf "$(RED)✗ Missing$(RESET)\n"
	@printf "  $(WHITE)%-35s$(RESET) " "TensorBoard in venv:"; $(PYTHON_VENV) -c "import tensorboard" 2>/dev/null && printf "$(GREEN)✓ Available$(RESET)\n" || printf "$(RED)✗ Missing$(RESET)\n"
	@printf "\n"
	@printf "$(YELLOW)Command verification:$(RESET)\n"
	@printf "  $(WHITE)✓ All Python commands use:$(RESET) $(PYTHON_VENV)\n"
	@printf "  $(WHITE)✓ All Poetry commands use:$(RESET) $(PYTHON_VENV) -m poetry\n"
	@printf "  $(WHITE)✓ TensorBoard uses virtual environment\n"
	@printf "\n$(GREEN)✓ Virtual environment verification completed!$(RESET)\n"

verify-venv: ## Run detailed virtual environment verification script
	@printf "$(BLUE)Running detailed virtual environment verification...$(RESET)\n"
	bash ./others/verify_venv.sh
