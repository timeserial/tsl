# TSL / TS-1 — testes, builds e deploy.
# `make help` lista tudo. Convenções: python do .venv; alvos idempotentes;
# nada aqui toca nos protocolos pré-registados (experiments/luta*).

PY      := .venv/bin/python
PIP     := .venv/bin/pip
CC      := cc
CFLAGS  := -std=c99 -O2
FQBN    := esp32:esp32:esp32
PORT    ?= /dev/cu.usbserial-0001   # make demo-flash PORT=/dev/cu.XXXX
SCRATCH := $(shell mktemp -d)

.DEFAULT_GOAL := help

# ---------------------------------------------------------------- ambiente
.PHONY: venv
venv: ## create .venv and install dependencies
	python3 -m venv .venv
	$(PIP) install -q numpy torch pytest matplotlib

# ---------------------------------------------------------------- testes
.PHONY: test
test: ## Python unit tests (143)
	$(PY) -m pytest -q

.PHONY: test-c
test-c: ## C contracts: float (0.0%) and fixed-point (-0.1%)
	$(CC) $(CFLAGS) c/twostroke.c -lm -Ic -o $(SCRATCH)/ts_float
	$(SCRATCH)/ts_float
	$(CC) $(CFLAGS) c/twostroke_fixed.c -Ic -o $(SCRATCH)/ts_fixed
	$(SCRATCH)/ts_fixed

.PHONY: test-all
test-all: test test-c ## everything fast that must always pass

# ---------------------------------------------------------------- reprodução
.PHONY: repro
repro: ## reproduce the synthetic milestone 0.579±0.004 (4 seeds, ~2 min)
	$(PY) -u experiments/profundidade_empilhamento.py --arm brick \
	  --seeds 0 1 2 3 --epochs 80

.PHONY: crossbar
crossbar: ## estudo físico de crossbar completo (82 treinos, ~10 min)
	$(PY) -u scripts/crossbar_study.py

.PHONY: golden
golden: ## regenera os cabeçalhos golden do C a partir do Python
	$(PY) scripts/gen_golden_c.py

# ---------------------------------------------------------------- paper
.PHONY: figures
figures: ## figuras F1-F3 do paper (PDF vetorial + PNG)
	$(PY) paper/figures/gen_figures.py

.PHONY: paper
paper: ## compila paper/paper.pdf (figuras PNG embutidas)
	sed 's/figures\/fig\([123]\)_\([a-z]*\)\.pdf/figures\/fig\1_\2.png/' \
	  paper/paper.md > $(SCRATCH)/paper_png.md
	pandoc $(SCRATCH)/paper_png.md -o paper/paper.pdf \
	  --pdf-engine=weasyprint --resource-path=paper --css=paper/paper.css \
	  -V margin-top=2.2cm -V margin-bottom=2.2cm \
	  -V margin-left=2.4cm -V margin-right=2.4cm
	@ls -la paper/paper.pdf

.PHONY: patente
patente: ## PDFs das patentes (nº 1 e nº 2, PT e EN)
	for f in descricao_invencao descricao_invencao_2 \
	         invention_disclosure_en invention_disclosure_2_en; do \
	  pandoc patente/$$f.md -o patente/$$f.pdf \
	    --pdf-engine=weasyprint --css=paper/paper.css \
	    -V margin-top=2.2cm -V margin-bottom=2.2cm \
	    -V margin-left=2.4cm -V margin-right=2.4cm; \
	done
	@ls -la patente/*.pdf

# ---------------------------------------------------------------- demo ESP32
.PHONY: demo-build
demo-build: ## compila o firmware (variante LDR/analógica)
	arduino-cli compile --fqbn $(FQBN) demo/esp32_surpresa

.PHONY: demo-build-mpu
demo-build-mpu: ## compila a variante MPU-6050
	@mkdir -p $(SCRATCH)/esp32_mpu
	sed 's|^\#define SENSOR_ANALOG 1|//\#define SENSOR_ANALOG 1|; \
	     s|^//\#define SENSOR_MPU6050 1|\#define SENSOR_MPU6050 1|' \
	  demo/esp32_surpresa/esp32_surpresa.ino > $(SCRATCH)/esp32_mpu/esp32_mpu.ino
	arduino-cli compile --fqbn $(FQBN) $(SCRATCH)/esp32_mpu

.PHONY: demo-flash
demo-flash: demo-build ## compila e grava no ESP32 (PORT=/dev/cu.XXXX)
	arduino-cli upload --fqbn $(FQBN) -p $(PORT) demo/esp32_surpresa

.PHONY: demo-monitor
demo-monitor: ## monitor série da demo (Ctrl+C para sair)
	arduino-cli monitor -p $(PORT) --config baudrate=115200

# ---------------------------------------------------------------- release
.PHONY: arxiv-check
arxiv-check: test-all figures paper ## tudo o que tem de estar verde antes de submeter
	@echo ""
	@echo "== verificações de submissão =="
	@grep -q "RASCUNHO" patente/descricao_invencao_2.md && \
	  echo "AVISO: provisório nº 2 ainda em RASCUNHO - depositar ANTES do arXiv" || true
	@grep -c "±" paper/paper.md >/dev/null && echo "paper: OK"
	@echo "lembrete: repo público + licença + endorsement (ver conversa)"

.PHONY: mirror
mirror: ## regenerate ../nn-public (history WITHOUT patente/ or paper/)
	rm -rf ../nn-public
	git clone -q . ../nn-public
	cd ../nn-public && $(CURDIR)/.venv/bin/git-filter-repo \
	  --invert-paths --path patente --path paper \
	  --path estudo --path infra \
	  --path demo/lista_compras.md --path demo/lista_compras.pdf --force
	@echo "espelho em ../nn-public - verificar hashes antes de publicar:"
	@cd ../nn-public && for h in b69c8f2 32efefe cc25ba9 a42f5b7; do \
	  git cat-file -t $$h >/dev/null 2>&1 && echo "  $$h OK" || echo "  $$h FALTA"; done

.PHONY: clean
clean: ## remove temporary build artifacts
	rm -rf $(SCRATCH) demo/esp32_surpresa/build .pytest_cache
	find . -name __pycache__ -type d -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true

.PHONY: help
help: ## this list
	@grep -E '^[a-zA-Z_-]+:.*?## ' Makefile | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
