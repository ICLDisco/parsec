# Makefile for PaRSEC4Julia (lives inside parsec/julia/)

.PHONY: help build-parsec wrappers clean

help:
	@echo "Available targets:"
	@echo "  build-parsec  Build parent PaRSEC + Julia wrappers (julia build_parsec4julia.jl)"
	@echo "  wrappers      Rebuild C wrappers only (requires ../build/install)"
	@echo "  clean         Remove generated .so files and parsec_env.sh"

build-parsec:
	julia build_parsec4julia.jl

wrappers:
	julia build_parsec4julia.jl --skip-parsec

clean:
	rm -f src/libdtd_wrapper.so src/libstencil_jl.so parsec_env.sh
