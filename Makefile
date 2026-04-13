.PHONY: all standard openmp-mac openmp-windows clean

all: standard

standard:
	$(MAKE) -C Standard

openmp-mac:
	$(MAKE) -C OpenMP-Mac OPENMP_CFLAGS="$(OPENMP_CFLAGS)" OPENMP_LDFLAGS="$(OPENMP_LDFLAGS)"

openmp-windows:
	$(MAKE) -C OpenMP-Windows OPENMP_CFLAGS="$(OPENMP_CFLAGS)" OPENMP_LDFLAGS="$(OPENMP_LDFLAGS)"

clean:
	$(MAKE) -C Standard clean
	$(MAKE) -C OpenMP-Mac clean
	$(MAKE) -C OpenMP-Windows clean
