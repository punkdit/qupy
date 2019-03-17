
test: metro.so
#	./css.py


metro.c: metro.pyx
	cython3 metro.pyx 

INCLUDE = -I/usr/include/python3.5 -I/usr/lib/python3/dist-packages/numpy/core/include/ \
  -I/usr/include/python3.6 -I/usr/lib/python3/dist-packages/numpy/core/include/


metro.so: metro.c
	gcc -g -shared -fPIC -o metro.so metro.c $(INCLUDE) 

fast: metro.c
	gcc -g -O3 -shared -fPIC -o metro.so metro.c $(INCLUDE) 
#	./css.py

