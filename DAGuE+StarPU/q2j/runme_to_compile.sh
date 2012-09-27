make -j;

/usr/bin/gcc -O2 -g -c src/starpu_struct.c -o src/starpu_struct.o;

/usr/bin/c++   -O2 -g    CMakeFiles/q2j.dir/src/omega_interface.cpp.o CMakeFiles/q2j.dir/src/driver.c.o CMakeFiles/q2j.dir/src/utility.c.o CMakeFiles/q2j.dir/src/symtab.c.o CMakeFiles/q2j.dir/q2j.y.c.o CMakeFiles/q2j.dir/q2j.l.c.o src/starpu_struct.o -o q2j -rdynamic -lm /opt/Omega//omega_lib/obj/libomega.a;

rm -f core.*