#ifndef _flopsutils_h_
#define _flopsutils_h_

#define CLEAN_MB( _desc, _m )  (  ( (_m) == ((_desc)->mt-1)) ? ( (_desc)->m - (_m)*(_desc)->mb ) : (_desc)->mb )
#define CLEAN_NB( _desc, _n )  (  ( (_n) == ((_desc)->nt-1)) ? ( (_desc)->n - (_n)*(_desc)->nb ) : (_desc)->nb )

#endif
