! Right-looking tile Cholesky

integer k,n,m,BB
real v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11, v100, v200
real a(BB,BB)

for ii=0 to BB-1 do
    for jj=0 to BB-1 do
!!      IN()    
        a(ii,jj) = v100
    endfor
endfor


for k = 0 to BB-1 do
!                 INOUT
!!    tile_spotrf(a(k,k));
    v0 = a(k,k)
    a(k,k) = v1
    for n = k+1 to BB-1 do
!                    IN      INOUT
!!        tile_strsm(a(k,k), a(n,k));
        v2 = a(k,k)
        v3 = a(n,k)
        a(n,k) = v4

!                    IN      INOUT
!!        tile_ssyrk(a(n,k), a(n,n));
        v5 = a(n,k)
        v6 = a(n,n)
        a(n,n) = v7
	endfor
    for m = k+2 to BB-1 do
        for n = k+1 to m-1 do
!                        IN      IN      INOUT
!!            tile_sgemm(a(n,k), a(m,k), a(m,n));
            v8  = a(n,k)
            v9  = a(m,k)
            v10 = a(m,n)
            a(m,n) = v11
	endfor
    endfor
endfor

for ii=0 to BB-1 do
    for jj=0 to BB-1 do
!!      OUT()    
        v200 = a(ii,jj)
    endfor
endfor
