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
!!    task_dpotrf(a(k,k))
    v3 = a(k,k)
    a(k,k) = v4

    for m = k+1 to BB-1 do
!                    IN      INOUT
!!        task_dtrsm(a(k,k), a(m,k))
        v9  = a(k,k)
        v10 = a(m,k)
        a(m,k) = v11
    endfor

    for m = k+1 to BB-1 do
        for n = k+1 to m-1 do
!                        IN      IN      INOUT
!!            task_dgemm(a(k,n), a(m,k), a(m,n))
            v5 = a(k,n)
            v6 = a(m,n)
            v7 = a(m,n)
            a(m,n) = v8
        endfor
!                    IN      INOUT
!!        task_dsyrk(a(m,m), a(m,k))
        v0 = a(m,m)
        v1 = a(m,k)
        a(m,k) = v2
    endfor
endfor

for ii=0 to BB-1 do
    for jj=0 to BB-1 do
!!      OUT()    
        v200 = a(ii,jj)
    endfor
endfor

