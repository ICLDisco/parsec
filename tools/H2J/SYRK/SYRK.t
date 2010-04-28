! 0 upper
! 1 lower

integer BB, k, m, n
real A(BB,BB), B(BB,BB)
real v0, v1, v2, v3, v4, v5, v6
real v100, v101, v200, v201


for ii=0 to BB-1 do
    for jj=0 to BB-1 do
!!    IN()
      A(ii,jj) = v100
      B(ii,jj) = v101
    endfor
endfor

for m = 0 to BB-1 do
    for n = 0 to m-1 do
        for k = 0 to BB-1 do
!                        INOUT      IN         IN
!!          task_GEMM( B:B(m,n), A1:A(m,k), A2:A(m,k) )
            v0 = B(m,n)
            v1 = A(m,k)
            v2 = A(n,k)
            B(m,n) = v3
        endfor
    endfor
    for k = 0 to BB-1 do
!                    INOUT     IN
!!      task_SYRK( B:B(m,m), A:A(m,k) )
        v4 = A(m,k)
        v5 = B(m,m)
        B(m,m) = v6
    endfor
endfor

for ii=0 to BB-1 do
    for jj=0 to BB-1 do
!!    OUT()      
      v200 += A(ii,jj)
      v201 += B(ii,jj)
    endfor
endfor
