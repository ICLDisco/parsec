! 0 upper
! 1 lower

integer BB, k, i, j
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

for  i = 0 to BB-1 do
    for  j = 0 to i-1 do
        for  k = 0 to BB-1 do
!                        INOUT      IN         IN
!!          task_GEMM( B:B(i,j), A1:A(i,k), A2:A(j,k) )
            v0 = B(i,j)
            v1 = A(i,k)
            v2 = A(j,k)
            B(i,j) = v3
        endfor
    endfor
    for  k = 0 to BB-1 do
!                    INOUT     IN
!!      task_SYRK( B:B(i,i), A:A(i,k) )
        v4 = A(i,k)
        v5 = B(i,i)
        B(i,i) = v6
    endfor
endfor

for ii=0 to BB-1 do
    for jj=0 to BB-1 do
!!    OUT()      
      v200 += A(ii,jj)
      v201 += B(ii,jj)
    endfor
endfor
