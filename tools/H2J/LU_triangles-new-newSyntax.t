! 0 upper
! 1 lower

integer BB, k, m, n
real A(BB,BB,2), IPIV(BB,BB), L(BB, BB)
real v0, v1, v2, v3, v4, v5, v6, v7, v8, v9
real v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20
real v82, v83, v84, v85, v86, v87, v89, v91, v92, v96, v97
real v100, v101, v102, v201, v202, v203


for ii=0 to BB-1 do
    for jj=0 to BB-1 do
!!    IN()
      A(ii,jj,0) = v100
      A(ii,jj,1) = v100
      IPIV(ii,jj) = v101
      L(ii,jj) = v102
    endfor
endfor

for k = 0 to BB-1 do
!           INOUT         OUT
!!  DGETRF( A:A(k, k):LU, IPIV:IPIV(k, k));
    v1  = A(k,k,0)
    v91 = A(k,k,1)
    A(k,k,0) = v92
    A(k,k,1) = v2
    IPIV(k,k) = v3

    for n = k+1 to BB-1 do
!               IN               IN           INOUT
!!      DGESSM( IPIV:IPIV(k, k), L:A(k, k):L, C1:A(k, n):LU);
        v10 = IPIV(k, k)
        v11 = A(k, k, 1)
        v12 = A(k, n, 0)
        v82 = A(k, n, 1)
        A(k, n, 0) = v13
        A(k, n, 1) = v83
    endfor

    for m = k+1 to BB-1 do
!               INOUT        INOUT         OUT         OUT
!!      DTSTRF( U:A(k, k):U, L:A(m, k):LU, dL:L(m, k), IPIV:IPIV(m, k));
        v4  = A(k,k,0)
        v6  = A(m,k,0)
        v96 = A(m,k,1)
        A(k,k,0)  = v5
        A(m,k,0)  = v7
        A(m,k,1)  = v97
        L(m,k)    = v8
        IPIV(m,k) = v9

        for n = k+1 to BB-1 do
!                   INOUT          INOUT          IN          IN            IN           
!!          DSSSSM( C1:A(k, n):LU, C2:A(m, n):LU, dL:L(m, k), L:A(m, k):LU, IPIV:IPIV(m, k));
            v14 = A(k, n, 0)
            v84 = A(k, n, 1)
            v16 = A(m, n, 0)
            v86 = A(m, n, 1)
            v18 = L(m, k)
            v19 = A(m, k, 0)
            v89 = A(m, k, 1)
            v20 = IPIV(m, k)
            A(k, n, 0) = v15
            A(k, n, 1) = v85
            A(m, n, 0) = v17
            A(m, n, 1) = v87
        endfor
    endfor
endfor

for ii=0 to BB-1 do
    for jj=0 to BB-1 do
!!    OUT()      
      v201 += A(ii,jj,0)
      v201 += A(ii,jj,1)
      v202 += IPIV(ii,jj)
      v203 += L(ii,jj)
    endfor
endfor
