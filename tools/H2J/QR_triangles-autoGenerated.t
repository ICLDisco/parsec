! 0 upper
! 1 lower

integer BB, k, m, n
real A(BB,BB,2), T(BB, BB)
real v0, v1, v2, v3, v4, v5, v6, v7, v8, v9
real v10, v11, v12, v13, v14, v15, v16, v17, v18, v19
real v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30
real v100, v101, v102, v200, v201, v202

for ii=0 to BB-1 do
    for jj=0 to BB-1 do
!!    IN()
      A(ii,jj,0) = v100
      A(ii,jj,1) = v101
      T(ii,jj) = v102
    endfor
endfor

for k=0 to BB-1 do
!   dgeqrt( INOUT:A[k][k], OUT:T[k][k]);
!!  dgeqrt( R1:A(k, k):LU, T:T(k, k));
    v1 = A(k, k, 0)
    v2 = A(k, k, 1)
    A(k, k, 0) = v3
    A(k, k, 1) = v4
    T(k, k) = v5

    for m=k+1 to BB-1 do
!       dtsqrt(INOUT:A[k][k]:UP, INOUT:A[m][k], OUT:T[m][k]);
!!      dtsqrt(R2:A(k, k):U, V2:A(m, k):LU, T:T(m, k));
        v6 = A(k, k, 0)
        v7 = A(m, k, 0)
        v8 = A(m, k, 1)
        A(k, k, 0) = v10
        A(m, k, 0) = v11
        A(m, k, 1) = v12
        T(m, k) = v13
    endfor

    for n=k+1 to BB-1 do
!       dlarfb(IN:A[k][k]:LO, IN:T[k][k], INOUT:A[k][n]);
!!      dlarfb(V1:A(k, k):L, T:T(k, k), C1:A(k, n):LU);
        v14 = A(k, k, 1)
        v15 = T(k, k)
        v16 = A(k, n, 0)
        v17 = A(k, n, 1)
        A(k, n, 0) = v18
        A(k, n, 1) = v19

        for m=k+1 to BB-1 do
!            dssrfb(IN:A[m][k], IN:T[m][k], INOUT:A[k][n], INOUT:A[m][n]);
!!           dssrfb(V3:A(m, k):LU, T:T(m, k), C1:A(k, n):LU, C2:A(m, n):LU);
             v20 = A(m, k, 0)
             v21 = A(m, k, 1)
             v22 = T(m, k)
             v23 = A(k, n, 0)
             v24 = A(k, n, 1)
             v25 = A(m, n, 0)
             v26 = A(m, n, 1)
             A(k, n, 0) = v27
             A(k, n, 1) = v28
             A(m, n, 0) = v29
             A(m, n, 1) = v30
        endfor
    endfor
endfor


for ii=0 to BB-1 do
    for jj=0 to BB-1 do
!!    OUT()      
      v200 += A(ii,jj,0)
      v201 += A(ii,jj,1)
      v202 += T(ii,jj)
    endfor
endfor
