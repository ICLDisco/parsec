integer BB, k, m, n
real A(BB,BB), IPIV(BB,BB), L(BB, BB)
real v0, v1, v2, v3, v4, v5, v6, v7, v8, v9
real v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20
real v100, v101, v102, v201, v202, v203

for ii=0 to BB-1 do
    for jj=0 to BB-1 do
      A(ii,jj) = v100
      IPIV(ii,jj) = v101
      L(ii,jj) = v102
    endfor
endfor

for k = 0 to BB-1 do
!           INOUT    OUT
!   DGETRF( A(k, k), IPIV(k, k));
    v1 = A(k,k)
    A(k,k) = v2
    IPIV(k,k) = v3

    for m = k+1 to BB-1 do
!               INOUT    INOUT    OUT      OUT
!       DTSTRF( A(k, k), A(m, k), L(m, k), IPIV(m, k));
        v4     = A(k,k)
        v6     = A(m,k)
        A(k,k) = v5
        A(m,k) = v7
        L(m,k)    = v8
        IPIV(m,k) = v9
    endfor

    for n = k+1 to BB-1 do
!               IN          IN       INOUT
!       DGESSM( IPIV(k, k), A(k, k), A(k, n));
        v10 = IPIV(k, k)
        v11 = A(k, k)
        v12 = A(k, n)
        A(k, n) = v13

        for m = k+1 to BB-1 do
!                   INOUT    INOUT    IN       IN       IN           
!           DSSSSM( A(k, n), A(m, n), L(m, k), A(m, k), IPIV(m, k));
            v14 = A(k, n)
            v16 = A(m,n)
            v18 = L(m, k)
            v19 = A(m, k)
            v20 = IPIV(m, k)
            A(k, n) = v15
            A(m, n) = v17
        endfor
    endfor
endfor

for ii=0 to BB-1 do
    for jj=0 to BB-1 do
      v201 += A(ii,jj)
      v202 += IPIV(ii,jj)
      v203 += L(ii,jj)
    endfor
endfor
