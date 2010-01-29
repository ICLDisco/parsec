integer TILES, k, m, n
real A(TILES,TILES), T(TILES,TILES)
real mgc0, mgc1, mgc2, mgc3, mgc4, mgc5, mgc6, mgc7, mgc8, mgc9, mgc10

for k = 0 to TILES-1 do
    !        INOUT    OUT
    !        RV1      T
    ! dgeqrt(A[k][k], T[k][k])
    mgc1 = A(k,k)
    A(k,k) = mgc0
    T(k,k) = mgc0

    for m = k+1 to TILES-1 do
        !        INOUT    INOUT    OUT
        !        R        V2       T
        ! dtsqrt(A[k][k], A[m][k], T[m][k])
        mgc2 = A(k,k)
        A(k,k) = mgc0
        mgc3 = A(m,k)
        A(m,k) = mgc0
        T(m,k) = mgc0
    endfor

    for n=k+1 to TILES-1 do
        !        IN       IN       INOUT
        !        V1       T        C1
        ! dlarfb(A[k][k], T[k][k], A[k][n])
        mgc4 = A(k,k)
        mgc5 = T(k,k)
        mgc6 = A(k,n)
        A(k,n) = mgc0

        for m=k+1 to TILES-1 do
            !        IN       IN       INOUT    INOUT
            !        V2       T        C1       C2
            ! dssrfb(A[m][k], T[m][k], A[k][n], A[m][n]
            mgc7 = A(m,k)
            mgc8 = T(m,k)
            mgc9 = A(k,n)
            A(k,n) = mgc0
            mgc10 = A(m,n)
            A(m,n) = mgc0
        endfor
    endfor
endfor
