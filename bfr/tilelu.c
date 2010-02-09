
void tile_el_you_sequential(double *A, double *L, int *IPIV, int IB, int NB, int BB)
{
    int NBNBSIZE = NB*NB;
    int IBNBSIZE = NB*IB;
    int MT = BB;

    int k, m, n;
    int IINFO;

    for (k = 0; k < BB; k++)
    {
        CORE_dgetrf(
            NB, NB, IB,
            A(k, k), NB,
            IPIV(k, k), &IINFO);

        for (n = k+1; n < BB; n++)
            CORE_dgessm(
                NB, NB, NB, IB,
                IPIV(k, k),
                A(k, k), NB,
                A(k, n), NB);

        for (m = k+1; m < BB; m++)
        {
            CORE_dtstrf(
                NB, NB, IB, NB,
                A(k, k), NB,
                A(m, k), NB,
                L(m, k), IB,
                IPIV(m, k), &IINFO);

            for (n = k+1; n < BB; n++)
                CORE_dssssm(
                    NB, NB, NB, IB, NB,
                    A(k, n), NB,
                    A(m, n), NB,
                    L(m, k), IB,
                    A(m, k), NB,
                    IPIV(m, k));
        }
    }
}
