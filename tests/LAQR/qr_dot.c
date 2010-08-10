#include <stdio.h>
#include <assert.h>

int main(int argc, char **argv)
{
   assert(argc == 2);    

   int m, n, k;
   int BB = atoi(argv[1]);


   printf("digraph G {\n");
/*
   for (k = 0; k < BB; k++)
   {
       printf("subgraph cluster%d {\n", k);

       for (n = 0; n < k; n++)
           printf("\tSYRK_%d_%d;\n", k, n);

       printf("\tPOTRF_%d;\n", k);

       for (m = k+1; m < BB; m++)
       {
           for (n = 0; n < k; n++)
               printf("\tGEMM_%d_%d_%d;\n", k, m, n);

           printf("\tTRSM_%d_%d;\n", k, m);
       }
       printf("}\n");
       printf("\n");
   }
*/


   printf("\tDGEQRT_0[style=filled,fillcolor=\"#4488AA\",fontcolor=\"white\",label=\"DGEQRT_0\"];\n");


   for (k = 0; k < BB; k++)
   {

       // DGEQRT
       if (k+1 < BB)
       {
           printf("\tDTSQRT_%d_%d [style=filled,fillcolor=\"#CC99EE\",fontcolor=\"black\",label=\"DTSQRT_%d_%d\"];\n",
                  k, k+1, k, k+1);
           printf("\t DGEQRT_%d -> DTSQRT_%d_%d;\n", k, k, k+1);
       }
       for (n = k+1; n < BB; n++)
       {
           printf("\tDLARFB_%d_%d [style=filled,fillcolor=\"#99CCFF\",fontcolor=\"black\",label=\"DLARFB_%d_%d\"];\n",
                  k, n, k, n);
           printf("\t DGEQRT_%d -> DLARFB_%d_%d;\n", k, k, n);
       }


       for (m = k+1; m < BB; m++)
       {

           // DTSQRT
           if (m+1 < BB)
           {
               printf("\tDTSQRT_%d_%d [style=filled,fillcolor=\"#CC99EE\",fontcolor=\"black\",label=\"DTSQRT_%d_%d\"];\n",
                      k, m+1, k, m+1);
               printf("\t DTSQRT_%d_%d -> DTSQRT_%d_%d;\n", k, m, k, m+1);
           }
           for (n = k+1; n < BB; n++)
           {
               printf("\tDSSRFB_%d_%d_%d [style=filled,fillcolor=\"#CCFF00\",fontcolor=\"black\",label=\"DSSRFB_%d_%d_%d\"];\n",
                      k, m, n, k, m, n);
               printf("\t DTSQRT_%d_%d -> DSSRFB_%d_%d_%d;\n", k, m, k, m, n);
           }
       }


       for (n = k+1; n < BB; n++)
       {

           // DLARFB
           printf("\tDSSRFB_%d_%d_%d [style=filled,fillcolor=\"#CCFF00\",fontcolor=\"black\",label=\"DSSRFB_%d_%d_%d\"];\n",
                  k, k+1, n, k, k+1, n);
           printf("\t DLARFB_%d_%d -> DSSRFB_%d_%d_%d;\n", k, n, k, k+1, n);


           for (m = k+1; m < BB; m++)
           {

               // DSSRFB
               if (m+1 < BB)
               {
                   printf("\tDSSRFB_%d_%d_%d [style=filled,fillcolor=\"#CCFF00\",fontcolor=\"black\",label=\"DSSRFB_%d_%d_%d\"];\n",
                           k, m+1, n, k, m+1, n);
                   printf("\t DSSRFB_%d_%d_%d -> DSSRFB_%d_%d_%d;\n", k, m, n, k, m+1, n);
               }
               if (k+1 < BB)
               {
                   if (m == k+1 && n == k+1)
                   {
                       printf("\tDGEQRT_%d [style=filled,fillcolor=\"#4488AA\",fontcolor=\"white\",label=\"DGEQRT_%d\"];\n", k+1, k+1);
                       printf("\t DSSRFB_%d_%d_%d -> DGEQRT_%d;\n", k, m, n, k+1);
                       continue;
                   }
                   if (m == k+1)
                   {
                       printf("\tDLARFB_%d_%d [style=filled,fillcolor=\"#99CCFF\",fontcolor=\"black\",label=\"DLARFB_%d_%d\"];\n", k+1, n, k+1, n);
                       printf("\t DSSRFB_%d_%d_%d -> DLARFB_%d_%d;\n", k, m, n, k+1, n);
                       continue;
                   }
                   if (n == k+1)
                   {
                       printf("\tDTSQRT_%d_%d [style=filled,fillcolor=\"#CC99EE\",fontcolor=\"black\",label=\"DTSQRT_%d_%d\"];\n", k+1, m, k+1, m);
                       printf("\t DSSRFB_%d_%d_%d -> DTSQRT_%d_%d;\n", k, m, n, k+1, m);
                       continue;
                   }
                   printf("\tDSSRFB_%d_%d_%d [style=filled,fillcolor=\"#CCFF00\",fontcolor=\"black\",label=\"DSSRFB_%d_%d_%d\"];\n", k+1, m, n, k+1, m, n);
                   printf("\t DSSRFB_%d_%d_%d -> DSSRFB_%d_%d_%d;\n", k, m, n, k+1, m, n);
               }
           }

       }

   }
   printf("}\n");

   return 1;
}


/*
printf("\tnode [style=filled,color=\"#9966CC0\",label=\"\"];\n");
printf("\tnode [style=filled,color=\"#CCFF00\",label=\"\"];\n");
*/



