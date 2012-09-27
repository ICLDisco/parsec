#include "starpu_struct.h"
#include <stdio.h>
#include <stdlib.h>

void print_param_list(StarPU_param_list *l)
{
    StarPU_param_list *tmp;
    for(tmp = l; tmp != NULL; tmp = tmp->next)
    {
	switch (tmp->p->type) 
	{
	case CODELET_MODE  : 
	    fprintf(stderr, "modes : %s\n", tmp->p->p.modes);
	    break;
	case CODELET_WHERE :
	    fprintf(stderr, "where : %s\n", tmp->p->p.where);
	    break;
	case CODELET_NBUFF :
	    fprintf(stderr, "nbuffers : %" PRIu64 "\n", tmp->p->p.nbuffers);
	    break;
	case CODELET_CPU   :
	    fprintf(stderr, "cpu :     %s\n", (tmp->p->p.l)->name);
	    break;
	case CODELET_CUDA  :
	    fprintf(stderr, "cuda :    %s\n", (tmp->p->p.l)->name);
	    break;
	default            :
	    fprintf(stderr, "bad stuff\n");
	}
    }
}

void print_codelet(StarPU_codelet *cl)
{
    fprintf(stderr, "codelet %s  :\n", cl->name);
    fprintf(stderr, "    Options :\n");
    print_param_list(cl->l);
    fprintf(stderr, "-------------\n");
}

void print_codelet_list(StarPU_codelet_list *l)
{
    StarPU_codelet_list *tmp;
    for(tmp = l; tmp != NULL; tmp = tmp->prev)
    {
	print_codelet(tmp->cl);
    }
}
