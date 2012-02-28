#ifndef _RBT_MAPPING_H_
#define _RBT_MAPPING_H_
#include "include/data_distribution.h"
#include "data_dist/matrix/matrix.h"

typedef struct{
  int m;
  int n;
} seg_count_t;

typedef struct{
  int m1;
  int m2;
  int n1;
  int n2;
} seg_size_t;

typedef struct{
  seg_count_t t_cnt, b_cnt, l_cnt, r_cnt, c_cnt;
  seg_size_t  t_sz,  b_sz,  l_sz,  r_sz,  c_sz;
  int c_seg_cnt;
  int tot_seg_cnt_m;
  int tot_seg_cnt_n;
  int spm, mpm, epm;
  int spn, mpn, epn;
} seg_info_t;

typedef struct dague_seg_ddesc{
 dague_ddesc_t super;
 tiled_matrix_desc_t *A_org;
 seg_info_t seg_info;
}dague_seg_ddesc_t;

/* forward declarations */
seg_info_t dague_rbt_calculate_constants(int N, int nb, int L, int ib, int jb);
void segment_to_tile(dague_seg_ddesc_t *seg_ddesc, int m, int n, int *m_tile, int *n_tile, uintptr_t *offset);
int type_index_to_sizes(seg_info_t seg, int mb, int nb, int type_index, int *m_off, int *n_off, int *m_sz, int *n_sz);
//void find_tile(seg_info_t seg, int mb, int nb, int i, int j);
int segment_to_arena_index(dague_seg_ddesc_t but_ddesc, int m, int n);


#endif
