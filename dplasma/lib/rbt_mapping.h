#ifndef _RBT_MAPPING_H_
#define _RBT_MAPPING_H_

typedef struct{
    int N;
    int nb;
    int L;
    int ib;
    int jb;
} input_t;


typedef struct{
  int x;
  int y;
} seg_count_t;

typedef struct{
  int x1;
  int x2;
  int y1;
  int y2;
} seg_size_t;

typedef struct{
  seg_count_t t_cnt, b_cnt, l_cnt, r_cnt, c_cnt;
  seg_size_t  t_sz,  b_sz,  l_sz,  r_sz,  c_sz;
  int c_seg_cnt;
  int tot_seg_cnt_x;
  int tot_seg_cnt_y;
  int spx, mpx, epx;
  int spy, mpy, epy;
} seg_info_t;

/* forward declarations */
void find_tile(seg_info_t seg, int mb, int nb, int i, int j);
seg_info_t calculate_constants(input_t args);


#endif
