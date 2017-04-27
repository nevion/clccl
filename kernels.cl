//The MIT License (MIT)
//
//Copyright (c) 2015 Jason Newton <nevion@gmail.com>
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.


#define PREFER_MCL_WORKGROUP_FUNCTIONS
#if __OPENCL_VERSION__ >= 200 && defined(PREFER_MCL_WORKGROUP_FUNCTIONS)
#define USE_CL2_WORKGROUP_FUNCTIONS
#endif

#include "clcommons/common.h"
#include "clcommons/image.h"
#include "clcommons/work_group.h"

#ifndef PIXELT
#define PIXELT uint
#endif
#ifndef LABELT
#define LABELT uint
#endif
#ifndef CONNECTIVITYPIXELT
#define CONNECTIVITYPIXELT uint
#endif
#ifndef LDSPIXELT
#define LDSPIXELT uint
#endif
#ifndef LDSLABELT
#define LDSLABELT uint
#endif
#ifndef LDSCONNECTIVITYPIXELT
#define LDSCONNECTIVITYPIXELT uint
#endif

typedef PIXELT PixelT;
typedef LDSPIXELT LDSPixelT;
typedef LABELT LabelT;
typedef LDSLABELT LDSLabelT;
typedef CONNECTIVITYPIXELT ConnectivityPixelT;
typedef LDSCONNECTIVITYPIXELT LDSConnectivityPixelT;

#ifndef BG_VALUE
#define BG_VALUE 0
#endif

#ifndef CONNECTIVITY  //4 or 8
#define CONNECTIVITY 8
#endif

#ifndef DYNAMIC_IMGDIMS
#define im_rows IMG_ROWS
#define im_cols IMG_COLS
#endif

#ifndef WORKGROUP_TILE_SIZE_X
#define WORKGROUP_TILE_SIZE_X 32
#endif
#ifndef WORKGROUP_TILE_SIZE_Y
#define WORKGROUP_TILE_SIZE_Y 2
#endif

#ifndef WORKITEM_REPEAT_X
#define WORKITEM_REPEAT_X 1
#endif
#ifndef WORKITEM_REPEAT_Y
#define WORKITEM_REPEAT_Y 16
#endif

//work TILEs
#define TILE_COLS (WORKGROUP_TILE_SIZE_X * WORKITEM_REPEAT_X)
#define TILE_ROWS (WORKGROUP_TILE_SIZE_Y * WORKITEM_REPEAT_Y)

#ifndef FUSED_MARK_KERNEL
#define FUSED_MARK_KERNEL 0
#endif

__constant const uint UP = (1<<0);
__constant const uint LEFT = (1<<1);
__constant const uint DOWN = (1<<2);
__constant const uint RIGHT = (1<<3);
__constant const uint LEFT_UP = (1<<4);
__constant const uint LEFT_DOWN = (1<<5);
__constant const uint RIGHT_UP = (1<<6);
__constant const uint RIGHT_DOWN = (1<<7);

#define isConnected(p1, p2) ((p1) == (p2))

#ifdef NVIDIA_ARCH
#define pixel_at(type, basename, r, c) image_pixel_at(type, PASTE(basename, _p), im_rows, im_cols, PASTE(basename, _pitch), (r), (c))
#else
#define pixel_at(type, basename, r, c) image_pixel_at(type, PASTE2(basename, _p), im_rows, im_cols, PASTE2(basename, _pitch), (r), (c))
#endif

#define CONNECTIVITY_TILE_OUTPUT 0

#define apron_pixel(apron, _t_r, _t_c) apron[(_t_r+ 1)][(_t_c + 1)]
//global dimensions: divUp(im_cols, tile_cols), divUp(im_rows, tile_rows);
__attribute__((reqd_work_group_size(WORKGROUP_TILE_SIZE_X, WORKGROUP_TILE_SIZE_Y, 1)))
__kernel void make_connectivity_image(
#ifdef DYNAMIC_IMGDIMS
    uint im_rows, uint im_cols,
#endif
    __global const PixelT *image_p, uint image_pitch, __global ConnectivityPixelT *connectivityim_p, uint connectivityim_pitch
){
    const uint tile_col_blocksize = TILE_COLS;
    const uint tile_row_blocksize = TILE_ROWS;
    const uint tile_col_block = get_group_id(0) + get_global_offset(0) / get_local_size(0);
    const uint tile_row_block = get_group_id(1) + get_global_offset(1) / get_local_size(1);
    const uint tile_col = get_local_id(0);
    const uint tile_row = get_local_id(1);

    uint tile_rows = tile_row_blocksize;
    uint tile_cols = tile_col_blocksize;

    const uint tile_row_start = tile_row_block * tile_rows;
    const uint tile_col_start = tile_col_block * tile_cols;
    const uint tile_row_end = min(tile_row_start + tile_rows, (uint) im_rows);
    const uint tile_col_end = min(tile_col_start + tile_cols, (uint) im_cols);
    //adjust to true tile dimensions
    tile_rows = tile_row_end - tile_row_start;
    tile_cols = tile_col_end - tile_col_start;
    const uint apron_tile_cols = tile_cols + 2;;
    //const uint n_tile_pixels = tile_rows * tile_cols;
    const uint n_work_items = get_local_size(0) * get_local_size(1);
    const uint n_apron_tile_pixels = (tile_rows + 2) * (apron_tile_cols);
    __local LDSPixelT im_tile[TILE_ROWS + 2][TILE_COLS + 2];

#if CONNECTIVITY_TILE_OUTPUT
    __local LDSConnectivityPixelT connectivity_tile[TILE_ROWS][TILE_COLS];
#endif

    const uint tid = get_local_linear_id();
    for(uint im_tile_fill_task_id = tid; im_tile_fill_task_id < n_apron_tile_pixels; im_tile_fill_task_id += n_work_items){
        const uint im_apron_tile_row = im_tile_fill_task_id / apron_tile_cols;
        const uint im_apron_tile_col = im_tile_fill_task_id % apron_tile_cols;
        const int g_c = ((int)(im_apron_tile_col + tile_col_start)) - 1;
        const int g_r = ((int)(im_apron_tile_row + tile_row_start)) - 1;

        im_tile[im_apron_tile_row][im_apron_tile_col] = image_tex2D(PixelT, image_p, (int) im_rows, (int) im_cols, image_pitch, g_r, g_c, ADDRESS_ZERO);
    }
    lds_barrier();

    #pragma unroll
    for (int i = 0; i < WORKITEM_REPEAT_Y; ++i){
        #pragma unroll
        for (int j = 0; j < WORKITEM_REPEAT_X; ++j){
            const uint t_c = get_local_id(0) + WORKGROUP_TILE_SIZE_X * j;
            const uint t_r = get_local_id(1) + WORKGROUP_TILE_SIZE_Y * i;
            const uint c = t_c + tile_col_start;
            const uint r = t_r + tile_row_start;
            PixelT pixel = apron_pixel(im_tile, t_r, t_c);
            ConnectivityPixelT connectivity = 0;

#if CONNECTIVITY == 8
            connectivity |= c > 0 && r > 0                         && isConnected(pixel, apron_pixel(im_tile, t_r-1, t_c - 1)) ? LEFT_UP : 0;
            connectivity |= c > 0                                  && isConnected(pixel, apron_pixel(im_tile, t_r  , t_c - 1)) ? LEFT : 0;
            connectivity |= c > 0 && r < im_rows - 1               && isConnected(pixel, apron_pixel(im_tile, t_r+1, t_c - 1)) ? LEFT_DOWN : 0;
            connectivity |=          r < im_rows - 1               && isConnected(pixel, apron_pixel(im_tile, t_r+1, t_c    )) ? DOWN : 0;
            connectivity |= c < im_cols - 1 && r < im_rows - 1     && isConnected(pixel, apron_pixel(im_tile, t_r+1, t_c + 1)) ? RIGHT_DOWN : 0;
            connectivity |= c < im_cols - 1                        && isConnected(pixel, apron_pixel(im_tile, t_r  , t_c + 1)) ? RIGHT : 0;
            connectivity |= c < im_cols - 1 && r > 0               && isConnected(pixel, apron_pixel(im_tile, t_r-1, t_c + 1)) ? RIGHT_UP : 0;
            connectivity |=          r > 0                         && isConnected(pixel, apron_pixel(im_tile, t_r-1, t_c    )) ? UP : 0;
#else
            connectivity |= c > 0                                  && isConnected(pixel, apron_pixel(im_tile, t_r  , t_c - 1)) ? LEFT : 0;
            connectivity |=          r < im_rows - 1               && isConnected(pixel, apron_pixel(im_tile, t_r+1, t_c    )) ? DOWN : 0;
            connectivity |= c < im_cols - 1                        && isConnected(pixel, apron_pixel(im_tile, t_r  , t_c + 1)) ? RIGHT : 0;
            connectivity |=          r > 0                         && isConnected(pixel, apron_pixel(im_tile, t_r-1, t_c    )) ? UP : 0;
#endif
            connectivity = (c < im_cols) & (r < im_rows) ? connectivity : 0;
#if CONNECTIVITY_TILE_OUTPUT
            connectivity_tile[t_r][t_c] = connectivity;
#else
            if((c < im_cols) & (r < im_rows)){
                pixel_at(ConnectivityPixelT, connectivityim, r, c) = connectivity;
            }
#endif

        }
    }
#if CONNECTIVITY_TILE_OUTPUT
    lds_barrier();

    for(uint im_tile_fill_task_id = tid; im_tile_fill_task_id < n_tile_pixels; im_tile_fill_task_id += n_work_items){
        const uint im_tile_row = im_tile_fill_task_id / tile_cols;
        const uint im_tile_col = im_tile_fill_task_id % tile_cols;
        const uint g_c = im_tile_col + tile_col_start;
        const uint g_r = im_tile_row + tile_row_start;

        pixel_at(ConnectivityPixelT, connectivityim, g_r, g_c) = connectivity_tile[im_tile_row][im_tile_col];
    }
#endif
}

__attribute__((reqd_work_group_size(WORKGROUP_TILE_SIZE_X, WORKGROUP_TILE_SIZE_Y, 1)))
__kernel void label_tiles(
#ifdef DYNAMIC_IMGDIMS
    uint im_rows, uint im_cols,
#endif
    __global LabelT *labelim_p, uint labelim_pitch, __global const ConnectivityPixelT *connectivityim_p, uint connectivityim_pitch
){
    const uint tile_col_blocksize = TILE_COLS;
    const uint tile_row_blocksize = TILE_ROWS;
    const uint tile_col_block = get_group_id(0) + get_global_offset(0) / get_local_size(0);
    const uint tile_row_block = get_group_id(1) + get_global_offset(1) / get_local_size(1);

    uint tile_rows = tile_row_blocksize;
    uint tile_cols = tile_col_blocksize;

    const uint tile_row_start = tile_row_block * tile_rows;
    const uint tile_col_start = tile_col_block * tile_cols;
    const uint tile_row_end = min(tile_row_start + tile_rows, (uint) im_rows);
    const uint tile_col_end = min(tile_col_start + tile_cols, (uint) im_cols);
    //adjust to true tile dimensions
    tile_rows = tile_row_end - tile_row_start;
    tile_cols = tile_col_end - tile_col_start;

    __local LDSLabelT label_tile_im[TILE_ROWS][TILE_COLS];
#ifdef SHM_EDGE_TILE
    __local LDSConnectivityPixelT  edge_tile_im[TILE_ROWS][TILE_COLS];
#endif

    LDSLabelT new_labels[WORKITEM_REPEAT_Y][WORKITEM_REPEAT_X];
    LDSLabelT old_labels[WORKITEM_REPEAT_Y][WORKITEM_REPEAT_X];
#ifndef SHM_EDGE_TILE
    LDSConnectivityPixelT edges[WORKITEM_REPEAT_Y][WORKITEM_REPEAT_X];
#endif

    #pragma unroll
    for (int i = 0; i < WORKITEM_REPEAT_Y; ++i){
        #pragma unroll
        for (int j = 0; j < WORKITEM_REPEAT_X; ++j){
            const uint tile_row = get_local_id(1) + WORKGROUP_TILE_SIZE_Y * i;
            const uint tile_col = get_local_id(0) + WORKGROUP_TILE_SIZE_X * j;
            const bool valid_pixel_task = (tile_col < tile_cols) & (tile_row < tile_rows);
            ConnectivityPixelT c = valid_pixel_task ? pixel_at(ConnectivityPixelT, connectivityim, tile_row_start + tile_row, tile_col_start + tile_col) : 0;

            c = tile_col == 0 ? c & ~(LEFT|LEFT_DOWN|LEFT_UP) : c;
            c = tile_row == 0 ? c & ~(UP|LEFT_UP|RIGHT_UP) : c;

            c = tile_col >= tile_cols - 1 ? c & ~(RIGHT|RIGHT_DOWN|RIGHT_UP) : c;
            c = tile_row >= tile_rows - 1 ? c & ~(DOWN|LEFT_DOWN|RIGHT_DOWN) : c;

            new_labels[i][j] = valid_pixel_task ? tile_row * tile_cols + tile_col : ((LDSLabelT) -1);
#ifdef SHM_EDGE_TILE
            edge_tile_im[tile_row][tile_col] = c;
#else
            edges[i][j] = c;
#endif
        }
    }

    for (uint k = 0; ;++k){
        //make copies
        #pragma unroll
        for (int i = 0; i < WORKITEM_REPEAT_Y; ++i){
            #pragma unroll
            for (int j = 0; j < WORKITEM_REPEAT_X; ++j){
                const uint tile_row = get_local_id(1) + WORKGROUP_TILE_SIZE_Y * i;
                const uint tile_col = get_local_id(0) + WORKGROUP_TILE_SIZE_X * j;
                old_labels[i][j] = new_labels[i][j];
                label_tile_im[tile_row][tile_col] = new_labels[i][j];
            }
        }
        lds_barrier();

        //take minimum label of local neighboorhood - single writer, multi reader version
        #pragma unroll
        for (int i = 0; i < WORKITEM_REPEAT_Y; ++i){
            #pragma unroll
            for (int j = 0; j < WORKITEM_REPEAT_X; ++j){
                const uint tile_row = get_local_id(1) + WORKGROUP_TILE_SIZE_Y * i;
                const uint tile_col = get_local_id(0) + WORKGROUP_TILE_SIZE_X * j;

#ifdef SHM_EDGE_TILE
                const ConnectivityPixelT connectivity = edge_tile_im[tile_row][tile_col];
#else
                const ConnectivityPixelT connectivity = edges[i][j];
#endif
                LDSLabelT label = new_labels[i][j];

#if CONNECTIVITY == 8
                label = connectivity & UP            ? min(label, label_tile_im[tile_row - 1][tile_col - 0]) : label;
                label = connectivity & LEFT_UP       ? min(label, label_tile_im[tile_row - 1][tile_col - 1]) : label;
                label = connectivity & LEFT          ? min(label, label_tile_im[tile_row - 0][tile_col - 1]) : label;
                label = connectivity & LEFT_DOWN     ? min(label, label_tile_im[tile_row + 1][tile_col - 1]) : label;
                label = connectivity & DOWN          ? min(label, label_tile_im[tile_row + 1][tile_col - 0]) : label;
                label = connectivity & RIGHT_DOWN    ? min(label, label_tile_im[tile_row + 1][tile_col + 1]) : label;
                label = connectivity & RIGHT         ? min(label, label_tile_im[tile_row + 0][tile_col + 1]) : label;
                label = connectivity & RIGHT_UP      ? min(label, label_tile_im[tile_row - 1][tile_col + 1]) : label;
#else
                label = connectivity & UP            ? min(label, label_tile_im[tile_row - 1][tile_col - 0]) : label;
                label = connectivity & LEFT          ? min(label, label_tile_im[tile_row - 0][tile_col - 1]) : label;
                label = connectivity & DOWN          ? min(label, label_tile_im[tile_row + 1][tile_col - 0]) : label;
                label = connectivity & RIGHT         ? min(label, label_tile_im[tile_row + 0][tile_col + 1]) : label;
#endif

                new_labels[i][j] = label;
            }
        }
        lds_barrier();

        __local uint changed;
        if((get_local_id(1) == 0) & (get_local_id(0) == 0)){
            changed = 0;
        }
        lds_barrier();

        uint pchanged = 0;
        #pragma unroll
        for(int i = 0; i < WORKITEM_REPEAT_Y; ++i){
            #pragma unroll
            for(int j = 0; j < WORKITEM_REPEAT_X; ++j){
                const uint tile_row = get_local_id(1) + WORKGROUP_TILE_SIZE_Y * i;
                const uint tile_col = get_local_id(0) + WORKGROUP_TILE_SIZE_X * j;
                const uint old_label = old_labels[i][j];
                pchanged += atomic_min(&label_tile_im[old_label / tile_cols][old_label % tile_cols], new_labels[i][j]) > new_labels[i][j];
            }
        }
        atomic_add(&changed, pchanged);
        lds_barrier();

        //if there are no updates, we are finished
        if(!changed){
            break;
        }

        //Compact paths
        #pragma unroll
        for(int i = 0; i < WORKITEM_REPEAT_Y; ++i){
            #pragma unroll
            for(int j = 0; j < WORKITEM_REPEAT_X; ++j){
                const uint tile_row = get_local_id(1) + WORKGROUP_TILE_SIZE_Y * i;
                const uint tile_col = get_local_id(0) + WORKGROUP_TILE_SIZE_X * j;
                const bool valid_pixel_task = (tile_col < tile_cols) & (tile_row < tile_rows);
                if(valid_pixel_task){
                    LDSLabelT label = new_labels[i][j];
                    //find root label
                    while(label_tile_im[label / tile_cols][label % tile_cols] < label){
                        label = label_tile_im[label / tile_cols][label % tile_cols];
                    }

                    new_labels[i][j] = label;
                }
            }
        }
        lds_barrier();
    }

    //save with adjusted global (untiled) labels
    #pragma unroll
    for(int i = 0; i < WORKITEM_REPEAT_Y; ++i){
        #pragma unroll
        for(int j = 0; j < WORKITEM_REPEAT_X; ++j){
            const LabelT tile_label = new_labels[i][j];
            //convert the tile label into it's 2-D equivilent
            const uint l_g_r = (tile_label / tile_cols) + tile_row_start;
            const uint l_g_c = (tile_label % tile_cols) + tile_col_start;

            const uint tile_row = get_local_id(1) + WORKGROUP_TILE_SIZE_Y * i;
            const uint tile_col = get_local_id(0) + WORKGROUP_TILE_SIZE_X * j;
            const uint g_r = tile_row_start + tile_row;
            const uint g_c = tile_col_start + tile_col;

            //adjust to global offset and convert to scanline order again - this is globally unique
            const LabelT glabel = l_g_r * im_cols + l_g_c;
            const bool valid_pixel_task = (g_c < im_cols) & (g_r < im_rows);
            if(valid_pixel_task){
                assert_val(tile_label < tile_rows * tile_cols, tile_label);
                pixel_at(LabelT, labelim, g_r, g_c) = glabel;
            }
        }
    }
}

inline
LabelT find_root_global(__global LabelT *labelim_p, uint labelim_pitch, LabelT label
#ifdef DYNAMIC_IMGDIMS
    , const uint im_rows, const uint im_cols
#else
    ,const uint d1, const uint d2
#endif
){
    for(;;){
        const uint y = label / im_cols;
        const uint x = label % im_cols;
        assert_val(y < im_rows, y);
        assert_val(x < im_cols, x);
        const LabelT parent = pixel_at(LabelT, labelim, y, x);

        if(label == parent){
            break;
        }

        label = parent;
    }
    return label;
}

inline
LabelT find_root_global_uncached(__global LabelT *labelim_p, uint labelim_pitch, LabelT label
#ifdef DYNAMIC_IMGDIMS
    ,const uint im_rows, const uint im_cols
#else
    ,const uint d1, const uint d2
#endif
){
    for(;;){
        const uint r = label / im_cols;
        const uint c = label % im_cols;
        assert_val(r < im_rows, r);
        assert_val(c < im_cols, c);
        LabelT parent = atomic_load(&pixel_at(LabelT, labelim, r, c));

        if(label == parent){
            break;
        }

        label = parent;
    }
    return label;
}

__kernel void compact_paths_global(
#ifdef DYNAMIC_IMGDIMS
    const uint im_rows, const uint im_cols,
#endif
    __global LabelT *labelim_p, uint labelim_pitch){
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if((x < im_cols) & (y < im_rows)){
        const LabelT label = pixel_at(LabelT, labelim, y, x);
        pixel_at(LabelT, labelim, y, x) = find_root_global(labelim_p, labelim_pitch, label, im_rows, im_cols);
    }
}

#ifndef MERGE_CONFLICT_STATS
#define MERGE_CONFLICT_STATS 0
#endif

uint merge_edge_labels(
#ifdef DYNAMIC_IMGDIMS
    const uint im_rows, const uint im_cols,
#else
    const uint d1, const uint d2,
#endif
    __global LabelT *labelim_p, const uint labelim_pitch, uint l1_r, uint l1_c, uint l2_r, uint l2_c, __global uint *conflicts){
    LabelT l1 = atomic_load(&pixel_at(LabelT, labelim, l1_r, l1_c));
    LabelT l2 = atomic_load(&pixel_at(LabelT, labelim, l2_r, l2_c));
    if(l1 == l2){
        return 0;
    }
    LabelT r1 = find_root_global_uncached(labelim_p, labelim_pitch, l1, im_rows, im_cols);
    LabelT r2 = find_root_global_uncached(labelim_p, labelim_pitch, l2, im_rows, im_cols);
    if(r1 == r2){
        return 0;
    }
    LabelT mi = min(r1, r2);
    LabelT ma = max(r1, r2);
    uint ret = 0;

    for(;;){

        //const uint mi_y = mi / im_cols;
        //const uint mi_x = mi % im_cols;
        const uint ma_y = ma / im_cols;
        const uint ma_x = ma % im_cols;

        __global LabelT *ma_lp = &pixel_at(LabelT, labelim, ma_y, ma_x);
        const LabelT old_label_of_ma = atomic_min(ma_lp, mi);

        if(old_label_of_ma >= mi){
            //printf("merge successful with mi = %d ma = %d old_label_ma: %d\n", mi, ma, old_label_of_ma);
            ret = old_label_of_ma == mi ? 0 : 1;
            break;
        }
#if MERGE_CONFLICT_STATS
        atomic_inc(conflicts);
#endif

        //printf("condition detected with mi = %d ma = %d old_label_ma: %d\n", mi, ma, old_label_of_ma);
        //else somebody snuck in and made the max label now smaller than ours! we need to now retry to merge
        l1 = atomic_load(&pixel_at(LabelT, labelim, l1_r, l1_c));
        l2 = atomic_load(&pixel_at(LabelT, labelim, l2_r, l2_c));

        r1 = find_root_global_uncached(labelim_p, labelim_pitch, l1, im_rows, im_cols);
        r2 = find_root_global_uncached(labelim_p, labelim_pitch, l2, im_rows, im_cols);
        if(r1 == r2){
            break;
        }
        mi = min(r1, r2);
        ma = max(r1, r2);
    }

    return ret;
}

#define MERGE_BOTH_EDGES 0
#define NWAY_MERGE_IN_ROW_TILES 2
#define NWAY_MERGE_IN_COL_TILES 2
#define nway_merge_in_row_tiles NWAY_MERGE_IN_ROW_TILES
#define nway_merge_in_col_tiles NWAY_MERGE_IN_COL_TILES

#define MERGE_TILE_HEADER                                                                                                 \
    size_t rmerge_job_id;                                                                                                 \
    size_t cmerge_job_id;                                                                                                 \
    if((nrow_tile_merges > 0) & (ncol_tile_merges > 0)){                                                                  \
        rmerge_job_id = get_group_id(0) / ncol_tile_merges;                                                               \
        cmerge_job_id = get_group_id(0) % ncol_tile_merges;                                                               \
        assert_val(rmerge_job_id < nrow_tile_merges, rmerge_job_id);                                                      \
        assert_val(cmerge_job_id < ncol_tile_merges, cmerge_job_id);                                                      \
    }else if(nrow_tile_merges > 0){/*ncol_tile_merges = 0*/                                                               \
        rmerge_job_id = get_group_id(0);                                                                                  \
        cmerge_job_id = 0;                                                                                                \
        assert_val(rmerge_job_id < nrow_tile_merges, rmerge_job_id);                                                      \
    }else{/*nrow_tile_merges = 0*/                                                                                        \
        rmerge_job_id = 0;                                                                                                \
        cmerge_job_id = get_group_id(0);                                                                                  \
        assert_val(cmerge_job_id < ncol_tile_merges, cmerge_job_id);                                                      \
    }                                                                                                                     \
                                                                                                                          \
    const size_t tid = get_local_id(0);                                                                                   \
    const uint rmerge_block_index_start = (rmerge_job_id + 0) * block_size_in_row_tiles * nway_merge_in_row_tiles;        \
    const uint rmerge_block_index_end = (rmerge_job_id + 1) * block_size_in_row_tiles * nway_merge_in_row_tiles;          \
    const uint rmerge_start = rmerge_block_index_start * TILE_ROWS;                                                       \
    const uint rmerge_end = min(rmerge_block_index_end * TILE_ROWS, im_rows);                                             \
    const uint cmerge_block_index_start = (cmerge_job_id + 0) * block_size_in_col_tiles * nway_merge_in_col_tiles;        \
    const uint cmerge_block_index_end = (cmerge_job_id + 1) * block_size_in_col_tiles * nway_merge_in_col_tiles;          \
    const uint cmerge_start = cmerge_block_index_start * TILE_COLS;                                                       \
    const uint cmerge_end = min(cmerge_block_index_end * TILE_COLS, im_cols);                                             \
                                                                                                                          \
    const size_t line_wg_id = get_group_id(1);                                                                            \
    const size_t nline_workers = get_num_groups(1);                                                                       \
    const size_t wg_size = get_local_size(0);                                                                             \
    const size_t line_block_size = wg_size;/*efficient block size*/

#define BLOCKED_LINE_HEADER(start, end)                                                                                                                   \
    const size_t nline_blocks = divUp((end) - (start), line_block_size);/*number of efficiently processible blocks*/                                      \
    const size_t nline_blocks_per_wg = nline_blocks / nline_workers;                                                                                      \
    const size_t nline_blocks_remainder = nline_blocks - (nline_workers * nline_blocks_per_wg);                                                           \
    const size_t nline_blocks_to_left = nline_blocks_per_wg * line_wg_id + (line_wg_id < nline_blocks_remainder ? line_wg_id : nline_blocks_remainder);   \
    const size_t n_wg_blocks = nline_blocks_per_wg + (line_wg_id < nline_blocks_remainder ? 1 : 0);                                                       \
    const size_t line_start_index = nline_blocks_to_left * line_block_size + (start);                                                                     \
    const size_t line_end_index = min(line_start_index + n_wg_blocks * line_block_size, (end));/*block aligned end*/

//ncalls: logUp(ntiles, nway_merge)
//group size: k, 1: k can be anything
//gdims: roundUpToMultiple(im_cols, k), nmerges : nmerges = ntiles // (nway_merge * block_size)
//block_size: nway_merge^(call_index) for call_index=[0, ncalls): 1<=block_size<=nway_merge^(logUp(nhorz_tiles, nway_merge)-1)
//a horizontal merge spanning vertically in cols
__kernel void merge_tiles(
#ifdef DYNAMIC_IMGDIMS
    const uint im_rows, const uint im_cols,
#endif
    const uint block_size_in_row_tiles,
    const uint block_size_in_col_tiles,
    const uint nrow_tile_merges, const uint ncol_tile_merges,
    const __global ConnectivityPixelT *connectivityim_p, const uint connectivityim_pitch,
    __global LabelT *labelim_p, const uint labelim_pitch
    ,__global uint *gn_merge_conflicts
){
    MERGE_TILE_HEADER

    uint pn_merge_conflicts;
    do{
        __local uint n_merge_conflicts;
        if(tid == 0){
            n_merge_conflicts = 0;
        }
        lds_barrier();
        pn_merge_conflicts = 0;
        if(nrow_tile_merges){
            BLOCKED_LINE_HEADER(cmerge_start, ((size_t)cmerge_end))
            assert_val(block_size_in_row_tiles * TILE_ROWS < im_rows, block_size_in_row_tiles * TILE_ROWS);
            assert_val(block_size_in_row_tiles < divUp(im_rows, TILE_ROWS), block_size_in_row_tiles);
            for(uint rmerge_sub_index = 1; rmerge_sub_index < nway_merge_in_row_tiles; rmerge_sub_index++){
                const uint rmerge_block_index = rmerge_block_index_start + block_size_in_row_tiles * rmerge_sub_index;
                assert_val(rmerge_sub_index < nway_merge_in_row_tiles, rmerge_sub_index);
                if((cmerge_start != cmerge_end) & (tid == 0)){
                    assert_val(r < im_rows, r);
                }
                {
                    const uint r = rmerge_block_index * TILE_ROWS;//the middle point to merge about
                    //merge along the columns - ie this merges to horizontally seperated tiles

                    for(uint c = line_start_index + tid; c < line_end_index; c += get_local_size(0)){
                    //for(uint c = cmerge_start + tid; c < cmerge_end; c += get_local_size(0)){
                        const ConnectivityPixelT e = pixel_at(ConnectivityPixelT, connectivityim, r, c);

                        if(e & UP){
                            pn_merge_conflicts += merge_edge_labels(im_rows, im_cols, labelim_p, labelim_pitch, r, c, r - 1, c, gn_merge_conflicts);
                        }
                        #if CONNECTIVITY == 8
                        if(e & LEFT_UP){
                            pn_merge_conflicts += merge_edge_labels(im_rows, im_cols, labelim_p, labelim_pitch, r, c, r - 1, c - 1, gn_merge_conflicts);
                        }
                        if(e & RIGHT_UP){
                            pn_merge_conflicts += merge_edge_labels(im_rows, im_cols, labelim_p, labelim_pitch, r, c, r - 1, c + 1, gn_merge_conflicts);
                        }
                        #endif
                    }
                }
                #if MERGE_BOTH_EDGES
                {
                    const uint r = rmerge_block_index * TILE_ROWS - 1;//the middle point to merge about
                    //merge along the columns - ie this merges to horizontally seperated tiles
                    for(uint c = line_start_index + tid; c < line_end_index; c += get_local_size(0)){
                    //for(uint c = cmerge_start + tid; c < cmerge_end; c += get_local_size(0)){
                        const ConnectivityPixelT e = pixel_at(ConnectivityPixelT, connectivityim, r, c);
                        if(e & DOWN){
                            pn_merge_conflicts += merge_edge_labels(im_rows, im_cols, labelim_p, labelim_pitch, r, c, r + 1, c, gn_merge_conflicts);
                        }
                        #if CONNECTIVITY == 8
                        if(e & LEFT_DOWN){
                            pn_merge_conflicts += merge_edge_labels(im_rows, im_cols, labelim_p, labelim_pitch, r, c, r + 1, c - 1, gn_merge_conflicts);
                        }
                        if(e & RIGHT_DOWN){
                            pn_merge_conflicts += merge_edge_labels(im_rows, im_cols, labelim_p, labelim_pitch, r, c, r + 1, c + 1, gn_merge_conflicts);
                        }
                        #endif
                    }
                }
                #endif
            }
        }

        if(ncol_tile_merges){
            BLOCKED_LINE_HEADER(rmerge_start, ((size_t)rmerge_end))

            assert_val(block_size_in_col_tiles < divUp(im_cols, TILE_COLS), block_size_in_col_tiles);
            assert_val(block_size_in_col_tiles * TILE_COLS < im_cols, block_size_in_col_tiles * TILE_COLS);
            for(uint cmerge_sub_index = 1; cmerge_sub_index < nway_merge_in_col_tiles; cmerge_sub_index++){
                const uint cmerge_block_index = cmerge_block_index_start + block_size_in_col_tiles * cmerge_sub_index;
                assert_val(cmerge_sub_index < nway_merge_in_row_tiles, cmerge_sub_index);
                if((rmerge_start != rmerge_end) & (tid == 0)){
                    assert_val(c < im_cols, c);
                }
                {
                    const uint c = cmerge_block_index * TILE_COLS;//the middle point to merge about
                    //merge along the rows - ie this merges to vertically seperated tiles
                    //for(uint r = rmerge_start + tid; r < rmerge_end; r += get_local_size(0)){
                    for(uint r = line_start_index + tid; r < line_end_index; r += get_local_size(0)){

                        const ConnectivityPixelT e = pixel_at(ConnectivityPixelT, connectivityim, r, c);
                        if(e & LEFT){
                            pn_merge_conflicts += merge_edge_labels(im_rows, im_cols, labelim_p, labelim_pitch, r, c, r, c - 1, gn_merge_conflicts);
                        }
                        #if CONNECTIVITY == 8
                        if(e & LEFT_UP){
                            pn_merge_conflicts += merge_edge_labels(im_rows, im_cols, labelim_p, labelim_pitch, r, c, r - 1, c - 1, gn_merge_conflicts);
                        }
                        if(e & LEFT_DOWN){
                            pn_merge_conflicts += merge_edge_labels(im_rows, im_cols, labelim_p, labelim_pitch, r, c, r + 1, c - 1, gn_merge_conflicts);
                        }
                        #endif
                    }
                }
                #if MERGE_BOTH_EDGES
                {
                    const uint c = cmerge_block_index * TILE_COLS - 1;//the middle point to merge about
                    //merge along the rows - ie this merges to vertically seperated tiles
                    //for(uint r = rmerge_start + tid; r < rmerge_end; r += get_local_size(0)){
                    for(uint r = line_start_index + tid; r < line_end_index; r += get_local_size(0)){
                        //if(r >= 2003 && r < 2012 &&  c >= 1183 && c < 1185){
                        //    printf("%d %d %d\n", r, c, lc);
                        //}

                        const ConnectivityPixelT e = pixel_at(ConnectivityPixelT, connectivityim, r, c);
                        if(e & RIGHT){
                            pn_merge_conflicts += merge_edge_labels(im_rows, im_cols, labelim_p, labelim_pitch, r, c, r, c + 1, gn_merge_conflicts);
                        }
                        #if CONNECTIVITY == 8
                        if(e & RIGHT_UP){
                            pn_merge_conflicts += merge_edge_labels(im_rows, im_cols, labelim_p, labelim_pitch, r, c, r - 1, c + 1, gn_merge_conflicts);
                        }
                        if(e & RIGHT_DOWN){
                            pn_merge_conflicts += merge_edge_labels(im_rows, im_cols, labelim_p, labelim_pitch, r, c, r + 1, c + 1, gn_merge_conflicts);
                        }
                        #endif
                    }
                }
                #endif
            }
        }

        //if(tid == 0){
        //    printf("merge_tile_on_rows nway_merge: %d merge_job_id: %d/%d merge_sub_index: %d merge_blocK_index: %d block_size: %d row: %d\n", nway_merge, merge_job_id, get_num_groups(1), merge_sub_index, merge_block_index, block_size, r);
        //}
        atomic_add(&n_merge_conflicts, pn_merge_conflicts);
        lds_barrier();
        pn_merge_conflicts = n_merge_conflicts;
    }while(pn_merge_conflicts);
}

__kernel void post_merge_convergence_check(
#ifdef DYNAMIC_IMGDIMS
    const uint im_rows, const uint im_cols,
#endif
    const uint block_size_in_row_tiles,
    const uint block_size_in_col_tiles,
    const uint nrow_tile_merges, const uint ncol_tile_merges,
    const __global ConnectivityPixelT *connectivityim_p, const uint connectivityim_pitch,
    const __global LabelT *labelim_p, const uint labelim_pitch
    ,__global uint *gn_failed_merges
){
    MERGE_TILE_HEADER

    uint pn_failed_merges;
    __local uint n_failed_merges;
    if(tid == 0){
        n_failed_merges = 0;
    }
    lds_barrier();
    pn_failed_merges = 0;
    if(nrow_tile_merges){
        BLOCKED_LINE_HEADER(cmerge_start, ((size_t)cmerge_end))
        assert_val(block_size_in_row_tiles * TILE_ROWS < im_rows, block_size_in_row_tiles * TILE_ROWS);
        assert_val(block_size_in_row_tiles < divUp(im_rows, TILE_ROWS), block_size_in_row_tiles);
        #pragma unroll
        for(uint rmerge_sub_index = 1; rmerge_sub_index < nway_merge_in_row_tiles; rmerge_sub_index++){
            const uint rmerge_block_index = rmerge_block_index_start + block_size_in_row_tiles * rmerge_sub_index;
            assert_val(rmerge_sub_index < nway_merge_in_row_tiles, rmerge_sub_index);
            if((cmerge_start != cmerge_end) & (tid == 0)){
                assert_val(r < im_rows, r);
            }
            {
                const uint r = rmerge_block_index * TILE_ROWS;//the middle point to merge about
                //merge along the columns - ie this merges to horizontally seperated tiles
                for(uint c = line_start_index + tid; c < line_end_index; c += get_local_size(0)){
                //for(uint c = cmerge_start + tid; c < cmerge_end; c += get_local_size(0)){
                    const ConnectivityPixelT e = pixel_at(ConnectivityPixelT, connectivityim, r, c);

                    const LabelT l1 = pixel_at(LabelT, labelim, r, c);
                    //const LabelT r1 = find_root_global(labelim_p, labelim_pitch, l1, im_rows, im_cols);
                    if(e & UP){
                        const LabelT l2 = pixel_at(LabelT, labelim, r - 1, c);
                        //const LabelT r2 = find_root_global(labelim_p, labelim_pitch, l2, im_rows, im_cols);
                        pn_failed_merges += l1 != l2;
                    }
                    #if CONNECTIVITY == 8
                    if(e & LEFT_UP){
                        const LabelT l2 = pixel_at(LabelT, labelim, r - 1, c - 1);
                        pn_failed_merges += l1 != l2;
                    }
                    if(e & RIGHT_UP){
                        const LabelT l2 = pixel_at(LabelT, labelim, r - 1, c + 1);
                        pn_failed_merges += l1 != l2;
                    }
                    #endif
                }
            }
        }
    }

    if(ncol_tile_merges){
        BLOCKED_LINE_HEADER(rmerge_start, ((size_t)rmerge_end))
        assert_val(block_size_in_col_tiles < divUp(im_cols, TILE_COLS), block_size_in_col_tiles);
        assert_val(block_size_in_col_tiles * TILE_COLS < im_cols, block_size_in_col_tiles * TILE_COLS);
        #pragma unroll
        for(uint cmerge_sub_index = 1; cmerge_sub_index < nway_merge_in_col_tiles; cmerge_sub_index++){
            const uint cmerge_block_index = cmerge_block_index_start + block_size_in_col_tiles * cmerge_sub_index;
            assert_val(cmerge_sub_index < nway_merge_in_row_tiles, cmerge_sub_index);
            if((rmerge_start != rmerge_end) & (tid == 0)){
                assert_val(c < im_cols, c);
            }
            {
                const uint c = cmerge_block_index * TILE_COLS;//the middle point to merge about
                //merge along the rows - ie this merges to vertically seperated tiles
                //for(uint r = rmerge_start + tid; r < rmerge_end; r += get_local_size(0)){
                for(uint r = line_start_index + tid; r < line_end_index; r += get_local_size(0)){

                    const ConnectivityPixelT e = pixel_at(ConnectivityPixelT, connectivityim, r, c);
                    const LabelT l1 = pixel_at(LabelT, labelim, r, c);
                    //const LabelT r1 = find_root_global(labelim_p, labelim_pitch, l1, im_rows, im_cols);
                    if(e & LEFT){
                        const LabelT l2 = pixel_at(LabelT, labelim, r, c - 1);
                        pn_failed_merges += l1 != l2;
                    }
                    #if CONNECTIVITY == 8
                    if(e & LEFT_UP){
                        const LabelT l2 = pixel_at(LabelT, labelim, r - 1, c - 1);
                        pn_failed_merges += l1 != l2;
                    }
                    if(e & LEFT_DOWN){
                        const LabelT l2 = pixel_at(LabelT, labelim, r + 1, c - 1);
                        pn_failed_merges += l1 != l2;
                    }
                    #endif
                }
            }
        }
    }
    atomic_add(&n_failed_merges, pn_failed_merges);
    lds_barrier();
    if(tid == 0){
        atomic_add(gn_failed_merges, n_failed_merges);
    }
}

//ncalls: logUp(ntiles, nway_merge)
//group size: k, 1: k can be anything
//gdims: roundUpToMultiple(im_cols, k), nmerges : nmerges = ntiles // (nway_merge * block_size)
//block_size: nway_merge^(call_index) for call_index=[0, ncalls): 1<=block_size<=nway_merge^(logUp(nhorz_tiles, nway_merge)-1)
//a horizontal merge spanning vertically in cols
__kernel void post_merge_flatten(
#ifdef DYNAMIC_IMGDIMS
    const uint im_rows, const uint im_cols,
#endif
    const uint block_size_in_row_tiles,
    const uint block_size_in_col_tiles,
    const uint nrow_tile_merges, const uint ncol_tile_merges,
    const __global ConnectivityPixelT *connectivityim_p, const uint connectivityim_pitch,
    __global LabelT *labelim_p, const uint labelim_pitch
){
    MERGE_TILE_HEADER

    if(nrow_tile_merges){
        BLOCKED_LINE_HEADER(cmerge_start, ((size_t)cmerge_end))
        assert_val(block_size_in_row_tiles * TILE_ROWS < im_rows, block_size_in_row_tiles * TILE_ROWS);
        assert_val(block_size_in_row_tiles < divUp(im_rows, TILE_ROWS), block_size_in_row_tiles);
        for(uint rmerge_sub_index = 1; rmerge_sub_index < nway_merge_in_row_tiles; rmerge_sub_index++){
            const uint rmerge_block_index = rmerge_block_index_start + block_size_in_row_tiles * rmerge_sub_index;
            assert_val(rmerge_sub_index < nway_merge_in_row_tiles, rmerge_sub_index);
            if((cmerge_start != cmerge_end) & (tid == 0)){
                assert_val(r < im_rows, r);
            }
            #pragma unroll
            for(uint i = 0; i < 2; ++i){
                const uint r = rmerge_block_index * TILE_ROWS - i;//the middle point to merge about
                //flatten along the columns - ie flattens on the line to horizontally seperated tiles
                //for(uint c = cmerge_start + tid; c < cmerge_end; c += get_local_size(0)){
                for(uint c = line_start_index + tid; c < line_end_index; c += get_local_size(0)){
                    const LabelT label = pixel_at(LabelT, labelim, r, c);
                    pixel_at(LabelT, labelim, r, c) = find_root_global(labelim_p, labelim_pitch, label, im_rows, im_cols);
                }
            }
        }
    }

    if(ncol_tile_merges){
        BLOCKED_LINE_HEADER(rmerge_start, ((size_t)rmerge_end))

        assert_val(block_size_in_col_tiles < divUp(im_cols, TILE_COLS), block_size_in_col_tiles);
        assert_val(block_size_in_col_tiles * TILE_COLS < im_cols, block_size_in_col_tiles * TILE_COLS);
        for(uint cmerge_sub_index = 1; cmerge_sub_index < nway_merge_in_col_tiles; cmerge_sub_index++){
            const uint cmerge_block_index = cmerge_block_index_start + block_size_in_col_tiles * cmerge_sub_index;
            assert_val(cmerge_sub_index < nway_merge_in_row_tiles, cmerge_sub_index);
            if((rmerge_start != rmerge_end) & (tid == 0)){
                assert_val(c < im_cols, c);
            }
            #pragma unroll
            for(uint i = 0; i < 2; ++i){
                const uint c = cmerge_block_index * TILE_COLS - i;//the middle point to merge about
                //merge along the rows - ie this merges to vertically seperated tiles
                //for(uint r = line_start_index + tid; r < line_end_index; r += get_local_size(0)){
                for(uint r = rmerge_start + tid; r < rmerge_end; r += get_local_size(0)){
                    const LabelT label = pixel_at(LabelT, labelim, r, c);
                    pixel_at(LabelT, labelim, r, c) = find_root_global(labelim_p, labelim_pitch, label, im_rows, im_cols);
                }
            }
        }
    }
}


__kernel void mark_root_classes(
#ifdef DYNAMIC_IMGDIMS
    const uint im_rows, const uint im_cols,
#endif
    __global PixelT *image_p, uint image_pitch,
    __global const LabelT* labelim_p, const uint labelim_pitch,
    __global uint* is_root_class_image_p, const uint is_root_class_image_pitch
){
    const uint c = get_global_id(0);
    const uint r = get_global_id(1);
    const bool valid_pixel_task = (c < im_cols) & (r < im_rows);

    const uint linear_index = c + r * im_cols;
    if(valid_pixel_task){
        const PixelT pixel = pixel_at(PixelT, image, r, c);
        const LabelT label = pixel_at(LabelT, labelim, r, c);
        pixel_at(uint, is_root_class_image, r, c) = (pixel != BG_VALUE) & (label == linear_index);
    }
}

#ifndef USE_CL2_WORKGROUP_FUNCTIONS
MAKE_WORK_GROUP_FUNCTIONS(uint, uint, 0U, UINT_MAX)
#endif

#define PREFIX_SUM_HEADER                                                                                                                \
    const uint narray_workers = get_num_groups(0);                                                                                       \
    const uint array_length = im_rows * im_cols;                                                                                         \
    const uint array_wg_id = get_group_id(0);                                                                                            \
    const uint wg_size = get_local_size(0);                                                                                              \
    const uint block_size = wg_size;/*efficient block size*/                                                                             \
    const uint nblocks = divUp(array_length, block_size);/*number of efficiently processible blocks*/                                    \
    const uint nblocks_per_wg = nblocks / narray_workers;                                                                                \
    const uint nblocks_to_merge = nblocks / nblocks_per_wg;                                                                              \
    const uint nblocks_remainder = nblocks - (narray_workers * nblocks_per_wg);                                                          \
    const uint nblocks_to_left = nblocks_per_wg * array_wg_id + (array_wg_id < nblocks_remainder ? array_wg_id : nblocks_remainder);     \
    const uint n_wg_blocks = nblocks_per_wg + (array_wg_id < nblocks_remainder ? 1 : 0);                                                 \
                                                                                                                                         \
    const uint start_index = nblocks_to_left * block_size;                                                                               \
    const uint end_index_ = start_index + n_wg_blocks * block_size;/*block aligned end*/

//root class inclusive prefix sums belonging to each compute unit given to each tile - note if narray_workers == 1, no merge step is necessary
//computes local prefix sums to get intra-wg blocksums, prefix sum that to get intra-wg offsets - this is needed to merge the final blocksums
//global dims: <wgs_per_histogram, n_tiles>, work_dims: <wg_size, 1>
//global blocksums[divUp(nblocks, blocks_per_wg)]
#ifdef GPU_ARCH
__attribute__((reqd_work_group_size(DEVICE_WAVEFRONT_SIZE, 1, 1)))
#endif
__kernel void mark_roots_and_make_intra_wg_block_local_prefix_sums(
#ifdef DYNAMIC_IMGDIMS
    const uint im_rows, const uint im_cols,
#endif
    __global const PixelT *image_p, uint image_pitch,
    __global const LabelT* labelim_p, const uint labelim_pitch,
    __global uint * restrict array_intra_wg_block_sums_p,
    __global uint * restrict array_prefix_sum_p, const uint array_prefix_sum_pitch
){

    PREFIX_SUM_HEADER
    (void) nblocks_to_merge;
    uint inter_block_sum = 0;

    for(uint linear_index = get_local_id(0) + start_index; linear_index < end_index_; linear_index += wg_size){
        const uint r = linear_index / im_cols;
        const uint c = linear_index % im_cols;

        uint count = 0;
        if(linear_index < array_length){
#if FUSED_MARK_KERNEL
            const PixelT pixel = pixel_at(PixelT, image, r, c);
            const LabelT label = pixel_at(LabelT, labelim, r, c);
            count = ((pixel != BG_VALUE) & (label == linear_index)) ? 1 : 0;
#else
            count = pixel_at(uint, array_prefix_sum, r, c);
#endif
        }

#ifdef USE_CL2_WORKGROUP_FUNCTIONS
        uint block_prefix_sum_inclusive = work_group_scan_inclusive_add(count);
#else
#ifdef GPU_ARCH
        __local uint lmem[WORK_GROUP_FUNCTION_MEMORY_SIZE_POWER2_(DEVICE_WAVEFRONT_SIZE)];
#else
        __local uint lmem[WORK_GROUP_FUNCTION_MAX_MEMORY_SIZE];
#endif
        uint block_prefix_sum_inclusive = clc_work_group_scan_inclusive_add_uint(count, lmem);
#endif

        block_prefix_sum_inclusive += inter_block_sum;
        if(linear_index < array_length){
            pixel_at(uint, array_prefix_sum, r, c) = block_prefix_sum_inclusive;
        }
#ifdef USE_CL2_WORKGROUP_FUNCTIONS
        inter_block_sum = work_group_broadcast(block_prefix_sum_inclusive, wg_size - 1);
#else
        __local uint value;
        inter_block_sum = clc_work_group_broadcast1_uint(block_prefix_sum_inclusive, wg_size - 1, &value);
#endif
    }
    if((array_intra_wg_block_sums_p != 0) & (get_local_id(0) == wg_size - 1)){
        array_intra_wg_block_sums_p[array_wg_id] = inter_block_sum;
    }
}

//exclusive prefix sums intra-wg blocksums to get intra-wg offsets - needed to merge together all the wg-local prefix sums
//global dims: <1, n_tiles>, work_dims: <wg_size, 1>
//global tile_intra_wg_block_sums[n_tiles][nblocks_to_merge]
#ifdef GPU_ARCH
__attribute__((reqd_work_group_size(DEVICE_WAVEFRONT_SIZE, 1, 1)))
#endif
__kernel void make_intra_wg_block_global_sums(
    __global uint * restrict intra_wg_block_sums_p, uint nblocks_to_merge
){
    const uint wg_size = get_local_size(0);

    __local uint intra_wg_block_sums[WG_SIZE_MAX + 1];
    if(get_local_id(0) == 0){
        intra_wg_block_sums[0] = 0;
    }
    lds_barrier();
    uint intra_wg_block_offset = 0;

    for(uint intra_wg_block_id = get_local_id(0); intra_wg_block_id < divUp(nblocks_to_merge, wg_size) * wg_size; intra_wg_block_id += wg_size){
        //get the unsumed blocksums
        const uint intra_wg_block_sum = intra_wg_block_id < nblocks_to_merge ? intra_wg_block_sums_p[intra_wg_block_id] : 0;
        intra_wg_block_sums[get_local_id(0) + 1] = intra_wg_block_sum;
        lds_barrier();

        const uint intra_wg_block_sum_delayed = intra_wg_block_sums[get_local_id(0)];
#ifdef USE_CL2_WORKGROUP_FUNCTIONS
        intra_wg_block_offset += work_group_scan_inclusive_add(intra_wg_block_sum_delayed);
#else
        __local uint lmem[WORK_GROUP_FUNCTION_MAX_MEMORY_SIZE];
        intra_wg_block_offset += clc_work_group_scan_inclusive_add_uint(intra_wg_block_sum_delayed, lmem);
#endif
        if(intra_wg_block_id < nblocks_to_merge){
            intra_wg_block_sums_p[intra_wg_block_id] = intra_wg_block_offset;
        }
#ifdef USE_CL2_WORKGROUP_FUNCTIONS
        intra_wg_block_offset = work_group_broadcast(intra_wg_block_offset, wg_size - 1);
#else
        __local uint value;
        intra_wg_block_offset = clc_work_group_broadcast1_uint(intra_wg_block_offset, wg_size - 1, &value);
#endif

        if(get_local_id(0) == wg_size-1){
            intra_wg_block_sums[0] = intra_wg_block_sums[wg_size];
        }
        lds_barrier();
    }
}

//merges global offsets of intra-wg-block offsets of prefix sums
//global dims: <wgs_per_sum>, work_dims: <wg_size> : wg_size >= nblocks_to_merge
//global array_of_prefix_sums[im_rows*im_cols] : as input partial sums, as output full prefix sum
#ifdef GPU_ARCH
__attribute__((reqd_work_group_size(DEVICE_WAVEFRONT_SIZE, 1, 1)))
#endif
__kernel void make_prefix_sums_with_intra_wg_block_global_sums(
#ifdef DYNAMIC_IMGDIMS
    const uint im_rows, const uint im_cols,
#endif
    __global const uint * restrict intra_wg_block_sums_p,
    __global uint * restrict array_of_prefix_sums_p, uint array_of_prefix_sums_pitch,
    __global LabelT* label_count_p
){
    PREFIX_SUM_HEADER
    (void) nblocks_to_merge;

    const uint inter_block_sum = intra_wg_block_sums_p[array_wg_id];
    for(uint linear_index = get_local_id(0) + start_index; linear_index < end_index_; linear_index += wg_size){
        if(linear_index < array_length){
            const uint r = linear_index / im_cols;
            const uint c = linear_index % im_cols;
            const uint count = image_pixel_at(uint, array_of_prefix_sums_p, im_rows, im_cols, array_of_prefix_sums_pitch, r, c) + inter_block_sum;
            if(linear_index == array_length - 1){
                *label_count_p = count + 1;//include BG
            }
            image_pixel_at(uint, array_of_prefix_sums_p, im_rows, im_cols, array_of_prefix_sums_pitch, r, c) = count;
        }
    }
}

__kernel void relabel_with_scanline_order(
#ifdef DYNAMIC_IMGDIMS
    const uint im_rows, const uint im_cols,
#endif
    __global LabelT* labelim_out_p, const uint labelim_out_pitch,
    __global const PixelT* image_p, const uint image_pitch,
    __global const LabelT* labelim_p, const uint labelim_pitch,
    __global uint* scanline_prefix_sum_of_root_classes_p, const uint scanline_prefix_sum_of_root_classes_pitch
){
    const uint c = get_global_id(0);
    const uint r = get_global_id(1);
    const bool valid_pixel_task = (c < im_cols) & (r < im_rows);

    if(valid_pixel_task){
        const PixelT pixel = pixel_at(PixelT, image, r, c);
        LabelT final_label = 0;
        if(pixel != BG_VALUE){
            const LabelT label = pixel_at(LabelT, labelim, r, c);
            const uint label_r = label / im_cols;
            const uint label_c = label % im_cols;
            final_label = pixel_at(uint, scanline_prefix_sum_of_root_classes, label_r, label_c);
        }
        pixel_at(LabelT, labelim_out, r, c) = final_label;
    }
}

__kernel void count_invalid_labels(
#ifdef DYNAMIC_IMGDIMS
    const uint im_rows, const uint im_cols,
#endif
    __global const LabelT* labelim_p, const uint labelim_pitch,
    __global const ConnectivityPixelT *connectivityim_p, const uint connectivityim_pitch,
    __global const uint *dcountim_p, const uint dcountim_pitch
){
    const uint c = get_global_id(0);
    const uint r = get_global_id(1);
    const bool valid_pixel_task = (c < im_cols) & (r < im_rows);

    if(valid_pixel_task){
        const ConnectivityPixelT connectivity = pixel_at(ConnectivityPixelT, connectivityim, r, c);
        const LabelT label = pixel_at(LabelT, labelim, r, c);
        uint dcount = 0;

        dcount += connectivity & UP         ? label != pixel_at(LabelT, labelim, r - 1, c - 0) : 0;
        dcount += connectivity & LEFT_UP    ? label != pixel_at(LabelT, labelim, r - 1, c - 1) : 0;
        dcount += connectivity & LEFT       ? label != pixel_at(LabelT, labelim, r - 0, c - 1) : 0;
        dcount += connectivity & LEFT_DOWN  ? label != pixel_at(LabelT, labelim, r + 1, c - 1) : 0;
        dcount += connectivity & DOWN       ? label != pixel_at(LabelT, labelim, r + 1, c - 0) : 0;
        dcount += connectivity & RIGHT_DOWN ? label != pixel_at(LabelT, labelim, r + 1, c + 1) : 0;
        dcount += connectivity & RIGHT      ? label != pixel_at(LabelT, labelim, r + 0, c + 1) : 0;
        dcount += connectivity & RIGHT_UP   ? label != pixel_at(LabelT, labelim, r - 1, c + 1) : 0;

        pixel_at(uint, dcountim, r, c) = dcount;
    }
}
