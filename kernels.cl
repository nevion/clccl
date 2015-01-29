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

#ifndef WORKGROUP_TILE_SIZE_X
#define WORKGROUP_TILE_SIZE_X 32
#endif
#ifndef WORKGROUP_TILE_SIZE_Y
#define WORKGROUP_TILE_SIZE_Y 8
#endif

#ifndef WORKITEM_REPEAT_X
#define WORKITEM_REPEAT_X 1
#endif
#ifndef WORKITEM_REPEAT_Y
#define WORKITEM_REPEAT_Y 4
#endif

#define TILE_COLS (WORKGROUP_TILE_SIZE_X * WORKITEM_REPEAT_X)
#define TILE_ROWS (WORKGROUP_TILE_SIZE_Y * WORKITEM_REPEAT_Y)

enum ConnectivityEnum {
    UP = (1<<0),
    LEFT = (1<<1),
    DOWN = (1<<2),
    RIGHT = (1<<3),
    LEFT_UP = (1<<4),
    LEFT_DOWN = (1<<5),
    RIGHT_UP = (1<<6),
    RIGHT_DOWN = (1<<7)
};

#define isConnected(p1, p2) ((p1) == (p2))

#define pixel_at(type, basename, r, c) image_pixel_at(type, PASTE2(basename, _p), im_rows, im_cols, PASTE2(basename, _pitch), (r), (c))

//global dimensions: divUp(image.cols, block.x), divUp(image.rows, block.y);
//__attribute__((reqd_work_group_size(WORKGROUP_TILE_SIZE_X, WORKGROUP_TILE_SIZE_Y, 1)))
//__kernel void
//make_connectivity_image(uint im_rows, uint im_cols, __global PixelT *image_p, uint image_pitch, __global ConnectivityPixelT *connectivityim_p, uint connectivityim_pitch){
//    const uint c = get_global_id(1);
//    const uint r = get_global_id(0);
//    const bool valid_pixel_task = (c < im_cols) & (r < im_rows);
//
//    if(valid_pixel_task){
//        PixelT pixel = pixel_at(PixelT, image, r, c);
//        ConnectivityPixelT connectivity = 0;
//
//#if CONNECTIVITY == 8
//        connectivity |= c > 0 && r > 0                     && isConnected(pixel, pixel_at(PixelT, image, r-1, c - 1)) ? LEFT_UP : 0
//        connectivity |= c > 0                              && isConnected(pixel, pixel_at(PixelT, image, r, c - 1)) ? LEFT : 0
//        connectivity |= c > 0 && r < im_rows - 1           && isConnected(pixel, pixel_at(PixelT, image, r+1, c - 1)) ? LEFT_DOWN : 0
//        connectivity |=          r < im_rows - 1           && isConnected(pixel, pixel_at(PixelT, image, r+1, c)) ? DOWN : 0
//        connectivity |= c < im_cols - 1 && r < im_rows - 1 && isConnected(pixel, pixel_at(PixelT, image, r+1, c + 1)) ? RIGHT_DOWN : 0
//        connectivity |= c < im_cols - 1                    && isConnected(pixel, pixel_at(PixelT, image, r, c + 1)) ? RIGHT : 0
//        connectivity |= c < im_cols - 1 && r > 0           && isConnected(pixel, pixel_at(PixelT, image, r-1, c + 1)) ? RIGHT_UP : 0
//        connectivity |=          r > 0                     && isConnected(pixel, pixel_at(PixelT, image, r-1, c)) ? UP : 0
//#else
//        connectivity |= c > 0                              && isConnected(pixel, pixel_at(PixelT, image, r, c - 1)) ? LEFT : 0
//        connectivity |=          r < im_rows - 1           && isConnected(pixel, pixel_at(PixelT, image, r+1, c)) ? DOWN : 0
//        connectivity |= c < im_cols - 1                    && isConnected(pixel, pixel_at(PixelT, image, r, c + 1)) ? RIGHT : 0
//        connectivity |=          r > 0                     && isConnected(pixel, pixel_at(PixelT, image, r-1, c)) ? UP : 0
//#endif
//
//        pixel_at(ConnectivityPixelT, connectivity, r, c) = connectivity;
//    }
//}

#define apron_pixel(apron, t_r, t_c) apron[(t_r+ 1)][(t_c + 1)]
//global dimensions: divUp(im_cols, tile_cols), divUp(im_rows, tile_rows);
__attribute__((reqd_work_group_size(WORKGROUP_TILE_SIZE_X, WORKGROUP_TILE_SIZE_Y, 1)))
__kernel void
make_connectivity_image(uint im_rows, uint im_cols, __global PixelT *image_p, uint image_pitch, __global ConnectivityPixelT *connectivityim_p, uint connectivityim_pitch){
    const uint tile_col_blocksize = get_local_size(0);
    const uint tile_row_blocksize = get_local_size(1);
    const uint tile_col_block = get_group_id(0) + get_global_offset(0) / tile_col_blocksize;
    const uint tile_row_block = get_group_id(1) + get_global_offset(1) / tile_row_blocksize;
    const uint tile_col = get_local_id(0);
    const uint tile_row = get_local_id(1);
    const bool valid_pixel_task = (get_global_id(0) < im_cols) & (get_global_id(1) < im_rows);

    uint tile_rows = tile_row_blocksize;
    uint tile_cols = tile_col_blocksize;

    const uint tile_row_start = tile_row_blocksize * tile_rows;
    const uint tile_col_start = tile_col_blocksize * tile_cols;
    const uint tile_row_end = min((tile_row_blocksize + 1) * tile_rows, (uint) im_rows);
    const uint tile_col_end = min((tile_col_blocksize + 1) * tile_cols, (uint) im_cols);
    //adjust to true tile dimensions
    tile_rows = tile_row_end - tile_row_start;
    tile_cols = tile_col_end - tile_col_start;
    const uint apron_tile_cols = tile_cols + 2;;
    //const uint n_tile_pixels = tile_rows * tile_cols;
    const uint n_work_items = get_local_size(0) * get_local_size(1);
    const uint n_apron_tile_pixels = (tile_rows + 2) * (apron_tile_cols);
    __local LDSPixelT im_tile[TILE_ROWS + 2][TILE_COLS + 2];
    __local LDSConnectivityPixelT connectivity_tile[TILE_ROWS][TILE_COLS];

    const uint tid = get_local_id(0) * get_local_size(0) + get_local_id(1);
    for(uint im_tile_fill_task_id = tid; im_tile_fill_task_id < n_apron_tile_pixels; im_tile_fill_task_id += n_work_items){
        const uint im_apron_tile_row = im_tile_fill_task_id / apron_tile_cols;
        const uint im_apron_tile_col = im_tile_fill_task_id % apron_tile_cols;
        const int g_c = ((int)(im_apron_tile_col + tile_col_block * tile_col_blocksize)) - 1;
        const int g_r = ((int)(im_apron_tile_row + tile_row_block * tile_row_blocksize)) - 1;

        im_tile[im_apron_tile_row][im_apron_tile_col] = image_tex2D(PixelT, image_p, (int) im_rows, (int) im_cols, image_pitch, g_r, g_c, ADDRESS_ZERO);
    }
    lds_barrier();

    #pragma unroll
    for (int i = 0; i < WORKITEM_REPEAT_Y; ++i){
        #pragma unroll
        for (int j = 0; j < WORKITEM_REPEAT_X; ++j){
            const uint c = get_local_id(0) + WORKGROUP_TILE_SIZE_X * j;
            const uint r = get_local_id(1) + WORKGROUP_TILE_SIZE_Y * i;
            const uint g_c = get_global_id(1);
            const uint g_r = get_global_id(0);
            PixelT pixel = apron_pixel(im_tile, r, c);
            ConnectivityPixelT connectivity = 0;

#if CONNECTIVITY == 8
            connectivity |= c > 0 && r > 0                         && isConnected(pixel, apron_pixel(im_tile, r-1, c - 1)) ? LEFT_UP : 0;
            connectivity |= c > 0                                  && isConnected(pixel, apron_pixel(im_tile, r  , c - 1)) ? LEFT : 0;
            connectivity |= c > 0 && r < tile_rows - 1             && isConnected(pixel, apron_pixel(im_tile, r+1, c - 1)) ? LEFT_DOWN : 0;
            connectivity |=          r < tile_rows - 1             && isConnected(pixel, apron_pixel(im_tile, r+1, c    )) ? DOWN : 0;
            connectivity |= c < tile_cols - 1 && r < tile_rows - 1 && isConnected(pixel, apron_pixel(im_tile, r+1, c + 1)) ? RIGHT_DOWN : 0;
            connectivity |= c < tile_cols - 1                      && isConnected(pixel, apron_pixel(im_tile, r  , c + 1)) ? RIGHT : 0;
            connectivity |= c < tile_cols - 1 && r > 0             && isConnected(pixel, apron_pixel(im_tile, r-1, c + 1)) ? RIGHT_UP : 0;
            connectivity |=          r > 0                         && isConnected(pixel, apron_pixel(im_tile, r-1, c    )) ? UP : 0;
#else
            connectivity |= c > 0                                  && isConnected(pixel, apron_pixel(im_tile, r  , c - 1)) ? LEFT : 0;
            connectivity |=          r < tile_rows - 1             && isConnected(pixel, apron_pixel(im_tile, r+1, c    )) ? DOWN : 0;
            connectivity |= c < tile_cols - 1                      && isConnected(pixel, apron_pixel(im_tile, r  , c + 1)) ? RIGHT : 0;
            connectivity |=          r > 0                         && isConnected(pixel, apron_pixel(im_tile, r-1, c    )) ? UP : 0;
#endif
            connectivity_tile[r][c] = connectivity;
        }
    }
    lds_barrier();

    for(uint im_tile_fill_task_id = tid; im_tile_fill_task_id < n_tile_pixels; im_tile_fill_task_id += n_work_items){
        const uint im_tile_row = im_tile_fill_task_id / tile_cols;
        const uint im_tile_col = im_tile_fill_task_id % tile_cols;
        const uint g_c = im_tile_col + tile_col_block * tile_col_blocksize;
        const uint g_r = im_tile_row + tile_row_block * tile_row_blocksize;

        pixel_at(ConnectivityPixelT, connectivityim, g_r, g_c) = connectivity_tile[im_tile_row][im_tile_col];
    }
}

__attribute__((reqd_work_group_size(WORKGROUP_TILE_SIZE_X, WORKGROUP_TILE_SIZE_Y, 1)))
__kernel void
label_tiles(uint im_rows, uint im_cols, __global ConnectivityPixelT *labelim_p, uint labelim_pitch, __global ConnectivityPixelT *connectivityim_p, uint connectivityim_pitch){
    const uint tile_col_start = get_local_id(0) + get_group_id(0) * TILE_COLS;
    const uint tile_row_start = get_local_id(1) + get_group_id(1) * TILE_ROWS;

    //if (x >= im_rows || y >= im_rows) return;

    __local LDSLabelT label_tile_im[TILE_ROWS][TILE_COLS];
    __local LDSConnectivityPixelT  edge_tile_im[TILE_ROWS][TILE_COLS];

    LDSLabelT new_labels[WORKITEM_REPEAT_Y][WORKITEM_REPEAT_X];
    LDSLabelT old_labels[WORKITEM_REPEAT_Y][WORKITEM_REPEAT_X];

    #pragma unroll
    for (int i = 0; i < WORKITEM_REPEAT_Y; ++i){
        #pragma unroll
        for (int j = 0; j < WORKITEM_REPEAT_X; ++j){
            const uint tile_row = get_local_id(1) + WORKGROUP_TILE_SIZE_Y * i;
            const uint tile_col = get_local_id(0) + WORKGROUP_TILE_SIZE_X * j;
            const bool valid_pixel_task = (tile_col < im_cols) & (tile_row < im_rows);
            ConnectivityPixelT c = valid_pixel_task ? pixel_at(ConnectivityPixelT, connectivityim, tile_row_start + WORKGROUP_TILE_SIZE_Y * i, tile_col_start + WORKGROUP_TILE_SIZE_X * j) : 0;

            c = tile_col == 0 ? c & ~(LEFT|LEFT_DOWN|LEFT_UP) : c;
            c = tile_row == 0 ? c & ~(UP|LEFT_UP|RIGHT_UP) : c;

            c = tile_col == TILE_COLS - 1 ? c & ~(RIGHT|RIGHT_DOWN|RIGHT_UP) : c;
            c = tile_row == TILE_ROWS - 1 ? c & ~(DOWN|LEFT_DOWN|RIGHT_DOWN) : c;

            new_labels[i][j] = tile_row * TILE_COLS + tile_col;
            edge_tile_im[tile_row][tile_col] = c;
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

                old_labels[i][j]          = new_labels[i][j];
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

                const ConnectivityPixelT connectivity = edge_tile_im[tile_row][tile_col];
                LDSLabelT label = new_labels[i][j];

#if CONNECTIVITY == 8
                if (connectivity & UP)
                   label = min(label, label_tile_im[tile_row - 1][tile_col]);
                if (connectivity & LEFT_UP)
                   label = min(label, label_tile_im[tile_row - 1][tile_col - 1]);
                if (connectivity & LEFT)
                   label = min(label, label_tile_im[tile_row][tile_col - 1]);
                if (connectivity &  LEFT_DOWN)
                   label = min(label, label_tile_im[tile_row + 1][tile_col - 1]);
                if (connectivity &  DOWN)
                   label = min(label, label_tile_im[tile_row + 1][tile_col]);
                if (connectivity & RIGHT_DOWN)
                   label = min(label, label_tile_im[tile_row - 1][tile_col + 1]);
                if (connectivity & RIGHT)
                   label = min(label, label_tile_im[tile_row][tile_col + 1]);
                if (connectivity & RIGHT_UP)
                   label = min(label, label_tile_im[tile_row + 1][tile_col + 1]);

#else
                if (connectivity & UP)
                   label = min(label, label_tile_im[tile_row - 1][tile_col]);
                if (connectivity & LEFT)
                   label = min(label, label_tile_im[tile_row][tile_col - 1]);
                if (connectivity &  DOWN)
                   label = min(label, label_tile_im[tile_row + 1][tile_col]);
                if (connectivity & RIGHT)
                   label = min(label, label_tile_im[tile_row][tile_col + 1]);
#endif

                new_labels[i][j] = label;
            }
        }
        lds_barrier();

        __local int changed;
        if((get_local_id(1) == 0) & (get_local_id(0) == 0)){
            changed = 0;
        }
        lds_barrier();

        __local LDSLabelT *labels = &label_tile_im[0][0];

        int pchanged = 0;
        #pragma unroll
        for(int i = 0; i < WORKITEM_REPEAT_Y; ++i){
            #pragma unroll
            for(int j = 0; j < WORKITEM_REPEAT_X; ++j){
                if(new_labels[i][j] < old_labels[i][j]){
                    pchanged++;
                    atomic_min(labels + old_labels[i][j], new_labels[i][j]);
                }
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
                LDSLabelT label = new_labels[i][j];

                //find root label
                while(labels[label] < label){
                    label = labels[label];
                }

                new_labels[i][j] = label;
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
            const uint g_r = (tile_label / TILE_COLS) + get_group_id(0) * TILE_COLS;
            const uint g_c = (tile_label % TILE_COLS) + get_group_id(1) * TILE_ROWS;

            //adjust to global offset and convert to scanline order again - this is globally unique
            const LabelT glabel = g_r * im_cols + g_c;
            const bool valid_pixel_task = (g_c < im_cols) & (g_r < im_rows);
            if(valid_pixel_task){
                pixel_at(LabelT, labelim, g_r, g_c) = glabel;
            }
        }
    }
}

inline
LabelT find_root_global(const __global LabelT *labelim_p, uint labelim_pitch, LabelT label, const uint im_rows, const uint im_cols){
    for(;;){
        const uint y = label / im_cols;
        const uint x = label % im_cols;
        const LabelT parent = pixel_at(LabelT, labelim, y, x);

        if(label == parent){
            break;
        }

        label = parent;
    }
    return label;
}

__kernel
void compact_paths_global(uint im_rows, uint im_cols, __global LabelT *labelim_p, uint labelim_pitch){
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if((x < im_cols) & (y < im_rows)){
        pixel_at(LabelT, labelim, y, x) = find_root_global(labelim_p, labelim_pitch, pixel_at(LabelT, labelim, y, x), im_rows, im_cols);
    }
}

inline
void merge_edge_labels(const uint im_rows, const uint im_cols, __global LabelT *labelim_p, const uint labelim_pitch, const LabelT l1, const LabelT l2, uint *changed){
    const LabelT r1 = find_root_global(labelim_p, labelim_pitch, l1, im_rows, im_cols);
    const LabelT r2 = find_root_global(labelim_p, labelim_pitch, l2, im_rows, im_cols);

    if(r1 == r2){
        return;
    }

    const LabelT mi = min(r1, r2);
    const LabelT ma = max(r1, r2);

    const uint y = ma / im_cols;
    const uint x = ma % im_cols;

    atomic_min(&pixel_at(LabelT, labelim, y, x), mi);
    *changed = true;
}

__kernel
void cross_merge(
    const uint n_vert_tiles, const uint n_horz_tiles,
    uint tile_rows, uint tile_cols,
    const uint im_rows, const uint im_cols,
    const __global ConnectivityPixelT *connectivityim_p, const uint connectivityim_pitch,
    __global LabelT *labelim_p, const uint labelim_pitch,
    const uint yIncomplete, uint xIncomplete
){
    const uint ngroups_x = get_num_groups(0);
    const uint ngroups_y = get_num_groups(1);

    const uint tid = get_local_linear_id();
    const uint stride = get_local_size(1) * get_local_size(0);

    const uint ybegin = get_group_id(1) * (n_vert_tiles * tile_rows);
    uint yend   = ybegin + n_vert_tiles * tile_rows;

    if (get_group_id(1) == ngroups_y - 1){
        yend -= yIncomplete * tile_rows;
        yend -= tile_rows;
        tile_rows = (im_rows % tile_rows);

        yend += tile_rows;
    }

    const uint xbegin = get_group_id(0) * n_horz_tiles * tile_cols;
    uint xend   = xbegin + n_horz_tiles * tile_cols;

    if(get_group_id(0) == ngroups_x - 1){
        if (xIncomplete) yend = ybegin;
        xend -= xIncomplete * tile_cols;
        xend -= tile_cols;
        tile_cols = (im_cols % tile_cols);

        xend += tile_cols;
    }

    if (get_group_id(1) == (ngroups_y - 1) && yIncomplete){
        xend = xbegin;
    }

    const uint tasksV = (n_horz_tiles - 1) * (yend - ybegin);
    const uint tasksH = (n_vert_tiles - 1) * (xend - xbegin);

    const uint total = tasksH + tasksV;

    __local uint changed;
    uint pchanged;
    do{
        pchanged = false;
        for(uint taskIdx = tid; taskIdx < total; taskIdx += stride){
            if(taskIdx < tasksH){
                const uint indexH = taskIdx;

                const uint row = indexH / (xend - xbegin);
                const uint col = indexH % (xend - xbegin);

                const uint y = ybegin + (row + 1) * tile_rows;
                const uint x = xbegin + col;

                const ConnectivityPixelT e = pixel_at(ConnectivityPixelT, connectivityim, y, x);
                if(e & UP){
                    const LabelT lc = pixel_at(LabelT, labelim, y, x);
                    const LabelT lu = pixel_at(LabelT, labelim, y - 1, x);
                    merge_edge_labels(im_rows, im_cols, labelim_p, labelim_pitch, lc, lu, &pchanged);
                }
            }else{
                const uint indexV = taskIdx - tasksH;

                const uint col = indexV / (yend - ybegin);
                const uint row = indexV % (yend - ybegin);

                const uint x = xbegin + (col + 1) * tile_cols;
                const uint y = ybegin + row;

                const ConnectivityPixelT e = pixel_at(ConnectivityPixelT, connectivityim, y, x);
                if(e & LEFT){
                    const LabelT lc = pixel_at(LabelT, labelim, y, x);
                    const LabelT lu = pixel_at(LabelT, labelim, y, x - 1);
                    merge_edge_labels(im_rows, im_cols, labelim_p, labelim_pitch, lc, lu, &pchanged);
                }
            }
        }
        atomic_add(&changed, pchanged);
        lds_barrier();
        pchanged = changed;
    }while(pchanged);
}

#if 0
__kernel
void mark_root_classes(
    uint im_rows, uint im_cols,
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
#endif

__kernel
void relabel_with_scanline_order(
    uint im_rows, uint im_cols,
    __global LabelT* labelim_out_p, const uint labelim_out_pitch,
    __global const PixelT* image_p, const uint image_pitch,
    __global const LabelT* labelim_p, const uint labelim_pitch,
    __global uint* scanline_prefix_sum_of_root_classes_p, const uint scanline_prefix_sum_of_root_classes_pitch
){
    const uint c = get_global_id(0);
    const uint r = get_global_id(1);
    const bool valid_pixel_task = (c < im_cols) & (r < im_rows);
    const uint linear_index = c + r * im_cols;

    if(valid_pixel_task){
        const LabelT pixel = pixel_at(PixelT, image, r, c);
        LabelT final_label = 0;
        if(pixel != BG_VALUE){
            const LabelT label = pixel_at(LabelT, labelim, r, c);
            const uint label_r = label / im_cols;
            const uint label_c = label % im_cols;
            const uint scan_id = pixel_at(uint, scanline_prefix_sum_of_root_classes, label_r, label_c);
            final_label = scan_id + 1;
        }
        pixel_at(LabelT, labelim, r, c) = final_label;
    }
}

//root class inclusive prefix sums belonging to each compute unit given to each tile - note if narray_workers == 1, no merge step is necessary
//computes local prefix sums to get intra-wg blocksums, prefix sum that to get intra-wg offsets - this is needed to merge the final blocksums
//global dims: <wgs_per_histogram, n_tiles>, work_dims: <wg_size, 1>
//global blocksums[divUp(nblocks, blocks_per_wg)]
__kernel
#ifdef PROMISE_WG_IS_WAVEFRONT
__attribute__((reqd_work_group_size(AMD_WAVEFRONT_SIZE, 1, 1)))
#endif
void mark_and_make_intra_wg_block_local_prefix_sums(uint im_rows, uint im_cols,
    __global PixelT *image_p, uint image_pitch,
    __global const LabelT* labelim_p, const uint labelim_pitch,
    __global const uint * restrict arrays_p, uint arrays_pitch,
    __global uint * restrict array_intra_wg_block_sums_p,
    __global uint * restrict array_prefix_sum_p
){
    const uint array_length = im_rows * im_cols;
    const uint array_wg_id = get_group_id(0);
    const uint narray_workers = get_num_groups(0);
    const uint wg_size = get_local_size(0);
    const uint block_size = wg_size;//efficient block size
    const uint nblocks = divUp(array_length, block_size);//number of efficiently processible blocks
    const uint nblocks_per_wg = nblocks / narray_workers;
    const uint nblocks_to_merge = nblocks / nblocks_per_wg;
    const uint nblocks_remainder = nblocks - (narray_workers * nblocks_per_wg);
    const uint nblocks_to_left = nblocks_per_wg * array_wg_id + (array_wg_id < nblocks_remainder ? array_wg_id : nblocks_remainder);
    const uint n_wg_blocks = nblocks_per_wg + (array_wg_id < nblocks_remainder ? 1 : 0);

    const uint start_bin = nblocks_to_left * block_size;
    const uint end_bin_ = start_bin + n_wg_blocks * block_size;//block aligned end

    uint inter_block_sum = 0;

    for(uint linear_index = get_local_id(0) + start_bin; linear_index < end_bin_; linear_index += wg_size){
        const uint r = linear_index / im_cols;
        const uint c = linear_index % im_cols;

        const uint linear_index = linear_index;
        uint count = 0;
        if(linear_index < array_length){
            const PixelT pixel = pixel_at(PixelT, image, r, c);
            const LabelT label = pixel_at(LabelT, labelim, r, c);
            count = (pixel != BG_VALUE) & (label == linear_index);
        }

#ifdef USE_CL2_WORKGROUP_FUNCTIONS
        uint block_prefix_sum_inclusive = work_group_scan_inclusive_add(linear_index_count);
#else
        __local uint lmem[WORK_GROUP_FUNCTION_MAX_MEMORY_SIZE];
        uint block_prefix_sum_inclusive = clc_work_group_scan_inclusive_add_uint(linear_index_count, lmem);
#endif

        block_prefix_sum_inclusive += inter_block_sum;
        if(linear_index < array_length){
            array_prefix_sum_p[linear_index] = block_prefix_sum_inclusive;
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
__kernel
#ifdef PROMISE_WG_IS_WAVEFRONT
__attribute__((reqd_work_group_size(AMD_WAVEFRONT_SIZE, 1, 1)))
#endif
void make_intra_wg_block_global_sums(
    __global uint * restrict intra_wg_block_sums_p, uint nblocks_to_merge
){
    const uint n_arrays = get_num_groups(1);
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
            image_pixel_at(uint, intra_wg_block_sums_p, n_arrays, nblocks_to_merge, intra_wg_block_sums_pitch, array_id, intra_wg_block_id) = intra_wg_block_offset;
        }
#ifdef USE_CL2_WORKGROUP_FUNCTIONS
        intra_wg_block_offset = work_group_broadcast(intra_wg_block_offset, wg_size - 1);
#else
        __local uint value;
        intra_wg_block_offset = clc_work_group_broadcast1_uint(intra_wg_block_offset, wg_size - 1, &value);
#endif

        if(get_local_id(0) == 0){
            intra_wg_block_sums[0] = intra_wg_block_sums[wg_size];
        }
        lds_barrier();
    }
}

//merges global offsets of intra-wg-block offsets of prefix sums
//global dims: <wgs_per_sum>, work_dims: <wg_size> : wg_size >= nblocks_to_merge
//global array_of_prefix_sums[im_rows*im_cols] : as input partial sums, as output full prefix sum
__kernel
#ifdef PROMISE_WG_IS_WAVEFRONT
__attribute__((reqd_work_group_size(AMD_WAVEFRONT_SIZE, 1, 1)))
#endif
void make_prefix_sums_with_intra_wg_block_global_sums(
        const uint im_rows, const uint im_cols,
        __global const uint * restrict intra_wg_block_sums_p, uint intra_wg_block_sums_pitch,
        __global uint * restrict array_of_prefix_sums_p, uint array_of_prefix_sums_pitch //input -> partial/local prefix sums, output: global prefix sums
){
    const uint array_length = im_rows * im_cols;
    const uint n_arrays = get_num_groups(1);
    const uint array_id = get_group_id(1);
    const uint array_wg_id = get_group_id(0);
    const uint narray_workers = get_num_groups(0);
    const uint wg_size = get_local_size(0);
    const uint block_size = wg_size;//efficient block size
    const uint nblocks = divUp(array_length, block_size);//number of efficiently processible blocks
    const uint nblocks_per_wg = nblocks / narray_workers;
    const uint nblocks_to_merge = nblocks / nblocks_per_wg;
    const uint nblocks_remainder = nblocks - (narray_workers * nblocks_per_wg);
    const uint nblocks_to_left = nblocks_per_wg * array_wg_id + (array_wg_id < nblocks_remainder ? array_wg_id : nblocks_remainder);
    const uint n_wg_blocks = nblocks_per_wg + (array_wg_id < nblocks_remainder ? 1 : 0);

    const uint start_bin = nblocks_to_left * block_size;
    const uint end_bin_ = start_bin + n_wg_blocks * block_size;//block aligned end

    const uint inter_block_sum = intra_wg_block_sums_p[array_wg_id];
    for(uint array_index = get_local_id(0) + start_bin; array_index < end_bin_; array_index += wg_size){
        if(array_index < array_length){
            const uint g_r = array_index / im_cols;
            const uint g_c = array_index % im_cols;
            image_pixel_at(uint, array_of_prefix_sums_p, im_rows, im_cols, array_of_prefix_sums_pitch, g_r, g_c) += inter_block_sum;
        }
    }
}
