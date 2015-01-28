#include "clcommons/common.h"
#include "clcommons/image.h"

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

//global dimensions: divUp(im_cols, workgroup_tile_size_x), divUp(im_rows, workgroup_tile_size_y);
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
    const uint n_tile_pixels = tile_rows * tile_cols;
    __local LDSPixelT im_tile[WORKGROUP_TILE_SIZE_Y][WORKGROUP_TILE_SIZE_X];
    __local ConnectivityPixelT connectivity_tile[WORKGROUP_TILE_SIZE_Y][WORKGROUP_TILE_SIZE_X];

    const uint tid = tile_row * tile_cols + tile_col;
    for(uint im_tile_fill_task_id = tid; im_tile_fill_task_id < n_tile_pixels; im_tile_fill_task_id += n_tile_pixels){
        const uint im_tile_row = im_tile_fill_task_id / tile_cols;
        const uint im_tile_col = im_tile_fill_task_id % tile_cols;
        const uint g_c = im_tile_col + tile_col_block * tile_col_blocksize;
        const uint g_r = im_tile_row + tile_row_block * tile_row_blocksize;

        im_tile[im_tile_row][im_tile_col] = pixel_at(PixelT, image, g_r, g_c);
    }
    lds_barrier();

    {
        const uint c = get_local_id(1);
        const uint r = get_local_id(0);
        const uint g_c = get_global_id(1);
        const uint g_r = get_global_id(0);
        PixelT pixel = valid_pixel_task ? pixel_at(PixelT, image, r, c) : 0;
        ConnectivityPixelT connectivity = 0;

#if CONNECTIVITY == 8
        connectivity |= c > 0 && r > 0                         && isConnected(pixel, im_tile[r-1][c - 1]) ? LEFT_UP : 0;
        connectivity |= c > 0                                  && isConnected(pixel, im_tile[r][c - 1]  ) ? LEFT : 0;
        connectivity |= c > 0 && r < tile_rows - 1             && isConnected(pixel, im_tile[r+1][c - 1]) ? LEFT_DOWN : 0;
        connectivity |=          r < tile_rows - 1             && isConnected(pixel, im_tile[r+1][c]    ) ? DOWN : 0;
        connectivity |= c < tile_cols - 1 && r < tile_rows - 1 && isConnected(pixel, im_tile[r+1][c + 1]) ? RIGHT_DOWN : 0;
        connectivity |= c < tile_cols - 1                      && isConnected(pixel, im_tile[r][c + 1]  ) ? RIGHT : 0;
        connectivity |= c < tile_cols - 1 && r > 0             && isConnected(pixel, im_tile[r-1][c + 1]) ? RIGHT_UP : 0;
        connectivity |=          r > 0                         && isConnected(pixel, im_tile[r-1][c]    ) ? UP : 0;
#else
        connectivity |= c > 0                                  && isConnected(pixel, im_tile[r][c - 1]) ? LEFT : 0;
        connectivity |=          r < tile_rows - 1             && isConnected(pixel, im_tile[r+1][c]  ) ? DOWN : 0;
        connectivity |= c < tile_cols - 1                      && isConnected(pixel, im_tile[r][c + 1]) ? RIGHT : 0;
        connectivity |=          r > 0                         && isConnected(pixel, im_tile[r-1][c]  ) ? UP : 0;
#endif
        connectivity_tile[r][c] = connectivity;

    }
    lds_barrier();

    for(uint im_tile_fill_task_id = tid; im_tile_fill_task_id < n_tile_pixels; im_tile_fill_task_id += n_tile_pixels){
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
    const uint x = get_local_id(0) + get_group_id(0) * TILE_COLS;
    const uint y = get_local_id(1) + get_group_id(1) * TILE_ROWS;

    //if (x >= im_rows || y >= im_rows) return;

    //currently x is 1
    //int bounds = ((y + WORKITEM_REPEAT_Y) < im_rows);

    __local LDSLabelT label_tile_im[TILE_ROWS][TILE_COLS];
    __local LDSConnectivityPixelT  edge_tile_im[TILE_ROWS][TILE_COLS];

    LDSLabelT new_labels[WORKITEM_REPEAT_Y][WORKITEM_REPEAT_X];
    LDSLabelT old_labels[WORKITEM_REPEAT_Y][WORKITEM_REPEAT_X];

    #pragma unroll
    for (int i = 0; i < WORKITEM_REPEAT_Y; ++i){
        #pragma unroll
        for (int j = 0; j < WORKITEM_REPEAT_X; ++j){
            const uint yloc = get_local_id(1) + WORKGROUP_TILE_SIZE_Y * i;
            const uint xloc = get_local_id(0) + WORKGROUP_TILE_SIZE_X * j;
            const bool valid_pixel_task = (xloc < im_cols) & (yloc < im_rows);
            ConnectivityPixelT c = valid_pixel_task ? pixel_at(ConnectivityPixelT, connectivityim, y + WORKGROUP_TILE_SIZE_Y * i, x + WORKGROUP_TILE_SIZE_X * j) : 0;

            c = xloc == 0 ? c & ~(LEFT|LEFT_DOWN|LEFT_UP) : c;
            c = yloc == 0 ? c & ~(UP|LEFT_UP|RIGHT_UP) : c;

            c = xloc == TILE_COLS - 1 ? c & ~(RIGHT|RIGHT_DOWN|RIGHT_UP) : c;
            c = yloc == TILE_ROWS - 1 ? c & ~(DOWN|LEFT_DOWN|RIGHT_DOWN) : c;

            new_labels[i][j] = yloc * TILE_COLS + xloc;
            edge_tile_im[yloc][xloc] = c;
        }
    }

    for (int k = 0; ;++k){
        //make copies
        #pragma unroll
        for (int i = 0; i < WORKITEM_REPEAT_Y; ++i){
            #pragma unroll
            for (int j = 0; j < WORKITEM_REPEAT_X; ++j){
                const uint yloc = get_local_id(1) + WORKGROUP_TILE_SIZE_Y * i;
                const uint xloc = get_local_id(0) + WORKGROUP_TILE_SIZE_X * j;

                old_labels[i][j]          = new_labels[i][j];
                label_tile_im[yloc][xloc] = new_labels[i][j];
            }
        }
        lds_barrier();

        //take minimum label of local neighboorhood - single writer, multi reader version
        #pragma unroll
        for (int i = 0; i < WORKITEM_REPEAT_Y; ++i){
            #pragma unroll
            for (int j = 0; j < WORKITEM_REPEAT_X; ++j){
                const uint yloc = get_local_id(1) + WORKGROUP_TILE_SIZE_Y * i;
                const uint xloc = get_local_id(0) + WORKGROUP_TILE_SIZE_X * j;

                const ConnectivityPixelT connectivity = edge_tile_im[yloc][xloc];
                LDSLabelT label = new_labels[i][j];

#if CONNECTIVITY == 8
                if (connectivity & UP)
                   label = min(label, label_tile_im[yloc - 1][xloc]);
                if (connectivity & LEFT)
                   label = min(label, label_tile_im[yloc][xloc - 1]);
                if (connectivity & LEFT_UP)
                   label = min(label, label_tile_im[yloc - 1][xloc - 1]);
                if (connectivity &  LEFT_DOWN)
                   label = min(label, label_tile_im[yloc + 1][xloc - 1]);
                if (connectivity &  DOWN)
                   label = min(label, label_tile_im[yloc + 1][xloc]);
                if (connectivity & RIGHT_DOWN)
                   label = min(label, label_tile_im[yloc - 1][xloc + 1]);
                if (connectivity & RIGHT)
                   label = min(label, label_tile_im[yloc][xloc + 1]);
                if (connectivity & RIGHT_UP)
                   label = min(label, label_tile_im[yloc + 1][xloc + 1]);

#else
                if (connectivity & UP)
                   label = min(label, label_tile_im[yloc - 1][xloc]);
                if (connectivity &  DOWN)
                   label = min(label, label_tile_im[yloc + 1][xloc]);
                if (connectivity & LEFT)
                   label = min(label, label_tile_im[yloc][xloc - 1]);
                if (connectivity & RIGHT)
                   label = min(label, label_tile_im[yloc][xloc + 1]);
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
            const uint yloc = (tile_label / TILE_COLS) + get_group_id(0) * TILE_COLS;
            const uint xloc = (tile_label % TILE_COLS) + get_group_id(1) * TILE_ROWS;

            const LabelT glabel = yloc * im_cols + xloc;
            const bool valid_pixel_task = (xloc < im_cols) & (yloc < im_rows);
            if(valid_pixel_task){
                pixel_at(LabelT, labelim, yloc, xloc) = glabel;
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
