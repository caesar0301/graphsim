//
//  tacsim.h
//  libtacsim
//
//  Created by Xiaming Chen on 12/18/15.
//  Copyright © 2015 Xiaming Chen. All rights reserved.
//

#ifndef tacsim_h
#define tacsim_h

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define USE_FLOATS 1
#define USE_SHORTS 1

#ifdef USE_SHORTS
typedef short INT;
#else
typedef int INT;
#endif

#ifdef USE_FLOATS
typedef float REAL;
#else
typedef double REAL;
#endif

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

typedef struct _MatrixInt {
    int w, h;
    int** m;
} MatrixInt;


typedef struct _MatrixReal {
    int w, h;
    REAL** m;
} MatrixReal;

typedef struct _VectorReal {
    int l;
    REAL* v;
} VectorReal;

MatrixInt* allocate_matrix_int(int h, int w, int fill, int defv);
MatrixReal* allocate_matrix_real(int h, int w, int fill, REAL defv);
VectorReal* allocate_vector_real(int l, int fill, REAL defv);

void free_matrix_int(MatrixInt *mat);
void free_matrix_real(MatrixReal *mat);
void free_vector_real(VectorReal *vec);

REAL strength_node(REAL nw1, REAL nw2, REAL ew);
REAL strength_edge(REAL ew1, REAL ew2, REAL nw);
REAL strength_coherence(REAL s1, REAL s2);

int normalize_vector(REAL **vec, int len);
int normalize_matrix(REAL ***mat, int m, int n);

int get_edge_count(MatrixInt **node_adjacency);
MatrixInt* get_edge_adjacency(MatrixInt **node_adjacency, int elen);
int is_converged(MatrixReal **simmat, MatrixReal **simmat_prev, REAL eps);
int copyTo(MatrixReal **simmat, MatrixReal **simmat_prev);

int graph_elements(MatrixInt *nnadj, VectorReal *node_weights, MatrixReal **nn_strength_mat,
                   MatrixInt *eeadj, VectorReal *edge_weights, MatrixReal **ee_strength_mat);

int tacsim(MatrixInt *g1_nnadj, MatrixInt *g1_eeadj, MatrixReal *g1_nn_strength_mat, MatrixReal *g1_ee_strength_mat,
           MatrixInt *g2_nnadj, MatrixInt *g2_eeadj, MatrixReal *g2_nn_strength_mat, MatrixReal *g2_ee_strength_mat,
           MatrixReal **nn_simmat, MatrixReal **ee_simmat, int max_iter, REAL eps, REAL tolerance);

int calculate_tacsim(int **A, REAL *Anw, REAL *Aew, int Anode, int Aedge,
                     int **B, REAL *Bnw, REAL *Bew, int Bnode, int Bedge,
                     REAL ***nsim, REAL ***esim,
                     int max_iter, REAL eps, REAL tol);

int calculate_tacsim_self(int **A, REAL *Anw, REAL *Aew, int Anode, int Aedge,
                          REAL ***nsim, REAL ***esim,
                          int max_iter, REAL eps, REAL tol);

#endif /* tacsim_h */
