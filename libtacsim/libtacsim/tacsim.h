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


typedef struct _MatrixDouble {
    int w, h;
    double** m;
} MatrixDouble;

typedef struct _VectorDouble {
    int l;
    double* v;
} VectorDouble;

MatrixInt* allocate_matrix_int(int h, int w, int fill, int defv);
MatrixDouble* allocate_matrix_double(int h, int w, int fill, double defv);
VectorDouble* allocate_vector_double(int l, int fill, double defv);

void free_matrix_int(MatrixInt *mat);
void free_matrix_double(MatrixDouble *mat);
void free_vector_double(VectorDouble *vec);

double strength_node(double nw1, double nw2, double ew);
double strength_edge(double ew1, double ew2, double nw);
double strength_coherence(double s1, double s2);

int normalize_vector(double **vec, int len);
int normalize_matrix(double ***mat, int m, int n);

int get_edge_count(MatrixInt **node_adjacency);
MatrixInt* get_edge_adjacency(MatrixInt **node_adjacency, int elen);
int is_converged(MatrixDouble **simmat, MatrixDouble **simmat_prev, double eps);
int copyTo(MatrixDouble **simmat, MatrixDouble **simmat_prev);

int graph_elements(MatrixInt *nnadj, VectorDouble *node_weights, MatrixDouble **nn_strength_mat,
                   MatrixInt *eeadj, VectorDouble *edge_weights, MatrixDouble **ee_strength_mat);

int tacsim(MatrixInt *g1_nnadj, MatrixInt *g1_eeadj, MatrixDouble *g1_nn_strength_mat, MatrixDouble *g1_ee_strength_mat,
           MatrixInt *g2_nnadj, MatrixInt *g2_eeadj, MatrixDouble *g2_nn_strength_mat, MatrixDouble *g2_ee_strength_mat,
           MatrixDouble **nn_simmat, MatrixDouble **ee_simmat, int max_iter, double eps, double tolerance);

int calculate_tacsim(int **A, double *Anw, double *Aew, int Anode, int Aedge,
                     int **B, double *Bnw, double *Bew, int Bnode, int Bedge,
                     double ***nsim, double ***esim,
                     int max_iter, double eps, double tol);

int calculate_tacsim_self(int **A, double *Anw, double *Aew, int Anode, int Aedge,
                          double ***nsim, double ***esim,
                          int max_iter, double eps, double tol);

#endif /* tacsim_h */
