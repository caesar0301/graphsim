//
//  tacsim.c
//  libtacsim
//
//  Created by Xiaming Chen on 12/18/15.
//  Copyright © 2015 Xiaming Chen. All rights reserved.
//
#include "tacsim.h"

/**
 * Allocate a 2D int array with h rows and w columns
 */
MatrixInt* allocate_matrix_int(int h, int w, int fill, int defv){
    MatrixInt *mat = malloc(sizeof(MatrixInt));
    mat->h = h;
    mat->w = w;
    mat->m = NULL;
    
    if (fill == TRUE) {
        int **arr;
        arr = (int**) malloc(h * sizeof(int*));
        for (int i = 0; i < h; i++) {
            arr[i] = (int*) malloc(w * sizeof(int));
            for (int j = 0; j < w; j++) {
                arr[i][j] = defv;
            }
        }
        
        mat->m = arr;
    }
    
    return mat;
}

/**
 * Allocate a 2D double array with h rows and w columns
 */
MatrixDouble* allocate_matrix_double(int h, int w, int fill, double defv){
    MatrixDouble *mat = malloc(sizeof(MatrixDouble));
    mat->h = h;
    mat->w = w;
    mat->m = NULL;
    
    if (fill == TRUE) {
        double **arr;
        arr = (double**) malloc(h * sizeof(double*));
        for (int i = 0; i < h; i++) {
            arr[i] = (double*) malloc(w * sizeof(double));
            for (int j = 0; j < w; j++) {
                arr[i][j] = defv;
            }
        }
        mat->m = arr;
    }
    
    return mat;
}

/**
 * Allocate a 1D double array with l elements.
 */
VectorDouble* allocate_vector_double(int l, int fill, double defv){
    VectorDouble *vec = malloc(sizeof(VectorDouble));
    vec->l = l;
    vec->v = NULL;
    
    if (fill == TRUE) {
        vec->v = malloc(vec->l * sizeof(double));
        for (int i=0; i < vec->l; i++) {
            vec->v[i] = defv;
        }
    }
    
    return vec;
}

/**
 * Free struct MatrixInt
 */
void free_matrix_int(MatrixInt *mat) {
    for (int i = 0; i < mat->h; i++) {
        if (mat->m[i] != NULL) {
            free(mat->m[i]);
            mat->m[i] = NULL;
        }
    }
    if (mat->m != NULL){
        free(mat->m);
        mat->m = NULL;
    }
    free(mat);
}

/**
 * Free struct MatrixDouble
 */
void free_matrix_double(MatrixDouble *mat) {
    for (int i = 0; i < mat->h; i++) {
        if (mat->m[i] != NULL) {
            free(mat->m[i]);
            mat->m[i] = NULL;
        }
    }
    if (mat->m != NULL){
        free(mat->m);
        mat->m = NULL;
    }
    free(mat);
}

/**
 * Free struct VectorDouble
 */
void free_vector_double(VectorDouble *vec) {
    if (vec->v != NULL) {
        free(vec->v);
        vec->v = NULL;
    }
    free(vec);
}

/**
 * Calculat the strength of neighboring nodes.
 */
double strength_node(double nw1, double nw2, double ew) {
    return 1.0 * nw1 * nw2 / pow(ew, 2);
}

/**
 * Calculat the strength of neighboring edges.
 */
double strength_edge(double ew1, double ew2, double nw) {
    return 1.0 * pow(nw, 2) / (ew1 * ew2);
}

/**
 * Calculat the strength coherence of two neighbor pairs.
 */
double strength_coherence(double s1, double s2) {
    if (s1 + s2 == 0) {
        printf("Invalid strength values: s1=%f, s2=%f\n", s1, s2);
        exit(-1);
    }
    
    return 2.0 * sqrt(s1 * s2) / (s1 + s2);
}

/**
 * Normalize a 1D array.
 */
int normalize_vector(double **vec, int len){
    double sum = 0;
    for (int i = 0; i < len; i++) {
        sum += pow((*vec)[i], 2);
    }
    double norm = sqrt(sum);
    for (int i = 0; i < len; i++) {
        (*vec)[i] /= norm;
    }
    return 0;
}

/**
 * Normalize a 2D array.
 */
int normalize_matrix(double ***mat, int m, int n){
    double sum = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            sum += pow((*mat)[i][j], 2);
        }
    }
    double norm = sqrt(sum);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            (*mat)[i][j] /= norm;
        }
    }
    return 0;
}

/**
 * Get the number of edges from node adjacency matrix.
 */
int get_edge_count(MatrixInt **node_adjacency){
    MatrixInt *nnadj = *node_adjacency;
    int elen = 0;
    for (int i = 0; i < nnadj->h; i++)
        for (int j = 0; j < nnadj->h; j++)
            if (nnadj->m[i][j] >= 0)
                elen += 1;
    return elen;
}


/**
 * Derive the source-terminal edge neighbors for each node.
 */
MatrixInt* get_edge_adjacency(MatrixInt **node_adjacency, int elen){
    MatrixInt *nnadj = *node_adjacency;
    MatrixInt *eeadj = allocate_matrix_int(elen, elen, TRUE, -1);
    
    // Assert h == w
    for (int i=0; i < nnadj->h; i++) {
        for (int j = 0; j < nnadj->w; j++) {
            if (nnadj->m[i][j] >= 0) {
                for (int k = 0; k < nnadj->w; k++) {
                    if (nnadj->m[j][k] >= 0) {
                        int src = nnadj->m[i][j];
                        int dst = nnadj->m[j][k];
                        eeadj->m[src][dst] = j;
                    }
                }
            }
        }
    }
    
    return eeadj;
}

/**
 * Check if two iterations are converged.
 */
int is_converged(MatrixDouble **simmat, MatrixDouble **simmat_prev, double eps) {
    MatrixDouble *sim = *simmat;
    MatrixDouble *sim_prev = *simmat_prev;
    
    for (int i = 0; i < sim->h; i++) {
        for (int j = 0; j < sim->w; j++) {
            if (fabs(sim->m[i][j] - sim_prev->m[i][j]) > eps)
                return 1;
        }
    }
    return 0;
}

/**
 * Copy the similarity from one matrix to another.
 */
int copyTo(MatrixDouble **simmat, MatrixDouble **simmat_prev) {
    MatrixDouble *sim = *simmat;
    MatrixDouble *sim_prev = *simmat_prev;
    
    for (int i = 0; i < sim->h; i++) {
        for (int j = 0; j < sim->w; j++) {
            sim_prev->m[i][j] = sim->m[i][j];
        }
    }
    
    return 0;
}

/**
 * Calculate the graph elements employed by the algorithm.
 */
int graph_elements(MatrixInt *nnadj, VectorDouble *node_weights, MatrixDouble **nn_strength_mat,
                   MatrixInt *eeadj, VectorDouble *edge_weights, MatrixDouble **ee_strength_mat) {

    if (nnadj->h != nnadj->w || eeadj->h != eeadj->w) {
        printf("The adjacent matrix should be square.\n");
        exit(-1);
    }
    
    if (node_weights->l != nnadj->h) {
        printf("The node weight vector should be the same length as nnadj.\n");
        exit(-1);
    }
    
    if (edge_weights->l != eeadj->h) {
        printf("The edge weight vector should be the same length as eeadj.\n");
        exit(-1);
    }
    
    MatrixDouble *nnsm = *nn_strength_mat;
    MatrixDouble *eesm = *ee_strength_mat;
    
    for (int i = 0; i < nnadj->h; i++) {
        for (int j = 0; j < nnadj->h; j++) {
            int edge_index = nnadj->m[i][j];
            if(edge_index >= 0) {
                nnsm->m[i][j] = strength_node(node_weights->v[i], node_weights->v[j], edge_weights->v[edge_index]);
            }
        }
    }
    
    for (int i = 0; i < eeadj->h; i++) {
        for (int j = 0; j < eeadj->h; j++) {
            int node_index = eeadj->m[i][j];
            if(node_index >= 0) {
                eesm->m[i][j] = strength_edge(edge_weights->v[i], edge_weights->v[j], node_weights->v[node_index]);
            }
        }
    }
    
    return 0;
}


/**
 * Set the value smaller than tolerance to zero.
 */
void mask_lower_values(MatrixDouble **simmat, double tolerance) {
    MatrixDouble *sim = *simmat;
    for (int i = 0; i < sim->h; i++){
        for (int j = 0; j < sim->w; j++){
            if ( fabs(sim->m[i][j]) < tolerance)
                sim->m[i][j] = 0;
        }
    }
}

/**
 * The algorithm to calculate the topology-attribute coupling similarity for two graphs.
 */
int tacsim(MatrixInt *g1_nnadj, MatrixInt *g1_eeadj, MatrixDouble *g1_nn_strength_mat, MatrixDouble *g1_ee_strength_mat,
           MatrixInt *g2_nnadj, MatrixInt *g2_eeadj, MatrixDouble *g2_nn_strength_mat, MatrixDouble *g2_ee_strength_mat,
           MatrixDouble **nn_simmat, MatrixDouble **ee_simmat, int max_iter, double eps, double tolerance) {
    
    MatrixDouble *nn_sim = *nn_simmat;
    MatrixDouble *ee_sim = *ee_simmat;
    MatrixDouble *nn_sim_prev = allocate_matrix_double(nn_sim->h, nn_sim->w, TRUE, 0);
    MatrixDouble *ee_sim_prev = allocate_matrix_double(ee_sim->h, ee_sim->w, TRUE, 0);
    
    int iter = 0;
    for (; iter < max_iter; iter++) {
        
        if (is_converged(&nn_sim, &nn_sim_prev, eps) == 0 &&
            is_converged(&ee_sim, &ee_sim_prev, eps) == 0)
            break;
        
        copyTo(&nn_sim, &nn_sim_prev);
        copyTo(&ee_sim, &ee_sim_prev);
        
        // Update node similarity, in and out node neighbors
        int N = nn_sim->h;
        int M = nn_sim->w;
        
        for (int i = 0; i < N; i++){
            for (int j = 0; j < M; j++) {
                //TODO: optimize the neighbor-pair finding
                
                // In neighbors
                for (int u = 0; u < N; u++){
                    double sui = g1_nn_strength_mat->m[u][i];
                    if (sui > 0) {
                        for (int v = 0; v < M; v++){
                            double svj = g2_nn_strength_mat->m[v][j];
                            if (svj > 0) {
                                int u_edge = g1_nnadj->m[u][i];
                                int v_edge = g2_nnadj->m[v][j];
                                
                                nn_sim->m[i][j] = nn_sim->m[i][j] + 0.5 \
                                * strength_coherence(sui, svj) \
                                * (nn_sim_prev->m[u][v] + ee_sim_prev->m[u_edge][v_edge]);
                            }
                        }
                    }
                }
                
                // Out neighbors
                for (int u = 0; u < N; u++){
                    double siu = g1_nn_strength_mat->m[i][u];
                    if (siu > 0) {
                        for (int v = 0; v < M; v++){
                            double sjv = g2_nn_strength_mat->m[j][v];
                            if (sjv > 0) {
                                int u_edge = g1_nnadj->m[i][u];
                                int v_edge = g2_nnadj->m[j][v];
                                
                                nn_sim->m[i][j] += 0.5 \
                                * strength_coherence(siu, sjv) \
                                * (nn_sim_prev->m[u][v] + ee_sim_prev->m[u_edge][v_edge]);
                            }
                        }
                    }
                }
            }
        }
        
        // Update edge similarity, in and out edge neighbors
        int P = ee_sim->h;
        int Q = ee_sim->w;
        
        for (int i = 0; i < P; i++){
            for (int j = 0; j < Q; j++) {
                // In neighbors
                for (int u = 0; u < P; u++){
                    double sui = g1_ee_strength_mat->m[u][i];
                    if (sui > 0) {
                        for (int v = 0; v < Q; v++){
                            double svj = g2_ee_strength_mat->m[v][j];
                            if (svj > 0) {
                                int u_node = g1_eeadj->m[u][i];
                                int v_node = g2_eeadj->m[v][j];
                                
                                ee_sim->m[i][j] += 0.5 \
                                * strength_coherence(sui, svj) \
                                * (ee_sim_prev->m[u][v] + nn_sim_prev->m[u_node][v_node]);
                            }
                        }
                    }
                }
                
                // Out neighbors
                for (int u = 0; u < P; u++){
                    double siu = g1_ee_strength_mat->m[i][u];
                    if (siu > 0) {
                        for (int v = 0; v < Q; v++){
                            double sjv = g2_ee_strength_mat->m[j][v];
                            if (sjv > 0) {
                                int u_node = g1_eeadj->m[i][u];
                                int v_node = g2_eeadj->m[j][v];
                                
                                ee_sim->m[i][j] += 0.5 \
                                * strength_coherence(siu, sjv) \
                                * (ee_sim_prev->m[u][v] + nn_sim_prev->m[u_node][v_node]);
                            }
                        }
                    }
                }
            }
        }
        
        // Normalize matrices before entering next iteration
        normalize_matrix(&(nn_sim->m), nn_sim->h, nn_sim->w);
        normalize_matrix(&(ee_sim->m), ee_sim->h, ee_sim->w);
        
    }
    
    mask_lower_values(&nn_sim, tolerance);
    mask_lower_values(&ee_sim, tolerance);
    
    free_matrix_double(nn_sim_prev);
    free_matrix_double(ee_sim_prev);
    
    printf("Converge after %d iterations (eps=%f).\n", iter, eps);
    
    return 0;
}

/**
 * Export interface of tacsim algorithm.
 */
int calculate_tacsim(int **A, double *Anw, double *Aew, int Anode, int Aedge,
                     int **B, double *Bnw, double *Bew, int Bnode, int Bedge,
                     double ***nsim, double ***esim,
                     int max_iter, double eps, double tol) {

    // create a new graph
    MatrixInt *graph = allocate_matrix_int(Anode, Anode, FALSE, -1);
    MatrixInt *graph_eeadj;
    VectorDouble *node_weights = allocate_vector_double(Anode, FALSE, -1);
    VectorDouble *edge_weights = allocate_vector_double(Aedge, FALSE, -1);
    MatrixDouble *nn_strength_mat = allocate_matrix_double(Anode, Anode, TRUE, -1);
    MatrixDouble *ee_strength_mat = allocate_matrix_double(Aedge, Aedge, TRUE, -1);
    
    // create another graph
    MatrixInt *graph2 = allocate_matrix_int(Bnode, Bnode, FALSE, -1);
    MatrixInt *graph_eeadj2;
    VectorDouble *node_weights2 = allocate_vector_double(Bnode, FALSE, -1);
    VectorDouble *edge_weights2 = allocate_vector_double(Bedge, FALSE, -1);
    MatrixDouble *nn_strength_mat2 = allocate_matrix_double(Bnode, Bnode, TRUE, -1);
    MatrixDouble *ee_strength_mat2 = allocate_matrix_double(Bedge, Bedge, TRUE, -1);
    
    MatrixDouble *nn_sim = allocate_matrix_double(Anode, Bnode, TRUE, 1);
    MatrixDouble *ee_sim = allocate_matrix_double(Aedge, Bedge, TRUE, 1);
    
    graph->m = A;
    node_weights->v = Anw;
    edge_weights->v = Aew;
    
    graph2->m = B;
    node_weights2->v = Bnw;
    edge_weights2->v = Bew;
    
    normalize_vector(&(node_weights->v), node_weights->l);
    normalize_vector(&(edge_weights->v), edge_weights->l);
    
    normalize_vector(&(node_weights2->v), node_weights2->l);
    normalize_vector(&(edge_weights2->v), edge_weights2->l);
    
    graph_eeadj = get_edge_adjacency(&graph, Aedge);
    graph_eeadj2 = get_edge_adjacency(&graph2, Bedge);
    
    graph_elements(graph, node_weights, &nn_strength_mat, graph_eeadj, edge_weights, &ee_strength_mat);
    graph_elements(graph2, node_weights2, &nn_strength_mat2, graph_eeadj2, edge_weights2, &ee_strength_mat2);
    
    tacsim(graph, graph_eeadj, nn_strength_mat, ee_strength_mat,
           graph2, graph_eeadj2, nn_strength_mat2, ee_strength_mat2,
           &nn_sim, &ee_sim, max_iter, eps, tol);
    
    // We should not free the accepted memory
//    free_vector_double(node_weights);
//    free_vector_double(edge_weights);
//    free_vector_double(node_weights2);
//    free_vector_double(edge_weights2);
//    free_matrix_int(graph);
//    free_matrix_int(graph2);

    free_matrix_int(graph_eeadj);
    free_matrix_double(nn_strength_mat);
    free_matrix_double(ee_strength_mat);
    free_matrix_int(graph_eeadj2);
    free_matrix_double(nn_strength_mat2);
    free_matrix_double(ee_strength_mat2);
    
    *nsim = nn_sim->m;
    *esim = ee_sim->m;
    
    return 0;
}

/**
 * Calculate the self-similarity via TACSim algrithm.
 */
int calculate_tacsim_self(int **A, double *Anw, double *Aew, int Anode, int Aedge,
                          double ***nsim, double ***esim,
                          int max_iter, double eps, double tol) {
    
    // create a new graph
    MatrixInt *graph = allocate_matrix_int(Anode, Anode, FALSE, -1);
    MatrixInt *graph_eeadj;
    VectorDouble *node_weights = allocate_vector_double(Anode, FALSE, -1);
    VectorDouble *edge_weights = allocate_vector_double(Aedge, FALSE, -1);
    MatrixDouble *nn_strength_mat = allocate_matrix_double(Anode, Anode, TRUE, -1);
    MatrixDouble *ee_strength_mat = allocate_matrix_double(Aedge, Aedge, TRUE, -1);
    
    graph->m = A;
    node_weights->v = Anw;
    edge_weights->v = Aew;
    
    normalize_vector(&(node_weights->v), node_weights->l);
    normalize_vector(&(edge_weights->v), edge_weights->l);
    
    graph_eeadj = get_edge_adjacency(&graph, Aedge);
    graph_elements(graph, node_weights, &nn_strength_mat, graph_eeadj, edge_weights, &ee_strength_mat);
    
    MatrixDouble *nn_sim = allocate_matrix_double(Anode, Anode, TRUE, 1);
    MatrixDouble *ee_sim = allocate_matrix_double(Aedge, Aedge, TRUE, 1);
    
    tacsim(graph, graph_eeadj, nn_strength_mat, ee_strength_mat,
           graph, graph_eeadj, nn_strength_mat, ee_strength_mat,
           &nn_sim, &ee_sim, max_iter, eps, tol);
    
    // We should not free the accepted memory
//    free_vector_double(node_weights);
//    free_vector_double(edge_weights);
//    free_matrix_int(graph);
    
    free_matrix_int(graph_eeadj);
    free_matrix_double(nn_strength_mat);
    free_matrix_double(ee_strength_mat);
    
    *nsim = nn_sim->m;
    *esim = ee_sim->m;
    
    return 0;
}
