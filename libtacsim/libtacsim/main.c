
#include "tacsim.h"


int main(int argc, const char * argv[]) {
    
    MatrixDouble *nn_sim = allocate_matrix_double(4, 3, 1);
    MatrixDouble *ee_sim = allocate_matrix_double(4, 2, 1);
    
    int nlen = 4;
    int elen = 4;
    
    // create a new graph
    int A[4][4] = {{-1,-1,0,-1}, {1,-1,2,-1}, {-1,-1,-1,3}, {-1,-1,-1,-1}};
    int Anw[4] = {1,1,5,1};
    int Aew[4] = {12,8,10,15};
    
    MatrixInt *graph = allocate_matrix_int(nlen, nlen, -1);
    MatrixInt *graph_eeadj;
    VectorDouble *node_weights = allocate_vector_double(nlen, -1);
    VectorDouble *edge_weights = allocate_vector_double(elen, -1);
    MatrixDouble *nn_strength_mat = allocate_matrix_double(nlen, nlen, -1);
    MatrixDouble *ee_strength_mat = allocate_matrix_double(elen, elen, -1);
    

    for (int i = 0; i<graph->h; i++) {
        for(int j = 0; j<graph->w; j++) {
            graph->m[i][j] = A[i][j];
        }
    }
    
    for (int i = 0; i < node_weights->l; i++)
        node_weights->v[i] = Anw[i];
    
    for (int i = 0; i < edge_weights->l; i++)
        edge_weights->v[i] = Aew[i];
    
    normalize_vector(&(node_weights->v), node_weights->l);
    normalize_vector(&(edge_weights->v), edge_weights->l);
    
    // derive edge-edge adjacency
    // elen = get_edge_count(&graph);
    graph_eeadj = get_edge_adjacency(&graph, elen);
    
    graph_elements(graph, node_weights, &nn_strength_mat, graph_eeadj, edge_weights, &ee_strength_mat);
    
    free_vector_double(node_weights);
    free_vector_double(edge_weights);
    
    // create a new graph
    int B[3][3] = {{-1,0,-1}, {-1,-1,1}, {-1,-1,-1}};
    int Bnw[3] = {1,3,1};
    int Bew[2] = {15,10};
    nlen = 3;
    elen = 2;
    
    MatrixInt *graph2 = allocate_matrix_int(nlen, nlen, -1);
    MatrixInt *graph_eeadj2;
    VectorDouble *node_weights2 = allocate_vector_double(nlen, -1);
    VectorDouble *edge_weights2 = allocate_vector_double(elen, -1);
    MatrixDouble *nn_strength_mat2 = allocate_matrix_double(nlen, nlen, -1);
    MatrixDouble *ee_strength_mat2 = allocate_matrix_double(elen, elen, -1);
    
    
    for (int i = 0; i<graph2->h; i++) {
        for(int j = 0; j<graph2->w; j++) {
            graph2->m[i][j] = B[i][j];
        }
    }
    
    for (int i = 0; i < node_weights2->l; i++)
        node_weights2->v[i] = Bnw[i];
    
    for (int i = 0; i < edge_weights2->l; i++)
        edge_weights2->v[i] = Bew[i];
    
    normalize_vector(&(node_weights2->v), node_weights2->l);
    normalize_vector(&(edge_weights2->v), edge_weights2->l);
    
    // derive edge-edge adjacency
    graph_eeadj2 = get_edge_adjacency(&graph2, elen);
    
    graph_elements(graph2, node_weights2, &nn_strength_mat2, graph_eeadj2, edge_weights2, &ee_strength_mat2);
    
    free_vector_double(node_weights2);
    free_vector_double(edge_weights2);
    
    tacsim(graph, graph_eeadj, nn_strength_mat, ee_strength_mat,
           graph2, graph_eeadj2, nn_strength_mat2, ee_strength_mat2,
           &nn_sim, &ee_sim, 100, 1e-4, 1e-6);
    
    free_matrix_int(graph);
    free_matrix_int(graph_eeadj);
    free_matrix_double(nn_strength_mat);
    free_matrix_double(ee_strength_mat);
    
    free_matrix_int(graph2);
    free_matrix_int(graph_eeadj2);
    free_matrix_double(nn_strength_mat2);
    free_matrix_double(ee_strength_mat2);
    
    printf("NN Sim\n");
    for (int i = 0; i<nn_sim->h; i++) {
        for(int j = 0; j<nn_sim->w; j++) {
            printf("%d,%d,%f\n", i, j, nn_sim->m[i][j]);
        }
    }
    printf("\n");
    
    printf("EE Sim\n");
    for (int i = 0; i<ee_sim->h; i++) {
        for(int j = 0; j<ee_sim->w; j++) {
            printf("%d,%d,%f\n", i, j, ee_sim->m[i][j]);
        }
    }
    printf("\n");
    
    free_matrix_double(nn_sim);
    free_matrix_double(ee_sim);
    
    return 0;
}