#include "testG1.cuh"

/**
 * @brief Test operation over G1 using KAT and self consistency:
 * 
 * inf==inf
 * inf+inf == inf
 * G+0 == 0+G == G
 * G+G == 2*G
 * 2*G == 2*G with KAT
 * G+2*G == 3*G with KAT
 * 2*2*2*3G == 24G with KAT
 * 24G-2G+3G == 25G with KAT
 * 25*G == 25G with KAT
 * addsub(2G, G) == 3G, G with KAT
 * addsub(G, G) = (2G, 2G) (dbl and add)
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
__global__ void G1TestKAT(bool *result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: G1TestKAT


    //TODO: Generate KATS
    g1p_t
        g1p_x0 = {
            {0,0,0,0},
            {0,0,0,0},
            {0,0,0,0}
        },
        g1p_x2 = {
            {0,0,0,0},
            {0,0,0,0},
            {0,0,0,0}
        },
        g1p_x3 = {
            {0,0,0,0},
            {0,0,0,0},
            {0,0,0,0}
        },
        g1p_x24 = {
            {0,0,0,0},
            {0,0,0,0},
            {0,0,0,0}
        },
        g1p_x25 = {
            {0,0,0,0},
            {0,0,0,0},
            {0,0,0,0}
        },
        g1p_wG = {
            {0,0,0,0},
            {0,0,0,0},
            {0,0,0,0}
        },
        g1p_8G = {
            {0,0,0,0},
            {0,0,0,0},
            {0,0,0,0}
        };

}
