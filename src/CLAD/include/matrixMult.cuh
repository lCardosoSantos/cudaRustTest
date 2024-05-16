// Copyright 2022-2024 Dag Arne Osvik
// Copyright 2022-2024 Luan Cardoso dos Santos

#pragma once

#include "fr.cuh"


extern "C"
void multiplyWitnessCUDA(fr_t *res, fr_t *witness);

extern "C"
void freeManagedMatrix();

extern "C"
void sparseMatrixLoadCUDA(const fr_t *A_data, const size_t *A_colidx, const size_t *A_indptr, const size_t A_NNZ, const size_t A_nRows, 
                          const fr_t *B_data, const size_t *B_colidx, const size_t *B_indptr, const size_t B_NNZ, const size_t B_nRows, 
                          const fr_t *C_data, const size_t *C_colidx, const size_t *C_indptr, const size_t C_NNZ, const size_t C_nRows, 
                          const size_t nCols_l);
