#!/bin/bash

# List of function names
functions=("g1a_fromUint64" "g1a_fromFp" "g1a_fromG1p" "g1a_cpy" "g1p_toUint64" "g1p_fromUint64" "g1p_fromG1a" "g1p_cpy" "g1p_eq" "g1p_neq" "g1p_isInf" "g1p_isPoint" "g1p_neg" "g1p_scale" "g1p_sub" "g1p_mul" "g1p_add" "g1p_addsub" "g1a_inf" "g1a_gen" "g1p_inf" "g1p_gen" "g1p_print" "g1a_print" )

# List of flags 1 to 1
flags=("extern \"C\" __device__ __host__ void" "extern \"C\" __device__ __host__ void" "extern \"C\" __device__ void         " "extern \"C\" __device__ __host__ void" "extern \"C\" __device__ void         " "extern \"C\" __device__ __host__ void" "extern \"C\" __device__ void         " "extern \"C\" __device__ __host__ void" "extern \"C\" __device__ bool" "extern \"C\" __device__ bool" "extern \"C\" __device__ bool" "extern \"C\" __device__ bool" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ __host__ void" "extern \"C\" __device__ __host__ void" "extern \"C\" __device__ __host__ void" "extern \"C\" __device__ __host__ void" "extern \"C\" __device__ __host__ void" "extern \"C\" __device__ __host__ void" )

# List of inputs
inputs=("g1a_t &a, const uint64_t *x, const uint64_t *y" "g1a_t &a, const fp_t &x, const fp_t &y" "g1a_t &a, const g1p_t &p" "g1a_t &a, const g1a_t &b" "const g1p_t &p, uint64_t *x, uint64_t *y, uint64_t *z" "g1p_t &p, const uint64_t *x, const uint64_t *y, const uint64_t *z" "g1p_t &p, const g1a_t &a" "g1p_t &p, const g1p_t &q" "const g1p_t &p, const g1p_t &q" "const g1p_t &p, const g1p_t &q" "const g1p_t &p" "const g1p_t &p" "g1p_t &p" "g1p_t &p, const fp_t &s" "g1p_t &p, const g1p_t &q" "g1p_t &p, const fr_t &x" "g1p_t &p, const g1a_t &q" "g1p_t &p, g1p_t &q" "g1a_t &a" "g1a_t &a" "g1p_t &p" "g1p_t &p" "const char *s, const g1p_t &p" "const char *s, const g1a_t &a" )

# Loop through each function
for i in "${!functions[@]}"; do
    funcname="${functions[$i]}"
    flag="${flags[$i]}"
    inp="${inputs[$i]}"
    
    # Create a new .cu file with the function name
    filename="$funcname.cu"
    echo "Creating file: $filename"
    touch "$filename"
    
    # Add test content to the file
    echo "#include \"g1.cuh\" " >> "$filename"
    echo "" >> "$filename"

    echo "$flag $funcname($inp){" >> "$filename"
    echo "    #warning Function not implemented: $funcname" >> "$filename"
    echo "}" >> "$filename"

    echo "File $filename created."
done
