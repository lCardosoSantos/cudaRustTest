#!/bin/bash

# List of function names
functions=("fp_fromUint64" "fp_toUint64" "fp_cpy" "fp_reduce6" "fp_neg" "fp_x2" "fp_x3" "fp_x4" "fp_x8" "fp_x12" "fp_add" "fp_sub" "fp_sqr" "fp_mul" "fp_mma" "fp_inv" "fp_zero" "fp_one" "fp_eq" "fp_neq" "fp_nonzero" "fp_iszero" "fp_isone" "fp_print"  )

# List of flags 1 to 1
flags=("extern \"C\" __device__ __host__ void" "extern \"C\" __device__ __host__ void" "extern \"C\" __device__ __host__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ __host__ void" "extern \"C\" __device__ __host__ void" "extern \"C\" __device__ bool" "extern \"C\" __device__ bool" "extern \"C\" __device__ bool" "extern \"C\" __device__ bool" "extern \"C\" __device__ bool" "extern \"C\" __device__ void")

# List of inputs
inputs=("fp_t &z, const uint64_t *x" "uint64_t *z, const fp_t &x" "fp_t &z, const fp_t &x" "fp_t &z" "fp_t &z, const fp_t &x" "fp_t &z, const fp_t &x" "fp_t &z, const fp_t &x" "fp_t &z, const fp_t &x" "fp_t &z, const fp_t &x" "fp_t &z, const fp_t &x" "fp_t &z, const fp_t &x, const fp_t &y" "fp_t &z, const fp_t &x, const fp_t &y" "fp_t &z, const fp_t &x" "fp_t &z, const fp_t &x, const fp_t &y" "fp_t &z, const fp_t &v, const fp_t &w, const fp_t &x, const fp_t &y" "fp_t &z, const fp_t &x" "fp_t &z" "fp_t &z" "const fp_t &x, const fp_t &y" "const fp_t &x, const fp_t &y" "const fp_t &x" "const fp_t &x" "const fp_t &x" "const char *s, const fp_t &x" )

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
    echo "#include \"fp.cuh\" " >> "$filename"
    echo "" >> "$filename"

    echo "$flag $funcname($inp){" >> "$filename"
    echo "    #warning Function not implemented: $funcname" >> "$filename"
    echo "}" >> "$filename"

    echo "File $filename created."
done
