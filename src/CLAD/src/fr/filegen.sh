#!/bin/bash

# List of function names
functions=("fr_fromUint64" "fr_toUint64" "fr_cpy" "fr_reduce4" "fr_neg" "fr_x2" "fr_x3" "fr_x4" "fr_x8" "fr_x12" "fr_add" "fr_sub" "fr_addsub" "fr_sqr" "fr_mul" "fr_inv" "fr_zero" "fr_one" "fr_eq" "fr_neq" "fr_nonzero" "fr_iszero" "fr_isone" "fr_print" )

# List of flags 1 to 1
flags=("extern \"C\" __device__ __host__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ __host__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ void" "extern \"C\" __device__ __host__ void" "extern \"C\" __device__ __host__ void" "extern \"C\" __device__ bool" "extern \"C\" __device__ bool" "extern \"C\" __device__ bool" "extern \"C\" __device__ bool" "extern \"C\" __device__ bool" "extern \"C\" __device__ void" )

# List of inputs
inputs=("fr_t &z, const uint64_t *x" "const fr_t &x, uint64_t *z" "fr_t &z, const fr_t &x" "fr_t &z" "fr_t &z" "fr_t &z" "fr_t &z" "fr_t &z" "fr_t &z" "fr_t &z" "fr_t &z, const fr_t &x" "fr_t &z, const fr_t &x" "fr_t &x, fr_t &y" "fr_t &z" "fr_t &z, const fr_t &x" "fr_t &z" "fr_t &z" "fr_t &z" "const fr_t &x, const fr_t &y" "const fr_t &x, const fr_t &y" "const fr_t &x" "const fr_t &x" "const fr_t &x" "const char *s, const fr_t &x" )

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
    echo "#include \"fr.cuh\" " >> "$filename"
    echo "" >> "$filename"

    echo "$flag $funcname($inp){" >> "$filename"
    echo "    #warning Function not implemented: $funcname" >> "$filename"
    echo "}" >> "$filename"

    echo "File $filename created."
done
