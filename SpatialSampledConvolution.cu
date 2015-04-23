#include "utils.h"

//from small to large kernel
__global__ void transfer_up_kernel(const int n,
    const float* from, float* to, 
    const int n_input, const int n_output,
    const int from_kH, const int from_kW, const int to_kH, const int to_kW,
    const int stride_h, const int stride_w) {

  CUDA_KERNEL_LOOP(index, n) {
    
    int n_i = index%n_input;
    index /= n_input;
    int n_o = index%n_output;

    const float* from_it = from + (n_o*n_input*from_kH*from_kW + n_i*from_kH*from_kW);
    float* to_it = to + (n_o*n_input*to_kH*to_kW + n_i*to_kH*to_kW);

    for (int h_col = 0; h_col < to_kH; h_col += 1+stride_h) {
      for (int w_col = 0; w_col < to_kW; w_col += 1+stride_w, ++from_it) {
        to_it[h_col*to_kW + w_col] = *from_it;
      }
    }

  }
}

__global__ void transfer_down_kernel(const int n,
    const float* from, float* to, 
    const int n_input, const int n_output,
    const int from_kH, const int from_kW, const int to_kH, const int to_kW,
    const int stride_h, const int stride_w) {

  CUDA_KERNEL_LOOP(index, n) {
    
    int n_i = index%n_input;
    index /= n_input;
    int n_o = index%n_output;

    const float* from_it = from + (n_o*n_input*from_kH*from_kW + n_i*from_kH*from_kW);
    float* to_it = to + (n_o*n_input*to_kH*to_kW + n_i*to_kH*to_kW);

    for (int h_col = 0; h_col < from_kH; h_col += 1+stride_h) {
      for (int w_col = 0; w_col < from_kW; w_col += 1+stride_w, ++to_it) {
        *to_it = from_it[h_col*from_kW + w_col];
      }
    }

  }
}

void transfer_up(const float* from, float* to, 
    const int n_input, const int n_output,
    const int from_kH, const int from_kW, const int to_kH, const int to_kW,
    const int stride_h, const int stride_w)
{
  int num_kernels = n_input * n_output;
  transfer_up_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(num_kernels, 
    from, to, 
    n_input, n_output, 
    from_kH, from_kW,
    to_kH, to_kW,
    stride_h, stride_w);
}

void transfer_down(const float* from, float* to, 
    const int n_input, const int n_output,
    const int from_kH, const int from_kW, const int to_kH, const int to_kW,
    const int stride_h, const int stride_w)
{
  int num_kernels = n_input * n_output;
  transfer_down_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(num_kernels, 
    from, to, 
    n_input, n_output, 
    from_kH, from_kW,
    to_kH, to_kW,
    stride_h, stride_w);
}

__global__ void printKernel(const float* k, const int n_input, const int n_output, const int kH, const int kW)
{
  for(int no = 0; no < n_output; ++no) {
    for(int ni = 0; ni < n_input; ++ni) {
      for(int y = 0; y < kH; ++y) {
        for(int x = 0; x < kW; ++x) {
          printf("%.3f, ",k[no*n_input*kH*kW + ni*kH*kW + y*kW + x]);
        }
        printf("\n");
      }
      printf("\n");
    }
  }
}

static int cunn_SpatialSampledConvolution_updateOutput(lua_State *L) {
  THCState *state = getCutorchState(L);
  // Input
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

  // Params:
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int tkW = luaT_getfieldcheckint(L, 1, "hkW");
  int tkH = luaT_getfieldcheckint(L, 1, "hkH");
  int dkW = luaT_getfieldcheckint(L, 1, "dkW");
  int dkH = luaT_getfieldcheckint(L, 1, "dkH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padding = luaT_getfieldcheckint(L, 1, "padding");

  THCudaTensor *weightTarget = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *columns = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.CudaTensor");
  THCudaTensor *ones = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "fgradInput", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  const int device = THCudaTensor_getDevice(state, weightTarget);
  luaL_argcheck(L, THCudaTensor_getDevice(state, bias) == device, 1,
                "weight and bias need to be on the same device");
  luaL_argcheck(L, THCudaTensor_getDevice(state, output) == device ||
                THCudaTensor_getDevice(state, output) == -1, 1,
                "weight and output need to be on the same device");
  luaL_argcheck(L, THCudaTensor_getDevice(state, input) == device, 2,
                "weight and input need to be on the same device");

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 3) {
    luaL_argcheck(L, input->size[0] == nInputPlane, 2, "input channels and nInputPlane dont match");
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
  } else {
    luaL_argcheck(L, input->size[1] == nInputPlane, 2, "input channels and nInputPlane dont match");
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padding - kH) / dH + 1;


  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCudaTensor_resize4d(state, output, batchSize, nOutputPlane, outputHeight, outputWidth);

  // Resize temporary columns
  THCudaTensor_resize2d(state, columns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCudaTensor_resize2d(state, ones, outputHeight, outputWidth);
    THCudaTensor_fill(state, ones, 1);
  }

  // Helpers
  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *output_n = THCudaTensor_new(state);

  THCudaTensor *weight = THCudaTensor_new(state);
  THCudaTensor_resize2d(state, weight, nOutputPlane, nInputPlane*kH*kW);
  float* weightData = THCudaTensor_data(state, weight);
  float* weightTargetData = THCudaTensor_data(state, weightTarget);

  //reset large kernel
  THCudaTensor_fill(state, weight, 0);
  //copy data into large kernel
  transfer_up(weightTargetData, weightData, nInputPlane, nOutputPlane, tkH, tkW, kH, kW, dkH, dkW);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, output_n, output, 0, elt);

    // Do Bias first:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long n_ = outputHeight * outputWidth;
    long k_ = 1;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_gemm(
        state,
        't', 'n',
        n_, m_, k_,
        1,
        THCudaTensor_data(state, ones), k_,
        THCudaTensor_data(state, bias), k_,
        0,
        THCudaTensor_data(state, output_n), n_
    );

    // Extract columns:
    im2col(
        THCudaTensor_data(state, input_n),
        nInputPlane, inputHeight, inputWidth, kH, kW, padding, padding, dH, dW,
        THCudaTensor_data(state, columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = weight->size[0];
    long n = columns->size[1];
    long k = weight->size[1];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_gemm(
        state,
        'n', 'n',
        n, m, k,
        1,
        THCudaTensor_data(state, columns), n,
        weightData, k,
        1,
        THCudaTensor_data(state, output_n), n
    );
  }

  // Free
  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, output_n);
  THCudaTensor_free(state, weight);

  // Resize output
  if (batch == 0) {
    THCudaTensor_resize3d(state, output, nOutputPlane, outputHeight, outputWidth);
    THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
  }

  // return output
  return 1;
}

static int cunn_SpatialSampledConvolution_updateGradInput(lua_State *L) {
  THCState *state = getCutorchState(L);
  // Inputs
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

  // Params
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int tkW = luaT_getfieldcheckint(L, 1, "hkW");
  int tkH = luaT_getfieldcheckint(L, 1, "hkH");
  int dkW = luaT_getfieldcheckint(L, 1, "dkW");
  int dkH = luaT_getfieldcheckint(L, 1, "dkH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padding = luaT_getfieldcheckint(L, 1, "padding");

  THCudaTensor *weightTarget = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *gradColumns = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  const int device = THCudaTensor_getDevice(state, weightTarget);
  luaL_argcheck(L, THCudaTensor_getDevice(state, input) == device, 2,
                "weight and input need to be on the same device");
  luaL_argcheck(L, THCudaTensor_getDevice(state, gradInput) == device
                || THCudaTensor_getDevice(state, gradInput) == -1, 2,
                "weight and gradInput need to be on the same device");
  luaL_argcheck(L, THCudaTensor_getDevice(state, gradOutput) == device
                || THCudaTensor_getDevice(state, gradOutput) == -1, 2,
                "weight and gradOutput need to be on the same device");


  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THCudaTensor_resize4d(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padding - kH) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCudaTensor_resize4d(state, gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

  // Resize temporary columns
  THCudaTensor_resize2d(state, gradColumns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Helpers
  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *gradInput_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);
  
  THCudaTensor *weight = THCudaTensor_new(state);
  THCudaTensor_resize2d(state, weight, nOutputPlane, nInputPlane*kH*kW);
  float* weightData = THCudaTensor_data(state, weight);
  float* weightTargetData = THCudaTensor_data(state, weightTarget);

  //reset large kernel
  THCudaTensor_fill(state, weight, 0);
  //copy data into large kernel
  transfer_up(weightTargetData, weightData, nInputPlane, nOutputPlane, tkH, tkW, kH, kW, dkH, dkW);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per sample:
    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, gradInput_n, gradInput, 0, elt);
    THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = weight->size[1];
    long n = gradColumns->size[1];
    long k = weight->size[0];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_gemm(
        state,
        'n', 't',
        n, m, k,
        1,
        THCudaTensor_data(state, gradOutput_n), n,
        weightData, m,
        0,
        THCudaTensor_data(state, gradColumns), n
    );

    // Unpack columns back into input:
    col2im(
        THCudaTensor_data(state, gradColumns),
        nInputPlane, inputHeight, inputWidth, kH, kW, padding, padding, dH, dW,
        THCudaTensor_data(state, gradInput_n)
    );
  }

  // Free
  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, gradInput_n);
  THCudaTensor_free(state, gradOutput_n);
  THCudaTensor_free(state, weight);

  // Resize output
  if (batch == 0) {
    THCudaTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight, outputWidth);
    THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
    THCudaTensor_resize3d(state, gradInput, nInputPlane, inputHeight, inputWidth);
  }

  // Return gradInput
  return 1;
}

static int cunn_SpatialSampledConvolution_accGradParameters(lua_State *L) {
  THCState *state = getCutorchState(L);
  // Inputs
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

  // Params
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int tkW = luaT_getfieldcheckint(L, 1, "hkW");
  int tkH = luaT_getfieldcheckint(L, 1, "hkH");
  int dkW = luaT_getfieldcheckint(L, 1, "dkW");
  int dkH = luaT_getfieldcheckint(L, 1, "dkH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padding = luaT_getfieldcheckint(L, 1, "padding");
  float scale = luaL_optnumber(L, 4, 1);

  THCudaTensor *gradWeightTarget = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradBias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");
  THCudaTensor *columns = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.CudaTensor");
  THCudaTensor *ones = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "fgradInput", "torch.CudaTensor");

  const int device = THCudaTensor_getDevice(state, gradWeightTarget);
  luaL_argcheck(L, THCudaTensor_getDevice(state, gradBias) == device, 1,
                "gradWeight and gradBias need to be on the same device");
  luaL_argcheck(L, THCudaTensor_getDevice(state, input) == device, 1,
                "gradWeight and input need to be on the same device");
  luaL_argcheck(L, THCudaTensor_getDevice(state, gradOutput) == device, 1,
                "gradWeight and gradOutput need to be on the same device");

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THCudaTensor_resize4d(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padding - kH) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCudaTensor_resize2d(state, ones, outputHeight, outputWidth);
    THCudaTensor_fill(state, ones, 1);
  }

  // Resize temporary columns
  THCudaTensor_resize2d(state, columns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Helpers
  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  THCudaTensor *gradWeight = THCudaTensor_new(state);
  THCudaTensor_resize2d(state, gradWeight, nOutputPlane, nInputPlane*kH*kW);
  float* gradWeightData = THCudaTensor_data(state, gradWeight);
  float* gradWeightTargetData = THCudaTensor_data(state, gradWeightTarget);

  //reset large kernel
  THCudaTensor_fill(state, gradWeight, 0);
  //copy data into large kernel
  transfer_up(gradWeightTargetData, gradWeightData, nInputPlane, nOutputPlane, tkH, tkW, kH, kW, dkH, dkW);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    im2col(
        THCudaTensor_data(state, input_n),
        nInputPlane, inputHeight, inputWidth, kH, kW, padding, padding, dH, dW,
        THCudaTensor_data(state, columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = gradWeight->size[0];
    long n = gradWeight->size[1];
    long k = columns->size[1];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_gemm(
        state,
        't', 'n',
        n, m, k,
        scale,
        THCudaTensor_data(state, columns), k,
        THCudaTensor_data(state, gradOutput_n), k,
        1,
        gradWeightData, n
    );

    // Do Bias:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long k_ = outputHeight * outputWidth;

    // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
    THCudaBlas_gemv(
        state,
        't',
        k_, m_,
        scale,
        THCudaTensor_data(state, gradOutput_n), k_,
        THCudaTensor_data(state, ones), 1,
        1,
        THCudaTensor_data(state, gradBias), 1
    );
  }

  //printKernel<<<1,1>>>(gradWeightTargetData, nInputPlane, nOutputPlane, tkH, tkW);
  //printf("\n");
  //printKernel<<<1,1>>>(gradWeightData, nInputPlane, nOutputPlane, kH, kW);

  //copy data into small kernel
  transfer_down(gradWeightData, gradWeightTargetData, nInputPlane, nOutputPlane, kH, kW, tkH, tkW, dkH, dkW);

  // Free
  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, gradOutput_n);
  THCudaTensor_free(state, gradWeight);

  // Resize
  if (batch == 0) {
    THCudaTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight, outputWidth);
    THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
  }

  // Return nothing
  return 0;
}

static const struct luaL_Reg cunn_SpatialSampledConvolution__ [] = {
  {"SpatialSampledConvolution_updateOutput", cunn_SpatialSampledConvolution_updateOutput},
  {"SpatialSampledConvolution_updateGradInput", cunn_SpatialSampledConvolution_updateGradInput},
  {"SpatialSampledConvolution_accGradParameters", cunn_SpatialSampledConvolution_accGradParameters},
  {NULL, NULL}
};

static void cunn_SpatialSampledConvolution_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SpatialSampledConvolution__, "nn");
  lua_pop(L,1);
}
