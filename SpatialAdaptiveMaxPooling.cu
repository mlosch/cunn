
#define CUDA_MAX_THREADS 1024   // this is safe, in reality 256 is our limit

/*
 * Description:
 *    this function adaptively maxpools an input 4D tensor along dimensions 2 and 3
 *    4D input, 4D output, 4D argmax x and y
 */
__global__ void adaptivemaxpool(float *input, float *output, float *indices_x, float *indices_y,
                        int input_n, int input_h, int input_w,
                        int output_h, int output_w)
{
  // iterators
  int xx, yy;

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;
  //int k = blockIdx.x % input_n;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  const int xx_step = blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  const int yy_step = blockDim.y*gridDim.y;

  // select input/output plane
  output = output + o*output_w*output_h;
  input = input + i*input_w*input_h;
  indices_x = indices_x + o*output_w*output_h;
  indices_y = indices_y + o*output_w*output_h;

  // For all output pixels...
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {

    int y_start = (int)floor(float(yy) / output_h * input_h);
    int y_end   = (int)ceil(float(yy+1) / output_h * input_h);
    int kH = y_end-y_start;

    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      int x_start = (int)floor(float(xx) / output_w * input_w);
      int x_end   = (int)ceil(float(xx + 1) / output_w * input_w);

      int kW = x_end-x_start;

      // Compute the mean of the input image...
      float *ptr_input = input + y_start*input_w + x_start;
      float *ptr_output = output + yy*output_w + xx;
      float *ptr_ind_x = indices_x + yy*output_w + xx;
      float *ptr_ind_y = indices_y + yy*output_w + xx;
      int argmax_x = -1;
      int argmax_y = -1;
      float max = -FLT_MAX;
      int kx, ky;
      for(ky = 0; ky < kH; ky++) {
        for(kx = 0; kx < kW; kx++) {
          float val = ptr_input[kx];
          if (val > max) {
            max = val;
            argmax_x = kx;
            argmax_y = ky;
          }
        }
        ptr_input += input_w; // next input line
      }
      // Update output and argmax
      *ptr_output = max;
      *ptr_ind_x = argmax_x + 1;
      *ptr_ind_y = argmax_y + 1;
    }
  }
}

/*
 * Description:
 *    this function computes the gradInput from weight and gradOutput
 */
__global__ void adaptivemaxgradinput(float *gradInput, float *gradOutput, float *indices_x, float *indices_y,
                             int input_n, int input_h, int input_w,
                             int output_h, int output_w)
{
  // iterators
  int xx, yy;

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;
  //int k = blockIdx.x % input_n;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  int yy_step = blockDim.y*gridDim.y;

  // select input/output plane
  gradOutput = gradOutput + o*output_w*output_h;
  gradInput = gradInput + i*input_w*input_h;
  indices_x = indices_x + o*output_w*output_h;
  indices_y = indices_y + o*output_w*output_h;

  // compute gradInput
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {

    int y_start = (int)floor(float(yy) / output_h * input_h);

    for(xx = xx_start; xx < xx_end; xx+=xx_step) {

      int x_start = (int)floor(float(xx) / output_w * input_w);

      float *ptr_gradInput = gradInput + y_start*input_w + x_start;
      float *ptr_gradOutput = gradOutput + yy*output_w + xx;
      float *ptr_ind_x = indices_x + yy*output_w + xx;
      float *ptr_ind_y = indices_y + yy*output_w + xx;
      float z = *ptr_gradOutput;

      int argmax_x = (*ptr_ind_x)-1;
      int argmax_y = (*ptr_ind_y)-1;

      ptr_gradInput[argmax_x + argmax_y*input_w] += z;
    }
  }
}

/*
 * Description:
 *    this function computes the gradInput from weight and gradOutput
 *    when kH != dH or kW != dW (uses atomic add)
 */
__global__ void atomicadaptivemaxgradinput(
  float *gradInput, float *gradOutput, float *indices_x, float *indices_y,
  int input_n, int input_h, int input_w, int output_h, int output_w
)
{
  // iterators
  int xx, yy;

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  int yy_step = blockDim.y*gridDim.y;

  // select input/output plane
  gradOutput = gradOutput + o*output_w*output_h;
  gradInput = gradInput + i*input_w*input_h;
  indices_x = indices_x + o*output_w*output_h;
  indices_y = indices_y + o*output_w*output_h;

  // compute gradInput
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {

    int y_start = (int)floor(float(yy) / output_h * input_h);

    for(xx = xx_start; xx < xx_end; xx+=xx_step) {

      int x_start = (int)floor(float(xx) / output_w * input_w);

      float *ptr_gradInput = gradInput + y_start*input_w + x_start;
      float *ptr_gradOutput = gradOutput + yy*output_w + xx;
      float *ptr_ind_x = indices_x + yy*output_w + xx;
      float *ptr_ind_y = indices_y + yy*output_w + xx;
      float z = *ptr_gradOutput;

      int argmax_x = (*ptr_ind_x)-1;
      int argmax_y = (*ptr_ind_y)-1;

      // atomic add since different threads could update same variable
      atomicAdd(&(ptr_gradInput[argmax_x + argmax_y*input_w]), z);
    }
  }
}

static int cunn_SpatialAdaptiveMaxPooling_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");

  long nOutputCols = luaT_getfieldcheckint(L, 1, "W");
  long nOutputRows = luaT_getfieldcheckint(L, 1, "H");

  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *indices = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 3, input, output, indices));

  float *indices_data;
  float *output_data;
  float *input_data;

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long nInputPlane = input->size[0];

    input = THCudaTensor_newContiguous(state, input);
    input_data = THCudaTensor_data(state, input);

    THCudaTensor_resize3d(state, output, nInputPlane, nOutputRows, nOutputCols);
    THCudaTensor_resize4d(state, indices, 2, nInputPlane, nOutputRows, nOutputCols);

    indices_data = THCudaTensor_data(state, indices);
    output_data = THCudaTensor_data(state, output);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane,yblocks);
    dim3 threads(32,8);

    // run maxpool kernel
    adaptivemaxpool <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (input_data, output_data,
                                   indices_data+nInputPlane*nOutputCols*nOutputRows, indices_data,
                                   nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols);
  } else {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nInputPlane = input->size[1];
    long nbatch = input->size[0];

    input = THCudaTensor_newContiguous(state, input);
    input_data = THCudaTensor_data(state, input);

    THCudaTensor_resize4d(state, output, nbatch, nInputPlane, nOutputRows, nOutputCols);
    THCudaTensor_resize5d(state, indices, 2, nbatch, nInputPlane, nOutputRows, nOutputCols);

    indices_data = THCudaTensor_data(state, indices);
    output_data = THCudaTensor_data(state, output);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane*nbatch,yblocks);
    dim3 threads(32,8);

    // run maxpool kernel
    adaptivemaxpool <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (input_data, output_data,
                                   indices_data+nbatch*nInputPlane*nOutputCols*nOutputRows, indices_data,
                                   nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols);
  }

  // clean
  THCudaTensor_free(state, input);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialAdaptiveMaxPooling.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}

static int cunn_SpatialAdaptiveMaxPooling_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

  bool atomic = true; // suboptimal, but without atomic it doesn't pass the tests

  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor *indices = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 4, input, indices, gradOutput, gradInput));

  float *indices_data;
  float *gradInput_data;
  float *gradOutput_data;

  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long nInputPlane = input->size[0];
    long nOutputCols = gradOutput->size[2];
    long nOutputRows = gradOutput->size[1];

    //bool atomic = (nInputCols%nOutputCols != 0) || (nInputRows%nOutputRows != 0);

    THCudaTensor_resizeAs(state, gradInput, input);
    THCudaTensor_zero(state, gradInput);

    indices_data = THCudaTensor_data(state, indices);
    gradOutput_data = THCudaTensor_data(state, gradOutput);
    gradInput_data = THCudaTensor_data(state, gradInput);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane,yblocks);
    dim3 threads(32,8);

    if(atomic)
    {
      // run updateGradInput kernel, accumulate gradients atomically
      atomicadaptivemaxgradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          indices_data+nInputPlane*nOutputCols*nOutputRows, indices_data,
                                          nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols);
    }
    else
    {
      // run updateGradInput kernel
      atomicadaptivemaxgradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          indices_data+nInputPlane*nOutputCols*nOutputRows, indices_data,
                                          nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols);
    }
  } else {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nInputPlane = input->size[1];
    long nbatch = input->size[0];
    long nOutputCols = gradOutput->size[3];
    long nOutputRows = gradOutput->size[2];

    //bool atomic = //(nInputCols%nOutputCols != 0) || (nInputRows%nOutputRows != 0);

    THCudaTensor_resizeAs(state, gradInput, input);
    THCudaTensor_zero(state, gradInput);

    indices_data = THCudaTensor_data(state, indices);
    gradOutput_data = THCudaTensor_data(state, gradOutput);
    gradInput_data = THCudaTensor_data(state, gradInput);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane*nbatch,yblocks);
    dim3 threads(32,8);

    if(atomic)
    {
      // run updateGradInput kernel, accumulate gradients atomically
      atomicadaptivemaxgradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          indices_data+nbatch*nInputPlane*nOutputCols*nOutputRows, indices_data,
                                          nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols);
    }
    else
    {
      // run updateGradInput kernel, accumulate gradients atomically
      adaptivemaxgradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          indices_data+nbatch*nInputPlane*nOutputCols*nOutputRows, indices_data,
                                          nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols);
    }
  }

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialAdaptiveMaxPooling.updateGradInput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}

static const struct luaL_Reg cunn_SpatialAdaptiveMaxPooling__ [] = {
  {"SpatialAdaptiveMaxPooling_updateOutput", cunn_SpatialAdaptiveMaxPooling_updateOutput},
  {"SpatialAdaptiveMaxPooling_updateGradInput", cunn_SpatialAdaptiveMaxPooling_updateGradInput},
  {NULL, NULL}
};

static void cunn_SpatialAdaptiveMaxPooling_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SpatialAdaptiveMaxPooling__, "nn");
  lua_pop(L,1);
}

#undef CUDA_MAX_THREADS
