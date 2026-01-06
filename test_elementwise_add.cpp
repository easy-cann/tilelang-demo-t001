#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>

using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR A_handle,  GM_ADDR B_handle,  GM_ADDR C_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  AscendC::GlobalTensor<float> A;  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> B;  B.SetGlobalBuffer((__gm__ float*)B_handle);
  AscendC::GlobalTensor<float> C;  C.SetGlobalBuffer((__gm__ float*)C_handle);

  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();

  auto cid = AscendC::GetBlockIdx();
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }

  auto a_ub = ascend_ub.GetWithOffset<float>(16384, 0);
  auto b_ub = ascend_ub.GetWithOffset<float>(16384, 65536);
  auto c_ub = ascend_ub.GetWithOffset<float>(16384, 131072);

  if ASCEND_IS_AIV {
    auto off_set = (cid / 4) * 131072 + (vid * 65536) + (cid % 4) * 256;
    AscendC::printf("======== cid=%d, vid=%d, off_set=%d ===== \n", cid, vid, off_set);

    tl::ascend::copy_gm_to_ub<float, 256, 64>(a_ub[0], A[off_set], 1024);
    tl::ascend::copy_gm_to_ub<float, 256, 64>(b_ub[0], B[off_set], 1024);
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::Add(c_ub[0], a_ub[0], b_ub[0], 16384);
    AscendC::PipeBarrier<PIPE_ALL>();

    tl::ascend::copy_ub_to_gm<float, 256, 64>(C[off_set], c_ub[0], 1024);
  }
}

extern "C" void call(uint8_t* A_handle, uint8_t* B_handle, uint8_t* C_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel<<<32, nullptr, stream>>>(A_handle, B_handle, C_handle, fftsAddr);
}