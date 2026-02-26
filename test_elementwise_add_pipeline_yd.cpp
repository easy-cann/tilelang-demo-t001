#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR A_handle,  GM_ADDR B_handle,  GM_ADDR C_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> B;
  B.SetGlobalBuffer((__gm__ float*)B_handle);
  AscendC::GlobalTensor<float> C;
  C.SetGlobalBuffer((__gm__ float*)C_handle);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 131072);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }
  auto a_ub = ascend_ub.GetWithOffset<float>(4096,0);
  auto b_ub = ascend_ub.GetWithOffset<float>(4096,16384);
  // auto c_ub = ascend_ub.GetWithOffset<float>(4096,32768);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::copy_gm_to_ub<float, 32, 64>(a_ub[0], A[((((cid / 4) * 131072) + (vid * 65536)) + ((cid % 4) * 256))], 1024);
    tl::ascend::copy_gm_to_ub<float, 32, 64>(b_ub[0], B[((((cid / 4) * 131072) + (vid * 65536)) + ((cid % 4) * 256))], 1024);
    AscendC::PipeBarrier<PIPE_MTE2>();
    tl::ascend::copy_gm_to_ub<float, 32, 64>(a_ub[2048], A[(((((cid / 4) * 131072) + (vid * 65536)) + ((cid % 4) * 256)) + 32)], 1024);
    tl::ascend::copy_gm_to_ub<float, 32, 64>(b_ub[2048], B[(((((cid / 4) * 131072) + (vid * 65536)) + ((cid % 4) * 256)) + 32)], 1024);
    for (int32_t n_i = 0; n_i < 6; ++n_i) {
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(5);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(5);
      AscendC::Add(a_ub[((n_i % 2) * 2048)], a_ub[((n_i % 2) * 2048)], b_ub[((n_i % 2) * 2048)], 2048);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
      tl::ascend::copy_gm_to_ub<float, 32, 64>(a_ub[((n_i % 2) * 2048)], A[((((((cid / 4) * 131072) + (vid * 65536)) + ((cid % 4) * 256)) + (n_i * 32)) + 64)], 1024);
      tl::ascend::copy_gm_to_ub<float, 32, 64>(b_ub[((n_i % 2) * 2048)], B[((((((cid / 4) * 131072) + (vid * 65536)) + ((cid % 4) * 256)) + (n_i * 32)) + 64)], 1024);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
      AscendC::PipeBarrier<PIPE_MTE3>();
      tl::ascend::copy_ub_to_gm<float, 32, 64>(C[(((((cid / 4) * 131072) + (vid * 65536)) + ((cid % 4) * 256)) + (n_i * 32))], a_ub[((n_i % 2) * 2048)], 1024);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
    AscendC::Add(a_ub[0], a_ub[0], b_ub[0], 2048);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
    AscendC::PipeBarrier<PIPE_MTE3>();
    tl::ascend::copy_ub_to_gm<float, 32, 64>(C[(((((cid / 4) * 131072) + (vid * 65536)) + ((cid % 4) * 256)) + 192)], a_ub[0], 1024);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
    AscendC::Add(a_ub[2048], a_ub[2048], b_ub[2048], 2048);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(4);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(4);
    AscendC::PipeBarrier<PIPE_MTE3>();
    tl::ascend::copy_ub_to_gm<float, 32, 64>(C[(((((cid / 4) * 131072) + (vid * 65536)) + ((cid % 4) * 256)) + 224)], a_ub[2048], 1024);
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* A_handle, uint8_t* B_handle, uint8_t* C_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<32, nullptr, stream>>>(A_handle, B_handle, C_handle, fftsAddr);
}
