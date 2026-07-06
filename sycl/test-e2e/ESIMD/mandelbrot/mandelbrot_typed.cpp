//==------- mandelbrot_typed.cpp - DPC++ ESIMD on-device test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The gather4.typed/scatter4.typed messages exist only on pre-Xe2 devices.
// REQUIRES: aspect-ext_intel_legacy_image
// UNSUPPORTED: arch-intel_gpu_bmg_g21 || arch-intel_gpu_bmg_g31 || arch-intel_gpu_lnl_m || arch-intel_gpu_ptl_u || arch-intel_gpu_ptl_h || arch-intel_gpu_wcl || arch-intel_gpu_nvl_u || arch-intel_gpu_nvl_s
// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}
// RUN: %{build} %{mathflags} -o %t.out
// RUN: %{run} %t.out
//
// Example application demonstrating the typed-surface (image) RGBA scatter
// ESIMD API scatter_rgba_typed. Each work-item computes the Mandelbrot escape
// count for N consecutive pixels of an image row, converts it to an RGBA color
// with one 32-bit channel per component, and writes the pixels to an output
// image addressed by pixel coordinates. The result is verified against a CPU
// reference (a small number of boundary pixels are allowed to differ due to
// floating point rounding differences between host and device).

#include "../esimd_test_utils.hpp"

#include <sycl/accessor_image.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <cmath>
#include <iostream>
#include <vector>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

static constexpr unsigned WIDTH = 512;
static constexpr unsigned HEIGHT = 512;
static constexpr unsigned N = 16; // pixels processed per work-item
static constexpr int CRUNCH = 256;
static constexpr float XOFF = -2.09798f;
static constexpr float YOFF = -1.19798f;
static constexpr float SCALE = 0.004f;

// Scalar Mandelbrot escape count - shared math used to build the CPU reference.
static int mandel(int ix, int iy) {
  float xPos = ix * SCALE + XOFF;
  float yPos = iy * SCALE + YOFF;
  float x = 0.0f, y = 0.0f, xx = 0.0f, yy = 0.0f;
  int m = 0;
  do {
    y = x * y * 2.0f + yPos;
    x = xx - yy + xPos;
    yy = y * y;
    xx = x * x;
    m += 1;
  } while ((m < CRUNCH) & (xx + yy < 4.0f));
  return m;
}

int main() {
  std::vector<uint32_t> OutBuf(WIDTH * HEIGHT * 4, 0);

  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  std::cout << "Running on "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  try {
    image<2> ImgOut(OutBuf.data(), image_channel_order::rgba,
                    image_channel_type::unsigned_int32,
                    range<2>{WIDTH, HEIGHT});

    range<2> GlobalRange{WIDTH / N, HEIGHT};

    q.submit([&](handler &cgh) {
       auto Acc = ImgOut.get_access<uint4, access::mode::write>(cgh);
       cgh.parallel_for<class MandelTyped>(
           GlobalRange, [=](item<2> it) SYCL_ESIMD_KERNEL {
             uint32_t BaseX = it.get_id(0) * N;
             uint32_t Y = it.get_id(1);

             simd<uint32_t, N> U(BaseX, 1);
             simd<uint32_t, N> V = Y;
             simd<uint32_t, N> R = 0;

             // Compute the escape count for each of the N pixels.
             simd<int, N> m = 0;
             for (int lane = 0; lane < N; ++lane)
               m.select<1, 1>(lane) = mandel(BaseX + lane, Y);

             // Assemble the pixels channel-major: [R..., G..., B..., A...].
             simd<uint32_t, N * 4> Px;
             Px.select<N, 1>(0 * N) = (m * 15) & 0xff;  // R
             Px.select<N, 1>(1 * N) = (m * 7) & 0xff;   // G
             Px.select<N, 1>(2 * N) = (m * 3) & 0xff;   // B
             Px.select<N, 1>(3 * N) = 0xff;             // A

             scatter_rgba_typed<uint32_t, N>(Acc, U, V, R, Px);
           });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 1;
  }

  // Verify against a CPU reference. Allow a small number of boundary pixels to
  // differ due to host/device floating point rounding differences.
  unsigned NumDiff = 0;
  for (unsigned y = 0; y < HEIGHT; ++y) {
    for (unsigned x = 0; x < WIDTH; ++x) {
      int m = mandel(x, y);
      uint32_t Exp[4] = {static_cast<uint32_t>((m * 15) & 0xff),
                         static_cast<uint32_t>((m * 7) & 0xff),
                         static_cast<uint32_t>((m * 3) & 0xff), 0xffu};
      unsigned Pixel = y * WIDTH + x;
      for (int C = 0; C < 4; ++C)
        if (OutBuf[Pixel * 4 + C] != Exp[C]) {
          ++NumDiff;
          break;
        }
    }
  }

  double DiffRatio = static_cast<double>(NumDiff) / (WIDTH * HEIGHT);
  std::cout << "Mismatching pixels: " << NumDiff << " ("
            << (DiffRatio * 100.0) << "%)\n";

  // Boundary FP differences should affect only a tiny fraction of pixels; a
  // broken typed store would corrupt (almost) all of them.
  bool Passed = DiffRatio < 0.01;
  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
