// PC Linux (WSL, glibc, any CPU) stand-in for dm/android/compat/darwin.h, included first (-include) when hle_thread.c is built for the lock tests
//   (make GAME=<id> linux-thread): the Android one reads arm64 signal frames and bionic extras. The pause signal's register copy is never used by these tests,
//   so its context views are dummies here. The rest of the Darwin calls come from dm/android/compat (mach/, pthread/qos.h) and t_linux_stub.c
#pragma once
#define _GNU_SOURCE 1
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <signal.h>
#include <ucontext.h>
struct __darwin_arm_thread_state64 { uint64_t __x[29], __fp, __lr, __sp, __pc; uint32_t __cpsr, __pad; };
#define __darwin_arm_thread_state64_get_pc(s) ((s).__pc)
#define __darwin_arm_thread_state64_get_lr(s) ((s).__lr)
#define __darwin_arm_thread_state64_get_sp(s) ((s).__sp)
#define __darwin_arm_thread_state64_get_fp(s) ((s).__fp)
static struct __darwin_arm_thread_state64 t_uc_ss_dummy; static __uint128_t t_uc_v_dummy[32];
#define DM_UC_SS(uc) (*((void)(uc), &t_uc_ss_dummy))
#define DM_UC_V(uc) ((void)(uc), t_uc_v_dummy)
#define DM_UC_FPCR(uc) ((void)(uc), 0u)
#define DM_UC_FPSR(uc) ((void)(uc), 0u)
