// PC Linux stubs for the lock tests (linux-thread): the Darwin calls dm/android/compat declares and dm/android/compat/compat.c defines on Android
#include <stdint.h>
#include <time.h>
#include <sys/uio.h>
#include <unistd.h>
#include <mach/mach.h>
uint64_t mach_absolute_time(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
mach_port_t mach_thread_self(void) { return (mach_port_t)gettid(); }
kern_return_t vm_read_overwrite(task_t t, vm_address_t a, vm_size_t n, vm_address_t to, vm_size_t *got) {
    (void)t; struct iovec l = { (void *)to, n }, r = { (void *)a, n };
    ssize_t k = process_vm_readv(getpid(), &l, 1, &r, 1, 0); if (k < 0) { *got = 0; return KERN_INVALID_ADDRESS; }
    *got = (vm_size_t)k; return KERN_SUCCESS;
}
