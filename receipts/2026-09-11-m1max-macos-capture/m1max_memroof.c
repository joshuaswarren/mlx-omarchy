// m1max_memroof.c — CPU large-copy memory-roof probe (pre-install capture)
// Method: 8 threads (8 P-cores), each memcpy's its own 128 MiB slice of a
// 1 GiB source to a 1 GiB destination; whole-pass wall time via
// CLOCK_MONOTONIC_nz; one untimed warmup pass; then enough timed passes to
// cover >= 1.5 s. GB/s = (passes * copied bytes) / wall seconds.
// Run on AC, quiet machine; prints one JSON object.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

#define GIB (1024UL * 1024UL * 1024UL)
#define SLICE (GIB / 8UL)
#define NTHREADS 8

typedef struct { char *src; char *dst; } slice_t;
static slice_t slices[NTHREADS];
static int passes;

static void *worker(void *arg) {
    slice_t *s = (slice_t *)arg;
    for (int i = 0; i < passes; ++i) {
        memcpy(s->dst, s->src, SLICE);
    }
    return NULL;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(void) {
    char *src = NULL, *dst = NULL;
    if (posix_memalign((void **)&src, 16384, GIB) != 0) { perror("src"); return 1; }
    if (posix_memalign((void **)&dst, 16384, GIB) != 0) { perror("dst"); return 1; }
    memset(src, 0x5A, GIB);
    memset(dst, 0, GIB);
    for (int i = 0; i < NTHREADS; ++i) {
        slices[i].src = src + (size_t)i * SLICE;
        slices[i].dst = dst + (size_t)i * SLICE;
    }
    pthread_t th[NTHREADS];

    // warmup: one full pass, untimed
    passes = 1;
    for (int i = 0; i < NTHREADS; ++i) pthread_create(&th[i], NULL, worker, &slices[i]);
    for (int i = 0; i < NTHREADS; ++i) pthread_join(th[i], NULL);

    double t0 = now_sec();
    passes = 1;
    for (int i = 0; i < NTHREADS; ++i) pthread_create(&th[i], NULL, worker, &slices[i]);
    for (int i = 0; i < NTHREADS; ++i) pthread_join(th[i], NULL);
    double one_pass = now_sec() - t0;

    int timed = (int)(1.5 / one_pass);
    if (timed < 3) timed = 3;
    if (timed > 200) timed = 200;

    t0 = now_sec();
    passes = timed;
    for (int i = 0; i < NTHREADS; ++i) pthread_create(&th[i], NULL, worker, &slices[i]);
    for (int i = 0; i < NTHREADS; ++i) pthread_join(th[i], NULL);
    double wall = now_sec() - t0;

    double bytes = (double)timed * (double)GIB;
    printf("{\n");
    printf("  \"method\": \"8-thread memcpy, 8x128MiB slices, 1GiB src -> 1GiB dst, posix_memalign 16K, memset-touched, 1 untimed warmup pass; traffic counted as read+write\",\n");
    printf("  \"threads\": %d,\n", NTHREADS);
    printf("  \"buffer_gib\": 1,\n");
    printf("  \"single_pass_s\": %.6f,\n", one_pass);
    printf("  \"timed_passes\": %d,\n", timed);
    printf("  \"wall_s\": %.6f,\n", wall);
    printf("  \"gbps\": %.2f,\n", 2 * bytes / wall / 1e9);
    printf("  \"dst_checksum_ok\": %s\n", memcmp(src, dst, GIB) == 0 ? "true" : "false");
    printf("}\n");
    return 0;
}
