// H.264 movie decoding on Android (hle_movie_avc.h), for now without a decoder: every access unit is refused once with a red line and no picture comes out, so H.264 movies
// play as black frames with their sound. VP9 movies (libvpx) are unaffected. The app phase replaces this with AMediaCodec
#include "hle_movie_avc.h"
#include <stdio.h>
#include <stdlib.h>
struct mavc { int warned; };
mavc *mavc_new(void) { return calloc(1, sizeof(mavc)); }
void mavc_free(mavc *a) { free(a); }
void mavc_param_sets(mavc *a, const uint8_t *p, size_t n) { (void)a; (void)p; (void)n; }
void mavc_avcc(mavc *a, const uint8_t *p, size_t n) { (void)a; (void)p; (void)n; }
int mavc_decode(mavc *a, const uint8_t *p, int n, int64_t pts_us) {
    (void)p; (void)n; (void)pts_us;
    if (!a->warned++) fprintf(stderr, "[dm] [RED] movie: no H.264 decoder on Android yet, the movie shows black\n");
    return -1;
}
int mavc_take(mavc *a, int flush, uint8_t *nv12, int w, int h, int64_t *pts_us) { (void)a; (void)flush; (void)nv12; (void)w; (void)h; (void)pts_us; return 0; }
int mavc_queued(const mavc *a) { (void)a; return 0; }
void mavc_set_rgba(mavc *a) { (void)a; }
void mavc_drop(mavc *a) { (void)a; }
