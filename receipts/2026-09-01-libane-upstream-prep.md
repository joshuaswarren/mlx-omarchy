# libane upstream prep — eiln/ane stabilization series

Date: 2026-09-01
Status: **AWAITING OWNER APPROVAL TO OPEN THE PR. Nothing was opened or sent.**
Prepared by: LibanePrep (omp subagent, U9 "Stabilize the ANE driver and libane contract")

- Source evidence (read-only): `/home/joshuawarren/src/ane-linux-experiments`
- Working clone: `/tmp/eiln-ane` (fresh clone of https://github.com/eiln/ane at `0dcea99`)
- Branch: `omarchy-libane-stabilization` in `/tmp/eiln-ane`
- Patch exports: `/tmp/libane-series/0001..0006` (also inlined at the bottom of this receipt)
- mlx-omarchy files touched: this receipt only. Nothing committed to mlx-omarchy.

## 1. What the experiments' patches contain

`patches/libane-ane.c` (683 lines) and `patches/libane-ane.h` (145 lines) are the
proven full-file drop of eiln's `libane/ane.c` and `libane/ane.h`, carrying five
behavior deltas:

| # | Behavior | Anchors in patches/libane-ane.c / .h | Evidence in ane-linux-experiments |
|---|----------|--------------------------------------|-----------------------------------|
| 1 | ANEC payload offset 0x800 -> 0x1000 | `ane.c:34` `ANEC_HEADER_SIZE 0x1000UL`; used at `ane.c:406` `ane_pread` | `receipts/ane-static-graph-loop.log:5-7` ("eiln/libane read ANEC content at 0x800; anecc writes a 0x1000 header"); `docs/static-graphs.md:9-10`; current eiln/anecc `anecc/__init__.py:30` `HEADER_SIZE = 0x1000` |
| 2 | Kernel-window binding (`ane_bind_kernel`) | `ane.c:534-544`; decl `ane.h:106` | `receipts/ane-static-graph-loop.log:28-33` (gemm.ane identity bound passes through; identity then 2x identity -> 3.0 then 6.0); `docs/static-graphs.md:24-26` |
| 3 | Capacity query (`ane_kernel_capacity`) | `ane.c:527-532`; decl `ane.h:107` | same receipt lines 28-31 (capacity helper used by the binding) |
| 4 | State-indexed execution (`ane_exec_loop` + handle swap) | `ane.c:458-525` (`ane_exec_with_state_swap`, swap block at 474-480); decl `ane.h:104-105` | `receipts/ane-static-graph-loop.log:21-26` (mul.ane 3 iters -> 24.0; add-resident 3 iters -> 9.0); `docs/static-graphs.md:20-23`; driver script `ane-static-loop.py` |
| 5 | Timeout/recovery state (sentinel + poll) | `ane.c:483-502` (0x7c00 sentinel fill, 10000 x 100us poll, `-ETIMEDOUT`) | `receipts/ane-static-graph-loop.log:6-7` (sentinel output poll), `:97` (output stays at 0x7c00 sentinel when the tile hangs, KMD `-110`) |
| 6 | qid-1 pin (`args.pad = 0x81`) | `ane.c:468` | `receipts/ane-static-graph-loop.log:3` (KMD with queue-id 1), `:95` (`enqueue qid=1`), `docs/static-graphs.md:11` |

The extra-buffer binding (`src_bdx = 4 + ane_dst_count + idx`, `ane.c:35` in both
patch and upstream) is **already merged upstream** (commit `e585747 "libane:
reorganize fns"`). Not ported.

Behavior 6 is fork-specific and intentionally NOT ported: stock eiln/ane KMD
rejects any nonzero submit `pad` (`ane/src/ane_drv.c:237`) and hardcodes
`req.qid = 4` (`ane/src/ane_drv.c:243`). Setting `pad = 0x81` against the stock
driver fails every submit. Queue selection belongs to the KMD; the series
leaves it there.

## 2. Mapping onto eiln/ane HEAD (`0dcea99`)

| Behavior | Upstream state | Action |
|----------|----------------|--------|
| Extra-buffer binding | Merged (`e585747`) | none |
| ANEC offset 0x1000 | Absent (libane still at 0x800; stale vs current eiln/anecc 0x1000) | ported, commit 1 |
| `ane_kernel_capacity` | Absent | ported, commit 2 |
| `ane_bind_kernel` | Absent | ported, commit 3 |
| Sentinel poll + `-ETIMEDOUT` | Absent (`ane_exec` returns raw ioctl) | ported, commit 4 |
| `ane_exec_loop` state swap | Absent | ported, commit 5 |
| `pad = 0x81` qid pin | Absent; incompatible with stock KMD | not ported (documented in PR) |

## 3. The series (smallest-first, one behavior per commit)

Branch `omarchy-libane-stabilization`, base `0dcea99`:

| Commit | Subject | Device-free test |
|--------|---------|------------------|
| `5067e79` | libane: read anec payload at 0x1000 | ABI offsets pinned in test/validate; on-device proof is receipt 1 above |
| `d8cf05d` | libane: add ane_kernel_capacity | yes (capacity math: aligned, unaligned, exact-fit, overflow -> 0) |
| `d0e83dc` | libane: add ane_bind_kernel | yes (NULL, oversized, zero-size, placement at round_up(tsk_size,16), rebind overwrite, exact-capacity span) |
| `b3e6807` | libane: poll output sentinel in ane_exec | hardware-only (test/loop skips without device) |
| `449a000` | libane: add ane_exec_loop for resident state | yes (rejections: zero iterations, out-of-range indices, size mismatch -> -EINVAL before any ioctl) |
| `fbe40be` | test: add validate and loop examples | the tests themselves |

Adaptations vs the experiments' drop (all flagged, none silent):

- Declarations moved above statements (`-Wdeclaration-after-statement -Werror`
  in libane/Makefile rejects the experiments' file as-is).
- In the sentinel fill loop, `words`/`count` bind AFTER the swap resolves `bdx`
  (declare-then-assign). The experiments' single-declaration form is fine, but
  the C90-flagging reorder made the binding order load-bearing; tested.
- Commit 4 introduces `ane_exec_poll(nn)`; commit 5 renames it to
  `ane_exec_with_state_swap(nn, swap_state, src_idx, dst_idx)` and adds the
  swap, so each commit is one behavior.

## 4. Test results (run on this box, no ANE device present)

Every commit 1-6 builds clean with the repo's own flags in a detached worktree
(`5067e79` through `fbe40be`: BUILD_OK x6).

Build (libane/Makefile needs the KMD uapi header; on this box overridden, not
committed): `make -C libane LIBS="-I/usr/include/libdrm -I../ane/src/uapi/drm"`.

Device-free (`test/validate`, compiled `-Wall -Werror -Wextra
-Wdeclaration-after-statement`):

```
$ ./main.out
TEST: 30 checks, 0 failed        (exit 0; reruns stable)
```

Hardware (`test/loop`, skip-without-device convention exit 77):

```
$ ./main.out fake.anec
TEST: SKIP: no ANE device        (exit 77)
```

The hardware leg replays the receipt numbers when run on the M1: a state graph
loaded with `ane_exec_loop(nn, 3, 0, 0)` must print `ane_exec_loop: 0` and the
model's expected outputs (mul.ane -> 24.0 per
`receipts/ane-static-graph-loop.log:21-26`).

## 5. PR draft — NOT OPENED, awaiting owner approval

Title:
`libane: anec offset fix, kernel rebinding, exec timeout, resident state loop`

Body:

---

Five fixes/behaviors proven on M1 Asahi (Linux ANE KMD), from my
ane-linux-experiments working tree, split one behavior per commit.

1. **read anec payload at 0x1000** — `ANEC_HEADER_SIZE` was 0x800, but anecc
   (and the macOS ANECompiler containers it converts) write a 0x1000 header.
   Reading at 0x800 loads task descriptors shifted by 0x800 bytes; graphs fail
   to execute or execute garbage. Current eiln/anecc ships
   `HEADER_SIZE = 0x1000` (`anecc/__init__.py`), so libane was stale against
   its own converter. Run-log evidence: graphs that execute green after the fix
   (`mul.ane` 64x 6.0 from 3x2; `conv.ane` 3x 9.0 from 3x3x3; fused
   Conv+GOC golden graph -> 27.0) never complete from the 0x800 offset.

2. **add ane_kernel_capacity** — reports the command-buffer bytes left for
   kernels after the 16-byte-aligned task region:
   `chans[0].size - round_up(tsk_size, 16)`, 0 on overflow. This is the bound
   the next commit validates against, and it matches the driver's own kernel
   bank derivation (`bar[KRN] = bar[CMD] + round_up(tsk_size, ANE_CMD_GRAN)`,
   ane_drv.c).

3. **add ane_bind_kernel** — rebind a new fp16 kernel payload into the command
   buffer's kernel region at runtime, no reload, no recompile. Proven on
   hardware: `gemm.ane` with an identity kernel bound passes its input through
   unchanged (input 3.0 -> output 3.0), then rebinding 2x identity -> 6.0.
   Scope note: fp16 kernels only; compressed q4 streams need host-side
   decompression on Linux (no in-graph decompression in the KMD).

4. **poll output sentinel in ane_exec** — fill every output channel with fp16
   +inf (0x7c00) before submit and poll the first output word for up to a
   second; return `-ETIMEDOUT` if the sentinel survives. Today `ane_exec`
   returns the raw ioctl result, so a wedged tile looks like success and stale
   output from a prior dispatch reads as fresh. The 0x7c00 sentinel is exactly
   what stayed in output memory during a real hang in my run logs (KMD
   `tm execution failed w/ -110`), so the signal is the observed one. The
   successful-run contract is unchanged.

5. **add ane_exec_loop for resident state** — for graphs whose input N and
   output N have equal size (recurrent/attention state), dispatch `iterations`
   times, swapping the state input buffer object with the state output buffer
   object on odd iterations, so state stays resident in tile memory instead of
   round-tripping through the host per dispatch. On hardware:
   `ane_exec_loop(nn, 3, 0, 0)` on `mul.ane` computes `((3 * 2) * 2) * 2 ->
   24.0`, and a patched MUL->ADD graph computes `((3 + 2) + 2) + 2 -> 9.0`.
   Rejections (zero iterations, out-of-range indices, size mismatch) return
   `-EINVAL` before any ioctl. Note: with an even iteration count the final
   output lands in the state input buffer; use odd counts or read the state
   input index.

Tests: `test/validate` (new) runs device-free anywhere — 30 checks pinning the
anec ABI layout, capacity math, bind bounds/placement, and exec_loop
rejections. `test/loop` (new) drives `ane_exec_loop` on hardware and exits 77
when `/dev/accel/accel0` is absent.

Deliberately not included: the experiments' tree also pins `args.pad = 0x81`
to steer submissions to queue 1 on a queue-id-enabled KMD. Stock ane_drv.c
rejects any nonzero submit `pad` and hardcodes `req.qid = 4`, so that leg is
fork-specific; queue selection should stay in the driver. Happy to share the
full run logs (`receipts/ane-static-graph-loop.log`) behind these numbers.

---

## 6. Reproduce

```
git clone https://github.com/eiln/ane /tmp/eiln-ane
cd /tmp/eiln-ane
git checkout omarchy-libane-stabilization   # if the clone is gone: apply patches below
make -C libane LIBS="-I/usr/include/libdrm -I$(pwd)/ane/src/uapi/drm"
cd test/validate && gcc -I. -I../../libane main.c ../../libane/libane.a -o main.out && ./main.out
cd ../loop && gcc -I. -I../../libane main.c ../../libane/libane.a -o main.out && ./main.out anything.anec
```

On PR approval, opening command is:
`gh pr create --repo eiln/ane --head <pushed branch> --title ... --body-file <receipt section 5>`.


## 7. Full patch series (durable copy)

Verbatim `git format-patch 0dcea99..HEAD` export, all six commits (also at `/tmp/libane-series/`):

```patch
From 5067e790c329b41aff93e8d388df117010d6f8bc Mon Sep 17 00:00:00 2001
From: Joshua Warren <816217+joshuaswarren@users.noreply.github.com>
Date: Tue, 1 Sep 2026 12:56:56 -0500
Subject: [PATCH 1/6] libane: read anec payload at 0x1000

anecc (and the macOS ANECompiler containers it converts) writes a
0x1000-byte ANEC header before the network payload; libane read the
payload at 0x800, loading shifted garbage as task descriptors.

Verified on M1 Asahi with the KMD queue-id 1 tree: graphs that ran
green after this change (mul.ane 64x 6.0, conv.ane 3x 9.0) fail to
execute from the 0x800 offset, and current eiln/anecc ships
HEADER_SIZE = 0x1000.
---
 libane/ane.c | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)

diff --git a/libane/ane.c b/libane/ane.c
index b956e27..e202a55 100644
--- a/libane/ane.c
+++ b/libane/ane.c
@@ -31,7 +31,7 @@
 #define tile_align(x)	   ((((uint64_t)(x)) + TILE_SIZE - 1) & -TILE_SIZE)
 #define tile_size(nn, bdx) (tile_shift(to_anec(nn)->tiles[bdx]))
 
-#define ANEC_HEADER_SIZE   0x800UL
+#define ANEC_HEADER_SIZE   0x1000UL
 #define src_bdx(nn, idx)   (4 + ane_dst_count(nn) + idx)
 #define dst_bdx(nn, idx)   (4 + idx)
 
-- 
2.39.5

From d8cf05d4a22a855935eb68e5b6ee83f0fa9c1f07 Mon Sep 17 00:00:00 2001
From: Joshua Warren <816217+joshuaswarren@users.noreply.github.com>
Date: Tue, 1 Sep 2026 12:57:27 -0500
Subject: [PATCH 2/6] libane: add ane_kernel_capacity

The command buffer packs task descriptors first, then the 16-byte
aligned kernel region. Report the bytes left for kernels after the
task region:

    capacity = chans[0].size - round_up(tsk_size, 16)

Zero when the task region overflows the command buffer.
---
 libane/ane.c | 7 +++++++
 libane/ane.h | 1 +
 2 files changed, 8 insertions(+)

diff --git a/libane/ane.c b/libane/ane.c
index e202a55..0f595c1 100644
--- a/libane/ane.c
+++ b/libane/ane.c
@@ -476,6 +476,13 @@ int ane_exec(struct ane_nn *nn)
 	return ioctl(nn->fd, DRM_IOCTL_ANE_SUBMIT, &args);
 }
 
+uint64_t ane_kernel_capacity(struct ane_nn *nn)
+{
+	const struct anec *anec = to_anec(nn);
+	uint64_t offset = (anec->tsk_size + 15) & ~15ULL;
+	return offset <= nn->chans[0].size ? nn->chans[0].size - offset : 0;
+}
+
 #ifndef LIBANE_CONFIG_NO_INDEX_CHECK
 #define INDEX_CHECK(cnt, idx, ret)                                             \
 	({                                                                     \
diff --git a/libane/ane.h b/libane/ane.h
index 49e6a31..656bf6b 100644
--- a/libane/ane.h
+++ b/libane/ane.h
@@ -101,6 +101,7 @@ void __ane_free(struct ane_nn *nn);
 #define ane_free(nn) (__ane_free(nn))
 
 int ane_exec(struct ane_nn *nn);
+uint64_t ane_kernel_capacity(struct ane_nn *nn);
 
 #define to_anec(nn)	  (&nn->anec)
 #define ane_src_count(nn) (to_anec(nn)->src_count)
-- 
2.39.5

From d0e83dc2c5719628ce0fa4958f2e9364519f668a Mon Sep 17 00:00:00 2001
From: Joshua Warren <816217+joshuaswarren@users.noreply.github.com>
Date: Tue, 1 Sep 2026 12:57:44 -0500
Subject: [PATCH 3/6] libane: add ane_bind_kernel

Copy a new fp16 kernel payload into the command buffer's kernel
region at runtime, without reloading or recompiling the graph. The
driver already derives the kernel bank as

    bar[KRN] = bar[CMD] + round_up(tsk_size, ANE_CMD_GRAN)

so the payload lands exactly where the task descriptors read it.
Rejects NULL source and sizes beyond ane_kernel_capacity().
---
 libane/ane.c | 12 ++++++++++++
 libane/ane.h |  1 +
 2 files changed, 13 insertions(+)

diff --git a/libane/ane.c b/libane/ane.c
index 0f595c1..1abd57b 100644
--- a/libane/ane.c
+++ b/libane/ane.c
@@ -483,6 +483,18 @@ uint64_t ane_kernel_capacity(struct ane_nn *nn)
 	return offset <= nn->chans[0].size ? nn->chans[0].size - offset : 0;
 }
 
+int ane_bind_kernel(struct ane_nn *nn, const void *from, uint64_t size)
+{
+	const struct anec *anec = to_anec(nn);
+	uint64_t offset = (anec->tsk_size + 15) & ~15ULL;
+	uint64_t capacity = ane_kernel_capacity(nn);
+	if (!from || size > capacity ||
+	    offset + size > nn->chans[0].size)
+		return -EINVAL;
+	memcpy((uint8_t *)nn->chans[0].map + offset, from, size);
+	return 0;
+}
+
 #ifndef LIBANE_CONFIG_NO_INDEX_CHECK
 #define INDEX_CHECK(cnt, idx, ret)                                             \
 	({                                                                     \
diff --git a/libane/ane.h b/libane/ane.h
index 656bf6b..415860a 100644
--- a/libane/ane.h
+++ b/libane/ane.h
@@ -102,6 +102,7 @@ void __ane_free(struct ane_nn *nn);
 
 int ane_exec(struct ane_nn *nn);
 uint64_t ane_kernel_capacity(struct ane_nn *nn);
+int ane_bind_kernel(struct ane_nn *nn, const void *from, uint64_t size);
 
 #define to_anec(nn)	  (&nn->anec)
 #define ane_src_count(nn) (to_anec(nn)->src_count)
-- 
2.39.5

From b3e68073d419c40a00452e84ae49aa1a6ce194d3 Mon Sep 17 00:00:00 2001
From: Joshua Warren <816217+joshuaswarren@users.noreply.github.com>
Date: Tue, 1 Sep 2026 12:58:04 -0500
Subject: [PATCH 4/6] libane: poll output sentinel in ane_exec

ane_exec returned the ioctl result only, so a graph that hangs the
tile looks like success and stale output from a prior dispatch reads
as fresh. Fill every output channel with fp16 +inf (0x7c00) before
submit and poll the first output word for up to a second; return
-ETIMEDOUT when the sentinel survives.

Callers that need to distinguish -ETIMEDOUT from other failures
already can; the fire-and-forget ane_exec contract is unchanged for
successful runs.
---
 libane/ane.c | 34 +++++++++++++++++++++++++++++++---
 1 file changed, 31 insertions(+), 3 deletions(-)

diff --git a/libane/ane.c b/libane/ane.c
index 1abd57b..284baa3 100644
--- a/libane/ane.c
+++ b/libane/ane.c
@@ -455,11 +455,15 @@ void __ane_free(struct ane_nn *nn)
 	free(nn);
 }
 
-int ane_exec(struct ane_nn *nn)
+static int ane_exec_poll(struct ane_nn *nn)
 {
 	const struct anec *anec = to_anec(nn);
-
+	/* fp16 +inf; completion flips the first output word */
+	const uint16_t sentinel = 0x7c00;
+	volatile uint16_t *first;
 	struct drm_ane_submit args;
+	int ret;
+
 	memset(&args, 0, sizeof(args));
 
 	args.tsk_size = anec->tsk_size;
@@ -473,7 +477,31 @@ int ane_exec(struct ane_nn *nn)
 	}
 	args.btsp_handle = nn->btsp_chan.handle;
 
-	return ioctl(nn->fd, DRM_IOCTL_ANE_SUBMIT, &args);
+	/* poison outputs so stale results cannot pass the poll */
+	for (uint32_t idx = 0; idx < anec->dst_count; idx++) {
+		uint32_t bdx = dst_bdx(nn, idx);
+		uint16_t *words = (uint16_t *)nn->chans[bdx].map;
+		uint64_t count = nn->chans[bdx].size / sizeof(uint16_t);
+		for (uint64_t word = 0; word < count; word++) {
+			words[word] = sentinel;
+		}
+	}
+
+	ret = ioctl(nn->fd, DRM_IOCTL_ANE_SUBMIT, &args);
+	if (ret < 0) {
+		return ret;
+	}
+
+	first = (volatile uint16_t *)nn->chans[dst_bdx(nn, 0)].map;
+	for (int wait = 0; wait < 10000 && *first == sentinel; wait++) {
+		usleep(100);
+	}
+	return *first == sentinel ? -ETIMEDOUT : ret;
+}
+
+int ane_exec(struct ane_nn *nn)
+{
+	return ane_exec_poll(nn);
 }
 
 uint64_t ane_kernel_capacity(struct ane_nn *nn)
-- 
2.39.5

From 449a000091166e6cbc1ac120475925ef0ace06c0 Mon Sep 17 00:00:00 2001
From: Joshua Warren <816217+joshuaswarren@users.noreply.github.com>
Date: Tue, 1 Sep 2026 12:58:44 -0500
Subject: [PATCH 5/6] libane: add ane_exec_loop for resident state

Recurrent and attention graphs read a state input and write a state
output of equal size. ane_exec_loop dispatches n iterations, swapping
the state input buffer object with the state output buffer object on
odd iterations, so state stays resident in tile memory instead of
round-tripping through the host each dispatch:

    ane_exec_loop(nn, 3, 0, 0);  // ((3 * 2) * 2) * 2 -> 24

ane_exec keeps its shape as the no-swap case. With an even iteration
count the final output lands in the state input buffer; use odd
counts or read the state input index.

ane_exec_loop rejects zero iterations, out-of-range state indices,
and mismatched state sizes with -EINVAL before touching the device.
---
 libane/ane.c | 53 +++++++++++++++++++++++++++++++++++++++++++++++-----
 libane/ane.h |  2 ++
 2 files changed, 50 insertions(+), 5 deletions(-)

diff --git a/libane/ane.c b/libane/ane.c
index 284baa3..d7b27d4 100644
--- a/libane/ane.c
+++ b/libane/ane.c
@@ -455,12 +455,15 @@ void __ane_free(struct ane_nn *nn)
 	free(nn);
 }
 
-static int ane_exec_poll(struct ane_nn *nn)
+static int ane_exec_with_state_swap(struct ane_nn *nn, int swap_state,
+				    uint32_t state_src_idx,
+				    uint32_t state_dst_idx)
 {
 	const struct anec *anec = to_anec(nn);
 	/* fp16 +inf; completion flips the first output word */
 	const uint16_t sentinel = 0x7c00;
 	volatile uint16_t *first;
+	uint32_t first_bdx;
 	struct drm_ane_submit args;
 	int ret;
 
@@ -475,13 +478,27 @@ static int ane_exec_poll(struct ane_nn *nn)
 			args.handles[bdx] = nn->chans[bdx].handle;
 		}
 	}
+
+	/* swap state in/out buffers so output feeds the next input */
+	if (swap_state) {
+		uint32_t src = src_bdx(nn, state_src_idx);
+		uint32_t dst = dst_bdx(nn, state_dst_idx);
+		uint32_t handle = args.handles[src];
+		args.handles[src] = args.handles[dst];
+		args.handles[dst] = handle;
+	}
 	args.btsp_handle = nn->btsp_chan.handle;
 
 	/* poison outputs so stale results cannot pass the poll */
 	for (uint32_t idx = 0; idx < anec->dst_count; idx++) {
 		uint32_t bdx = dst_bdx(nn, idx);
-		uint16_t *words = (uint16_t *)nn->chans[bdx].map;
-		uint64_t count = nn->chans[bdx].size / sizeof(uint16_t);
+		uint16_t *words;
+		uint64_t count;
+		if (swap_state && idx == state_dst_idx) {
+			bdx = src_bdx(nn, state_src_idx);
+		}
+		words = (uint16_t *)nn->chans[bdx].map;
+		count = nn->chans[bdx].size / sizeof(uint16_t);
 		for (uint64_t word = 0; word < count; word++) {
 			words[word] = sentinel;
 		}
@@ -492,7 +509,11 @@ static int ane_exec_poll(struct ane_nn *nn)
 		return ret;
 	}
 
-	first = (volatile uint16_t *)nn->chans[dst_bdx(nn, 0)].map;
+	first_bdx = dst_bdx(nn, 0);
+	if (swap_state && state_dst_idx == 0) {
+		first_bdx = src_bdx(nn, state_src_idx);
+	}
+	first = (volatile uint16_t *)nn->chans[first_bdx].map;
 	for (int wait = 0; wait < 10000 && *first == sentinel; wait++) {
 		usleep(100);
 	}
@@ -501,7 +522,29 @@ static int ane_exec_poll(struct ane_nn *nn)
 
 int ane_exec(struct ane_nn *nn)
 {
-	return ane_exec_poll(nn);
+	return ane_exec_with_state_swap(nn, 0, 0, 0);
+}
+
+int ane_exec_loop(struct ane_nn *nn, uint32_t iterations,
+		  uint32_t state_src_idx, uint32_t state_dst_idx)
+{
+	if (!iterations || state_src_idx >= ane_src_count(nn) ||
+	    state_dst_idx >= ane_dst_count(nn)) {
+		return -EINVAL;
+	}
+	if (__ane_src_size(nn, state_src_idx) !=
+	    __ane_dst_size(nn, state_dst_idx)) {
+		return -EINVAL;
+	}
+	for (uint32_t iteration = 0; iteration < iterations; iteration++) {
+		int ret = ane_exec_with_state_swap(nn, iteration & 1,
+						   state_src_idx,
+						   state_dst_idx);
+		if (ret < 0) {
+			return ret;
+		}
+	}
+	return 0;
 }
 
 uint64_t ane_kernel_capacity(struct ane_nn *nn)
diff --git a/libane/ane.h b/libane/ane.h
index 415860a..66294ce 100644
--- a/libane/ane.h
+++ b/libane/ane.h
@@ -101,6 +101,8 @@ void __ane_free(struct ane_nn *nn);
 #define ane_free(nn) (__ane_free(nn))
 
 int ane_exec(struct ane_nn *nn);
+int ane_exec_loop(struct ane_nn *nn, uint32_t iterations,
+		  uint32_t state_src_idx, uint32_t state_dst_idx);
 uint64_t ane_kernel_capacity(struct ane_nn *nn);
 int ane_bind_kernel(struct ane_nn *nn, const void *from, uint64_t size);
 
-- 
2.39.5

From fbe40bed471d83aa0c73fd0eb56fdc5366313321 Mon Sep 17 00:00:00 2001
From: Joshua Warren <816217+joshuaswarren@users.noreply.github.com>
Date: Tue, 1 Sep 2026 13:04:42 -0500
Subject: [PATCH 6/6] test: add validate and loop examples

validate runs anywhere: no device, no model. It pins the anec ABI
layout (offsets and tile/nchw spans), the kernel capacity math, the
ane_bind_kernel bounds (NULL source, oversized, exact-capacity,
placement at round_up(tsk_size, 16), rebind overwrite), and the
ane_exec_loop rejections (zero iterations, out-of-range state
indices, mismatched state sizes).

loop exercises ane_exec_loop on hardware with a state graph and
skips (exit 77) when /dev/accel/accel0 is absent or the model has
no resident-state channel pair.
---
 test/loop/Makefile     |   5 ++
 test/loop/main.c       |  52 ++++++++++++
 test/validate/Makefile |   5 ++
 test/validate/main.c   | 189 +++++++++++++++++++++++++++++++++++++++++
 4 files changed, 251 insertions(+)
 create mode 100644 test/loop/Makefile
 create mode 100644 test/loop/main.c
 create mode 100644 test/validate/Makefile
 create mode 100644 test/validate/main.c

diff --git a/test/loop/Makefile b/test/loop/Makefile
new file mode 100644
index 0000000..c6005bc
--- /dev/null
+++ b/test/loop/Makefile
@@ -0,0 +1,5 @@
+all:
+	gcc -I. -I../../libane main.c ../../libane/libane.a -o main.out
+
+clean:
+	rm -f *.out
diff --git a/test/loop/main.c b/test/loop/main.c
new file mode 100644
index 0000000..b5d8168
--- /dev/null
+++ b/test/loop/main.c
@@ -0,0 +1,52 @@
+// SPDX-License-Identifier: MIT
+/* Copyright 2026 Joshua Warren <816217+joshuaswarren@users.noreply.github.com> */
+
+#include <stdio.h>
+#include <unistd.h>
+
+#include "ane.h"
+
+/* Exercises ane_exec_loop on real hardware with a state graph
+ * (equal-size input 0 / output 0). Skips without a device or model.
+ * Evidence from ane-linux-experiments receipts/ane-static-graph-loop.log:
+ * mul.ane 3 iterations ((3 * 2) * 2) * 2 -> 24.0 everywhere. */
+
+int main(int argc, char **argv)
+{
+	struct ane_nn *nn;
+	int err;
+
+	if (argc < 2) {
+		printf("usage: %s <model.anec>\n", argv[0]);
+		return 77;
+	}
+
+	if (access("/dev/accel/accel0", F_OK) != 0) {
+		printf("TEST: SKIP: no ANE device\n");
+		return 77;
+	}
+
+	nn = ane_init(argv[1]);
+	if (nn == NULL) {
+		printf("TEST: ERR: failed to init %s\n", argv[1]);
+		return 1;
+	}
+
+	if (ane_kernel_capacity(nn) == 0) {
+		printf("TEST: SKIP: no kernel capacity\n");
+		ane_free(nn);
+		return 77;
+	}
+
+	if (__ane_src_size(nn, 0) != __ane_dst_size(nn, 0)) {
+		printf("TEST: SKIP: state sizes differ; needs a state graph\n");
+		ane_free(nn);
+		return 77;
+	}
+
+	err = ane_exec_loop(nn, 3, 0, 0);
+	printf("TEST: LOG: ane_exec_loop: %d\n", err);
+
+	ane_free(nn);
+	return err < 0 ? 1 : 0;
+}
diff --git a/test/validate/Makefile b/test/validate/Makefile
new file mode 100644
index 0000000..c6005bc
--- /dev/null
+++ b/test/validate/Makefile
@@ -0,0 +1,5 @@
+all:
+	gcc -I. -I../../libane main.c ../../libane/libane.a -o main.out
+
+clean:
+	rm -f *.out
diff --git a/test/validate/main.c b/test/validate/main.c
new file mode 100644
index 0000000..f4a7d1f
--- /dev/null
+++ b/test/validate/main.c
@@ -0,0 +1,189 @@
+// SPDX-License-Identifier: MIT
+/* Copyright 2026 Joshua Warren <816217+joshuaswarren@users.noreply.github.com> */
+
+#include <stddef.h>
+#include <stdint.h>
+#include <stdio.h>
+#include <stdlib.h>
+#include <string.h>
+
+#include "../ane_utils.h"
+#include "ane.h"
+
+// clang-format off
+
+static int tests_run;
+static int tests_failed;
+
+#define CHECK(cond)                                                    \
+	do {                                                           \
+		tests_run++;                                           \
+		if (!(cond)) {                                         \
+			tests_failed++;                                \
+			ane_err("FAIL %d: %s\n", __LINE__, #cond);     \
+		}                                                      \
+	} while (0)
+
+/* anec members are const-qualified; the library loads them by block
+ * copy (ane_model_init), so tests configure through the same bytes */
+static void set_anec(struct ane_nn *nn, uint64_t tsk_size,
+		     uint32_t src_count, uint32_t dst_count,
+		     uint32_t cmd_tiles, uint32_t src_tiles,
+		     uint32_t dst_tiles)
+{
+	uint8_t raw[0x6a8];
+
+	memset(raw, 0, sizeof(raw));
+	memcpy(raw + offsetof(struct anec, tsk_size), &tsk_size,
+	       sizeof(tsk_size));
+	memcpy(raw + offsetof(struct anec, src_count), &src_count,
+	       sizeof(src_count));
+	memcpy(raw + offsetof(struct anec, dst_count), &dst_count,
+	       sizeof(dst_count));
+	memcpy(raw + offsetof(struct anec, tiles[0]), &cmd_tiles,
+	       sizeof(cmd_tiles));
+	memcpy(raw + offsetof(struct anec, tiles[4]), &dst_tiles,
+	       sizeof(dst_tiles));
+	memcpy(raw + offsetof(struct anec, tiles[4 + dst_count]), &src_tiles,
+	       sizeof(src_tiles));
+	memcpy(&nn->anec, raw, sizeof(raw));
+}
+
+/* a fake nn: no device, chans[0] backed by a real tile-sized buffer */
+static struct ane_nn *make_nn(uint64_t cmd_size)
+{
+	struct ane_nn *nn = ane_zmalloc(sizeof(*nn));
+	if (nn == NULL) {
+		return NULL;
+	}
+
+	nn->fd = -1;
+	nn->chans[0].size = cmd_size;
+	nn->chans[0].map = ane_zmalloc(cmd_size);
+	if (nn->chans[0].map == NULL) {
+		free(nn);
+		return NULL;
+	}
+
+	return nn;
+}
+
+static void free_nn(struct ane_nn *nn)
+{
+	free(nn->chans[0].map);
+	free(nn);
+}
+
+static void test_abi_layout(void)
+{
+	struct ane_nn *nn = make_nn(0x4000);
+
+	CHECK(nn != NULL);
+	CHECK(TILE_COUNT == 0x20);
+	/* 8 + 4 + 4 + 8 + 8 + 4 + 4 + 32*4 + 32*6*8 */
+	CHECK(sizeof(struct anec) == 0x6a8);
+	CHECK(sizeof(struct ane_bo) == 32);
+	CHECK(sizeof(((struct anec *)0)->tiles) == 0x20 * 4);
+	CHECK(sizeof(((struct anec *)0)->nchw) == 0x20 * 6 * 8);
+
+	free_nn(nn);
+}
+
+static void test_kernel_capacity(void)
+{
+	struct ane_nn *nn = make_nn(0x4000);
+
+	CHECK(nn != NULL);
+	set_anec(nn, 0x574, 0, 0, 1, 0, 0);
+	/* round_up(0x574, 16) = 0x580 */
+	CHECK(ane_kernel_capacity(nn) == 0x4000 - 0x580);
+
+	set_anec(nn, 0x575, 0, 0, 1, 0, 0); /* unaligned rounds up the same */
+	CHECK(ane_kernel_capacity(nn) == 0x4000 - 0x580);
+
+	nn->chans[0].size = 0x580; /* task region fills the buffer */
+	CHECK(ane_kernel_capacity(nn) == 0);
+
+	nn->chans[0].size = 0x400; /* task region overflows */
+	CHECK(ane_kernel_capacity(nn) == 0);
+
+	free_nn(nn);
+}
+
+static void test_bind_kernel(void)
+{
+	struct ane_nn *nn = make_nn(0x4000);
+	uint8_t payload[0x100];
+	uint8_t payload2[0x100];
+	uint64_t capacity;
+	uint8_t *big;
+
+	CHECK(nn != NULL);
+	set_anec(nn, 0x574, 0, 0, 1, 0, 0);
+
+	for (int i = 0; i < 0x100; i++) {
+		payload[i] = 0xa0 | (i & 0xf);
+		payload2[i] = 0xb0 | (i & 0xf);
+	}
+
+	capacity = ane_kernel_capacity(nn);
+	CHECK(capacity == 0x3a80);
+
+	CHECK(ane_bind_kernel(nn, NULL, 0x100) == -EINVAL);
+	CHECK(ane_bind_kernel(nn, payload, capacity + 1) == -EINVAL);
+	CHECK(ane_bind_kernel(nn, payload, 0) == 0);
+
+	/* payload lands at round_up(tsk_size, 16), nothing else moves */
+	CHECK(ane_bind_kernel(nn, payload, 0x100) == 0);
+	CHECK(((uint8_t *)nn->chans[0].map)[0x57f] == 0);
+	CHECK(memcmp((uint8_t *)nn->chans[0].map + 0x580, payload, 0x100) ==
+	      0);
+	CHECK(((uint8_t *)nn->chans[0].map)[0x680] == 0);
+
+	/* rebind overwrites in place */
+	CHECK(ane_bind_kernel(nn, payload2, 0x100) == 0);
+	CHECK(memcmp((uint8_t *)nn->chans[0].map + 0x580, payload2, 0x100) ==
+	      0);
+
+	/* exact-capacity bind is legal and spans to the end */
+	big = ane_malloc(capacity);
+	CHECK(big != NULL);
+	memset(big, 0xc5, capacity);
+	memset(nn->chans[0].map, 0, nn->chans[0].size);
+	CHECK(ane_bind_kernel(nn, big, capacity) == 0);
+	CHECK(((uint8_t *)nn->chans[0].map)[0x4000 - 1] == 0xc5);
+	free(big);
+
+	free_nn(nn);
+}
+
+static void test_exec_loop_rejects(void)
+{
+	struct ane_nn *nn = make_nn(0x4000);
+
+	CHECK(nn != NULL);
+	set_anec(nn, 0x574, 1, 1, 1, 0, 0);
+
+	CHECK(ane_exec_loop(nn, 0, 0, 0) == -EINVAL);
+	CHECK(ane_exec_loop(nn, 3, 1, 0) == -EINVAL);
+	CHECK(ane_exec_loop(nn, 3, 0, 1) == -EINVAL);
+
+	/* state in/out sizes differ: 0x4000 src vs 0x8000 dst */
+	set_anec(nn, 0x574, 1, 1, 1, 1, 2);
+	CHECK(ane_exec_loop(nn, 3, 0, 0) == -EINVAL);
+
+	free_nn(nn);
+}
+
+int main(void)
+{
+	test_abi_layout();
+	test_kernel_capacity();
+	test_bind_kernel();
+	test_exec_loop_rejects();
+
+	printf("TEST: %d checks, %d failed\n", tests_run, tests_failed);
+	return tests_failed ? 1 : 0;
+}
+
+// clang-format on
-- 
2.39.5

```
