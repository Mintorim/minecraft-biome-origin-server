// biome_origin_server.c
//
// File last edited for 26.3
//
// Requires Linux/macOS
//
// Depends on Cubiomes. Requires access to specific Cubiomes files.
//
// Copyright (c) 2026 Mintorim. Public domain under CC0 1.0 Universal.
// Portions of the code was created with AI assistance.
//
// Local use only: the HTTP interface binds to 127.0.0.1 and has no authentication.
//
// Tallies the biome at block (0,0)
// for every consecutive seed. Results on http://localhost:8787
//
// Build:
//   clang -O3 -mcpu=native -flto -o biome_origin biome_origin_server.c libcubiomes.a -lm -lpthread   (Apple Silicon)
//   clang -O3 -march=native -flto -o biome_origin biome_origin_server.c libcubiomes.a -lm -lpthread  (Intel / Linux)
// Run:  ./biome_origin   then open http://localhost:8787

#include "generator.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define MC_VERSION        MC_NEWEST
#define HTTP_PORT         8787
#define MAX_ID            512
#define MAX_THREADS_CAP   64
#define BATCH             1024   // seeds per lock acquisition; also the stop granularity
#define SAVE_INTERVAL_SEC 30
#define MIN_Y_BLOCK       -64
#define MAX_Y_BLOCK       319
#define DEFAULT_Y_BLOCK   252
#define Z_95              1.959963985

typedef enum { YMODE_ADAPTIVE = 0, YMODE_FIXED = 1 } YMode;

// ------------------------------------------------------------ shared state
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_runLock = PTHREAD_MUTEX_INITIALIZER;  // serialises start/stop (taken before g_lock)
static pthread_t g_threads[MAX_THREADS_CAP];
static int  g_running = 0, g_stop = 0, g_numWorkers = 0, g_workersAlive = 0;
static int  g_safeSeed = 0;            // 1 = fall back to applySeed() (set by startup self-check)

static int  g_loaded = 0;              // a run identity has been selected
static YMode g_yMode = YMODE_FIXED;
static int  g_yBlock = DEFAULT_Y_BLOCK, g_yQuart = DEFAULT_Y_BLOCK >> 2;
static char g_path[300] = "";
static long g_nextSeed = 0, g_total = 0;
static long g_counts[MAX_ID];
static double g_lastSave = 0;

// Reference snapshots for the "delta vs earlier" column. Only touched when
// results are requested, never by the workers.
typedef struct { int valid; long total; long counts[MAX_ID]; } Snap;
static Snap g_ref, g_pend;

static long g_rateTotal = 0;
static double g_rateTime = 0;

static volatile sig_atomic_t g_quit = 0;

static double nowSec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + t.tv_nsec * 1e-9;
}

static long detectCoreCount(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 1;
    if (n > MAX_THREADS_CAP) n = MAX_THREADS_CAP;
    return n;
}

// ------------------------------------------------------------ checkpoints
static void checkpointPath(YMode m, int yBlock, char *out, size_t len) {
    char suf[24] = "";
    if (m == YMODE_ADAPTIVE) snprintf(suf, sizeof(suf), "_adaptivesurface");
    else if (yBlock != DEFAULT_Y_BLOCK) snprintf(suf, sizeof(suf), "_y%d", yBlock);
    snprintf(out, len, "border_state_origin%s.txt", suf);
}

// Periodic saves record the claimed seed counter, so a crash can leave a gap
// of unprocessed seeds (harmless for statistics, never a double count). A
// clean stop has no gap.
static void saveCheckpointLocked(void) {
    if (!g_path[0]) return;
    char tmp[sizeof(g_path) + 8];
    snprintf(tmp, sizeof(tmp), "%s.tmp", g_path);
    FILE *f = fopen(tmp, "w");
    if (!f) return;
    fprintf(f, "nextseed %ld\n", g_nextSeed);
    for (int i = 0; i < MAX_ID; i++)
        if (g_counts[i] > 0) fprintf(f, "count %d %ld\n", i, g_counts[i]);
    int bad = fflush(f) != 0;
    bad |= fclose(f) != 0;
    if (bad || rename(tmp, g_path) != 0) remove(tmp);
}

static void clearStateLocked(void) {
    memset(g_counts, 0, sizeof(g_counts));
    g_total = 0;
    g_nextSeed = 0;
    g_ref.valid = g_pend.valid = 0;
}

// Total is the sum of the counts, so older origin checkpoints load as-is.
static void loadCheckpointLocked(const char *path) {
    clearStateLocked();
    FILE *f = fopen(path, "r");
    if (!f) return;
    char line[128];
    long v; int id;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "nextseed %ld", &v) == 1 && v >= 0) g_nextSeed = v;
        else if (sscanf(line, "count %d %ld", &id, &v) == 2 && id >= 0 && id < MAX_ID && v > 0) {
            g_counts[id] = v;
            g_total += v;
        }
    }
    fclose(f);
}

// ------------------------------------------------------------ sampling
// Surface mode reads the biome at the terrain height; fixed mode at one Y.
static inline int floorDiv4(int a) { return a >> 2; }

// Seeds the climate noise directly. Skips applySeed()'s extra per-seed work
// that is only needed for scales finer than 4 (e.g. the Voronoi hash).
static inline void seedGen(Generator *g, uint64_t s, int safe) {
    if (safe) applySeed(g, DIM_OVERWORLD, s);
    else { setBiomeSeed(&g->bn, s, 0); g->seed = s; }
}

static inline int biomeAtOrigin(Generator *g, int adaptive, int yQuart, int64_t *np, uint64_t *hint) {
    if (adaptive) {
        float h;
        if (mapApproxHeight(&h, NULL, g, NULL, 0, 0, 1, 1) != 0) return -1;
        yQuart = floorDiv4((int) lroundf(h));
    }
    return sampleBiomeNoise(&g->bn, np, 0, yQuart, 0, hint, 0);
}

// Startup check: the fast path must match the library's own applySeed() +
// getBiomeAt() / mapApproxHeight() results, otherwise workers use applySeed().
static int fastPathMatches(void) {
    Generator a, b;
    setupGenerator(&a, MC_VERSION, 0);
    setupGenerator(&b, MC_VERSION, 0);
    a.dim = b.dim = DIM_OVERWORLD;
    int64_t np[NP_MAX];
    for (uint64_t i = 0; i < 48; i++) {
        uint64_t s = i * 1000003ULL + i * i;
        applySeed(&a, DIM_OVERWORLD, s);
        seedGen(&b, s, 0);
        for (int y = -16; y <= 63; y += 13)
            if (getBiomeAt(&a, 4, 0, y, 0) != sampleBiomeNoise(&b.bn, np, 0, y, 0, NULL, 0)) return 0;
        float ha, hb;
        int ra = mapApproxHeight(&ha, NULL, &a, NULL, 0, 0, 1, 1);
        int rb = mapApproxHeight(&hb, NULL, &b, NULL, 0, 0, 1, 1);
        if (ra != rb || (ra == 0 && ha != hb)) return 0;
    }
    return 1;
}

static void *workerFn(void *arg) {
    (void) arg;
    Generator gen;
    setupGenerator(&gen, MC_VERSION, 0);
    gen.dim = DIM_OVERWORLD;

    pthread_mutex_lock(&g_lock);
    int adaptive = (g_yMode == YMODE_ADAPTIVE);
    int yQuart = g_yQuart;
    pthread_mutex_unlock(&g_lock);

    const int safe = g_safeSeed;
    uint32_t local[MAX_ID];
    long n = 0, seed = 0;
    int64_t np[NP_MAX];
    uint64_t hint = 0;   // last biome-tree leaf; speeds up the next search, result unchanged
    memset(local, 0, sizeof(local));

    for (;;) {
        // One lock per batch: merge the finished batch, then claim the next.
        pthread_mutex_lock(&g_lock);
        if (n) {
            for (int i = 0; i < MAX_ID; i++) g_counts[i] += local[i];
            g_total += n;
        }
        if (g_stop) { pthread_mutex_unlock(&g_lock); break; }
        seed = g_nextSeed;
        g_nextSeed += BATCH;
        double t = nowSec();
        if (t - g_lastSave >= SAVE_INTERVAL_SEC) { g_lastSave = t; saveCheckpointLocked(); }
        pthread_mutex_unlock(&g_lock);

        memset(local, 0, sizeof(local));
        n = 0;
        for (long s = seed; s < seed + BATCH; s++) {
            seedGen(&gen, (uint64_t) s, safe);
            int id = biomeAtOrigin(&gen, adaptive, yQuart, np, &hint);
            if ((unsigned) id < MAX_ID) { local[id]++; n++; }
        }
    }

    pthread_mutex_lock(&g_lock);
    if (--g_workersAlive == 0) {
        saveCheckpointLocked();
        g_running = 0;
    }
    pthread_mutex_unlock(&g_lock);
    return NULL;
}

// seedOverride < 0 continues from the checkpoint. Returns NULL or an error.
static const char *startSampling(YMode yMode, int yBlock, long seedOverride, int threads) {
    if (threads < 1) threads = 1;
    if (threads > MAX_THREADS_CAP) threads = MAX_THREADS_CAP;
    if (yBlock < MIN_Y_BLOCK) yBlock = MIN_Y_BLOCK;
    if (yBlock > MAX_Y_BLOCK) yBlock = MAX_Y_BLOCK;

    pthread_mutex_lock(&g_runLock);
    pthread_mutex_lock(&g_lock);
    if (g_running) {
        pthread_mutex_unlock(&g_lock);
        pthread_mutex_unlock(&g_runLock);
        return "Already running. Stop it first.";
    }

    int changed = !g_loaded || yMode != g_yMode || (yMode == YMODE_FIXED && yBlock != g_yBlock);
    if (changed) {
        g_loaded = 1;
        g_yMode = yMode;
        g_yBlock = yBlock;
        g_yQuart = floorDiv4(yBlock);
        checkpointPath(yMode, yBlock, g_path, sizeof(g_path));
        loadCheckpointLocked(g_path);
    }
    if (seedOverride >= 0) g_nextSeed = seedOverride;

    g_stop = 0;
    g_lastSave = nowSec();
    g_running = 1;
    g_numWorkers = g_workersAlive = 0;

    int created = 0;
    while (created < threads &&
           pthread_create(&g_threads[created], NULL, workerFn, NULL) == 0)
        created++;
    g_numWorkers = g_workersAlive = created;

    const char *err = NULL;
    if (created != threads) {
        g_stop = 1;
        if (created == 0) g_running = 0;
        err = "Could not create all worker threads.";
    }
    pthread_mutex_unlock(&g_lock);
    if (err) for (int i = 0; i < created; i++) pthread_join(g_threads[i], NULL);
    pthread_mutex_unlock(&g_runLock);
    return err;
}

static void stopSampling(void) {
    pthread_mutex_lock(&g_runLock);
    pthread_mutex_lock(&g_lock);
    int n = g_running ? g_numWorkers : 0;
    g_stop = 1;
    pthread_mutex_unlock(&g_lock);
    for (int i = 0; i < n; i++) pthread_join(g_threads[i], NULL);
    pthread_mutex_unlock(&g_runLock);
}

// ------------------------------------------------------------ statistics / JSON
// Wilson score interval; p and the bounds are fractions (0..1).
static void wilson(double p, double n, double *lo, double *hi) {
    if (n <= 0) { *lo = 0; *hi = 1; return; }
    double z2 = Z_95 * Z_95, d = 1.0 + z2 / n;
    double c = (p + z2 / (2.0 * n)) / d;
    double h = (Z_95 / d) * sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n));
    *lo = c - h < 0 ? 0 : c - h;
    *hi = c + h > 1 ? 1 : c + h;
}

typedef struct { int id; long count; } Entry;
static int cmpDesc(const void *a, const void *b) {
    long x = ((const Entry *) a)->count, y = ((const Entry *) b)->count;
    return (y > x) - (y < x);
}

// Delta compares against a snapshot at least max(200, 5%) samples old.
static char *buildStatusJson(void) {
    Entry ent[MAX_ID];
    long refCounts[MAX_ID];
    int n = 0, haveRef = 0;
    pthread_mutex_lock(&g_lock);
    int running = g_running, workers = g_numWorkers;
    long nextSeed = g_nextSeed, total = g_total;
    YMode ym = g_yMode; int yb = g_yBlock;
    for (int i = 0; i < MAX_ID; i++)
        if (g_counts[i] > 0) { ent[n].id = i; ent[n].count = g_counts[i]; n++; }

    long look = total / 20;
    if (look < 200) look = 200;
    if (!g_pend.valid) {
        g_pend.valid = 1; g_pend.total = total;
        memcpy(g_pend.counts, g_counts, sizeof(g_counts));
    } else if (total - g_pend.total >= look) {
        g_ref = g_pend;
        g_pend.total = total;
        memcpy(g_pend.counts, g_counts, sizeof(g_counts));
    }
    double tNow = nowSec(), rate = 0;
    if (g_rateTime > 0 && total >= g_rateTotal && tNow - g_rateTime > 0.05)
        rate = (double) (total - g_rateTotal) / (tNow - g_rateTime);
    g_rateTotal = total; g_rateTime = tNow;
    long refTotal = g_ref.total;
    if (g_ref.valid && refTotal > 0) {
        haveRef = 1;
        memcpy(refCounts, g_ref.counts, sizeof(refCounts));
    }
    pthread_mutex_unlock(&g_lock);

    qsort(ent, n, sizeof(Entry), cmpDesc);

    size_t cap = 512 + (size_t) n * 220;
    char *buf = malloc(cap);
    size_t off = 0;
    char height[32];
    if (ym == YMODE_ADAPTIVE) snprintf(height, sizeof(height), "Surface");
    else snprintf(height, sizeof(height), "Y=%d", yb);
    off += snprintf(buf + off, cap - off,
        "{\"running\":%s,\"workers\":%d,\"nextSeed\":%ld,\"total\":%ld,\"rate\":%.0f,\"height\":\"%s\",\"results\":[",
        running ? "true" : "false", workers, nextSeed, total, rate, height);

    for (int i = 0; i < n && off + 260 < cap; i++) {
        double p = total > 0 ? (double) ent[i].count / total : 0.0;
        double lo, hi;
        wilson(p, (double) total, &lo, &hi);
        const char *name = biome2str(MC_VERSION, ent[i].id);
        off += snprintf(buf + off, cap - off,
            "%s{\"name\":\"%s\",\"count\":%ld,\"pct\":%.4f,\"lo\":%.4f,\"hi\":%.4f,\"d\":",
            i ? "," : "", name ? name : "?", ent[i].count, 100.0 * p, 100.0 * lo, 100.0 * hi);
        if (haveRef)
            off += snprintf(buf + off, cap - off, "%.4f}",
                100.0 * p - 100.0 * (double) refCounts[ent[i].id] / (double) refTotal);
        else
            off += snprintf(buf + off, cap - off, "null}");
    }
    snprintf(buf + off, cap - off, "]}");
    return buf;
}

// ------------------------------------------------------------ HTTP
static int writeAll(int fd, const char *p, size_t len) {
    while (len > 0) {
        ssize_t w = write(fd, p, len);
        if (w <= 0) return 0;
        p += w;
        len -= (size_t) w;
    }
    return 1;
}

static void sendText(int fd, const char *ctype, const char *body) {
    char head[192];
    size_t len = strlen(body);
    int h = snprintf(head, sizeof(head),
        "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
        ctype, len);
    if (writeAll(fd, head, (size_t) h)) writeAll(fd, body, len);
}

static int hexNibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int getParam(const char *query, const char *key, char *out, size_t outlen) {
    if (!query) return 0;
    size_t kl = strlen(key);
    const char *p = query;
    while (p && *p) {
        if (strncmp(p, key, kl) == 0 && p[kl] == '=') {
            p += kl + 1;
            size_t i = 0;
            while (*p && *p != '&' && i + 1 < outlen) {
                if (*p == '%' && p[1] && p[2] && hexNibble(p[1]) >= 0 && hexNibble(p[2]) >= 0) {
                    out[i++] = (char) ((hexNibble(p[1]) << 4) | hexNibble(p[2]));
                    p += 3;
                } else {
                    out[i++] = (*p == '+') ? ' ' : *p;
                    p++;
                }
            }
            out[i] = 0;
            return 1;
        }
        p = strchr(p, '&');
        if (p) p++;
    }
    return 0;
}

static void parseHeight(const char *query, YMode *m, int *y) {
    char b[32] = "";
    *m = (getParam(query, "ymode", b, sizeof(b)) && strcasecmp(b, "adaptive") == 0)
         ? YMODE_ADAPTIVE : YMODE_FIXED;
    *y = DEFAULT_Y_BLOCK;
    if (getParam(query, "y", b, sizeof(b)) && b[0]) *y = atoi(b);
    if (*y < MIN_Y_BLOCK) *y = MIN_Y_BLOCK;
    if (*y > MAX_Y_BLOCK) *y = MAX_Y_BLOCK;
}

static const char *PAGE_HTML =
"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Biome Percentages at (0,0)</title><style>"
":root{color-scheme:dark;--bg:#1b1c1e;--panel:#242527;--p2:#2c2d30;--bd:#3a3b3e;--tx:#e8e8ea;--mu:#a3a4a8;--ac:#5fb3ff}"
"*{box-sizing:border-box}"
"body{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:860px;margin:32px auto;padding:0 16px;color:var(--tx);background:var(--bg)}"
"h1{font-size:20px;font-weight:600}"
".row{margin:10px 0;display:flex;align-items:center;flex-wrap:wrap;gap:8px}"
"label{display:inline-block;min-width:120px;color:var(--mu)}"
"input,select{padding:7px 8px;font-size:14px;background:var(--p2);color:var(--tx);border:1px solid var(--bd);border-radius:6px}"
"button{padding:8px 16px;font-size:14px;margin-right:8px;cursor:pointer;background:var(--p2);color:var(--tx);border:1px solid var(--bd);border-radius:6px}"
"button:hover{background:#37383b}"
"button.primary{background:var(--ac);color:#0b1622;border-color:var(--ac);font-weight:600}"
"#st{background:var(--panel);padding:12px;border-radius:8px;margin:16px 0;font-family:ui-monospace,Menlo,monospace;font-size:13px;border:1px solid var(--bd)}"
"table{border-collapse:collapse;width:100%}"
"td,th{padding:6px 8px;border-bottom:1px solid var(--bd);text-align:left;font-size:13px}"
"th{color:var(--mu);font-weight:600}"
"#err{color:#e0665f;min-height:1em;font-size:13px}"
".hint{color:var(--mu);font-size:12px}"
"</style></head><body>"
"<h1>Biome Percentages at (0,0)</h1>"

"<div class='row'><label>Sample height</label>"
"<select id='ym' onchange='heightChanged()'><option value='adaptive'>Surface</option><option value='fixed' selected>Fixed Y</option></select>"
"<span id='yw'><input type='number' id='y' style='width:90px'> <span class='hint'>block Y, -64 to 319</span></span></div>"

"<div class='row'><label>Start seed</label><input type='number' id='sd' placeholder='blank = continue'></div>"
"<div class='row'><label>Threads</label><input type='number' id='th' min='1' style='width:80px'></div>"

"<div class='row'><label>Results update</label>"
"<select id='um' onchange='schedule()'><option value='manual' selected>Manual (fastest)</option><option value='auto'>Auto</option></select>"
"<span id='usw' style='display:none'>every <input type='number' id='us' value='2' min='0.25' step='0.25' style='width:70px' onchange='schedule()'> s</span></div>"

"<div class='row'>"
"<button class='primary' onclick='start()'>Start</button>"
"<button onclick='stop()'>Stop</button>"
"<button id='ub' onclick='update()'>Update results</button>"
"<button onclick='reset()'>Reset</button></div>"
"<div id='err'></div>"
"<div id='st'>Stopped</div>"

"<table><thead><tr><th>Biome</th><th>Count</th><th>%</th><th>95% CI (%)</th><th>Delta vs earlier</th></tr></thead>"
"<tbody id='rows'></tbody></table>"

"<script>"
"const $=id=>document.getElementById(id);"
"let timer=null;"
"const get=async p=>(await fetch(p)).json();"
"function q(){return new URLSearchParams({ymode:$('ym').value,y:$('y').value||'252'});}"
"function heightChanged(){$('yw').style.display=$('ym').value==='fixed'?'':'none';}"
"function render(j){"
"  $('st').innerHTML=(j.running?'Running on '+j.workers+' thread(s)':'Stopped')+' | Height: '+j.height+'<br>'"
"    +'Samples: '+j.total.toLocaleString()+' | Next seed: '+j.nextSeed.toLocaleString()"
"    +(j.running&&j.rate?' | '+j.rate.toLocaleString()+' seeds/s (since last update)':'');"
"  $('rows').innerHTML=j.results.map(r=>'<tr><td>'+r.name+'</td><td>'+r.count.toLocaleString()+'</td><td>'+r.pct.toFixed(4)+'</td><td>'"
"    +r.lo.toFixed(4)+' - '+r.hi.toFixed(4)+'</td><td>'+(r.d===null?'-':(r.d>=0?'+':'')+r.d.toFixed(4))+'</td></tr>').join('');"
"  return j.running;"
"}"
"function stopTimer(){if(timer){clearInterval(timer);timer=null;}}"
"async function update(){try{if(!render(await get('/api/status')))stopTimer();}catch(e){}}"
"function schedule(){"
"  stopTimer();"
"  const auto=$('um').value==='auto';"
"  $('ub').style.display=auto?'none':'';"
"  $('usw').style.display=auto?'':'none';"
"  if(auto)timer=setInterval(update,Math.max(0.25,+$('us').value||2)*1000);"
"}"
"async function start(){"
"  const p=q();p.set('threads',$('th').value||'1');if($('sd').value)p.set('seed',$('sd').value);"
"  const j=await get('/api/start?'+p);$('err').textContent=j.error||'';"
"  if(!j.error){await update();schedule();}"
"}"
"async function stop(){await get('/api/stop');await update();stopTimer();}"
"async function reset(){"
"  if(!confirm('Delete saved totals for this sample height?'))return;"
"  const j=await get('/api/reset?'+q());$('err').textContent=j.error||'';await update();"
"}"
"(async()=>{"
"  const i=await get('/api/info');"
"  $('th').value=i.threads;$('y').value=i.defaultY;$('y').min=i.minY;$('y').max=i.maxY;"
"  heightChanged();await update();schedule();"
"})();"
"</script></body></html>";

static void handleConnection(int fd) {
    char buf[2048];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    if (n <= 0) { close(fd); return; }
    buf[n] = 0;

    char method[8] = "", path[512] = "";
    sscanf(buf, "%7s %511s", method, path);
    char *query = strchr(path, '?');
    if (query) *query++ = 0;

    char resp[256];
    if (strcmp(path, "/") == 0) {
        sendText(fd, "text/html; charset=utf-8", PAGE_HTML);
    } else if (strcmp(path, "/api/info") == 0) {
        snprintf(resp, sizeof(resp), "{\"threads\":%ld,\"defaultY\":%d,\"minY\":%d,\"maxY\":%d}",
                 detectCoreCount(), DEFAULT_Y_BLOCK, MIN_Y_BLOCK, MAX_Y_BLOCK);
        sendText(fd, "application/json", resp);
    } else if (strcmp(path, "/api/status") == 0) {
        char *s = buildStatusJson();
        sendText(fd, "application/json", s);
        free(s);
    } else if (strcmp(path, "/api/start") == 0) {
        YMode ym; int y;
        parseHeight(query, &ym, &y);
        char b[32];
        long seed = (getParam(query, "seed", b, sizeof(b)) && b[0]) ? atol(b) : -1;
        int threads = (int) detectCoreCount();
        if (getParam(query, "threads", b, sizeof(b)) && b[0]) threads = atoi(b);
        const char *err = startSampling(ym, y, seed, threads);
        if (err) snprintf(resp, sizeof(resp), "{\"error\":\"%s\"}", err);
        else snprintf(resp, sizeof(resp), "{\"ok\":true}");
        sendText(fd, "application/json", resp);
    } else if (strcmp(path, "/api/stop") == 0) {
        stopSampling();
        sendText(fd, "application/json", "{\"ok\":true}");
    } else if (strcmp(path, "/api/reset") == 0) {
        YMode ym; int y;
        parseHeight(query, &ym, &y);
        pthread_mutex_lock(&g_lock);
        if (g_running) {
            pthread_mutex_unlock(&g_lock);
            sendText(fd, "application/json", "{\"error\":\"Stop it before resetting.\"}");
        } else {
            char p[300];
            checkpointPath(ym, y, p, sizeof(p));
            remove(p);
            if (g_loaded && ym == g_yMode && (ym == YMODE_ADAPTIVE || y == g_yBlock))
                clearStateLocked();
            pthread_mutex_unlock(&g_lock);
            sendText(fd, "application/json", "{\"ok\":true}");
        }
    } else {
        sendText(fd, "text/plain", "not found");
    }
    close(fd);
}

static void *connectionThread(void *arg) {
    handleConnection((int) (intptr_t) arg);
    return NULL;
}

static void onSignal(int sig) { (void) sig; g_quit = 1; }

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = onSignal;          // no SA_RESTART: accept() returns so can save and exit
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    g_safeSeed = !fastPathMatches();
    if (g_safeSeed) printf("Note: fast seeding path did not match applySeed(). Using the slower safe path.\n");

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    addr.sin_port = htons(HTTP_PORT);
    if (bind(srv, (struct sockaddr *) &addr, sizeof(addr)) != 0) {
        perror("bind failed (is the port already in use?)");
        return 1;
    }
    listen(srv, 16);
    printf("Open http://localhost:%d  (%ld cores detected, Ctrl+C to quit)\n", HTTP_PORT, detectCoreCount());

    while (!g_quit) {
        int c = accept(srv, NULL, NULL);
        if (c < 0) continue;
        pthread_t t;
        if (pthread_create(&t, NULL, connectionThread, (void *) (intptr_t) c) == 0) pthread_detach(t);
        else close(c);
    }
    stopSampling();   // joins workers, the last one writes the final checkpoint
    return 0;
}