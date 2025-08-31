import math
import sqlite3
import time
import heapq
from functools import lru_cache
from tqdm import tqdm


def _group_orders_by_mpid(orders_by_id):
    groups = {}
    for oid, o in orders_by_id.items():
        pid = o.get("MpId")
        if pid is None:
            continue
        groups.setdefault(pid, []).append(int(oid))
    return groups


def _closeness_scores(orders_by_mpid, warehouse_id, fast_dist):
    s = {}
    pbar = tqdm(total=len(orders_by_mpid), desc="Stage1: closeness", unit="mp")
    for pid, ids in orders_by_mpid.items():
        best = None
        for oid in ids:
            d = fast_dist(int(oid), int(warehouse_id))
            if best is None or d < best:
                best = d
        if best is None:
            best = 10_000_000
        s[pid] = float(best)
        pbar.update(1)
    pbar.close()
    return s


def _pick_k_policy(closeness, tau=0.7):
    items = sorted(closeness.items(), key=lambda kv: kv[1])
    n = len(items)
    k_by = {}
    if n <= 1:
        pid = items[0][0] if items else None
        if pid is not None:
            k_by[pid] = 10
        return k_by
    for rank, (pid, _) in enumerate(items):
        p = rank / (n - 1)
        b_i = 1.0 - p
        k_by[pid] = 15 if b_i >= tau else 10
    return k_by


def _select_portals(orders_by_mpid, warehouse_id, fast_dist, k_by_mpid):
    portals = {}
    pbar = tqdm(total=len(orders_by_mpid), desc="Stage1: portals", unit="mp")
    for pid, ids in orders_by_mpid.items():
        k = int(k_by_mpid.get(pid, 10))
        if len(ids) <= k:
            portals[pid] = list(ids)
            pbar.update(1)
            continue
        portals[pid] = sorted(ids, key=lambda oid: fast_dist(int(oid), int(warehouse_id)))[:k]
        pbar.update(1)
    pbar.close()
    return portals


def _compute_centroids(orders_by_mpid, orders_by_id):
    centroids = {}
    pbar = tqdm(total=len(orders_by_mpid), desc="Stage1: centroids", unit="mp")
    for pid, ids in orders_by_mpid.items():
        if not ids:
            pbar.update(1)
            continue
        sx = 0.0
        sy = 0.0
        cnt = 0
        for oid in ids:
            o = orders_by_id.get(oid)
            if not o:
                continue
            lat = o.get("Lat")
            lon = o.get("Long")
            if lat is None or lon is None:
                continue
            sx += float(lat)
            sy += float(lon)
            cnt += 1
        if cnt == 0:
            pbar.update(1)
            continue
        centroids[pid] = (sx / cnt, sy / cnt)
        pbar.update(1)
    pbar.close()
    return centroids


def _neighbors_by_centroid(centroids, H_by_mpid):
    pids = list(centroids.keys())
    neigh = {pid: [] for pid in pids}
    pbar = tqdm(total=len(pids), desc="Stage1: neighbors (centroid kNN)", unit="mp")
    for i, a in enumerate(pids):
        ax, ay = centroids[a]
        cand = []
        for j, b in enumerate(pids):
            if a == b:
                continue
            bx, by = centroids[b]
            d2 = (ax - bx) * (ax - bx) + (ay - by) * (ay - by)
            cand.append((d2, b))
        cand.sort(key=lambda x: x[0])
        H = int(H_by_mpid.get(a, 20))
        neigh[a] = [b for _, b in cand[:H]]
        pbar.update(1)
    pbar.close()
    for a in pids:
        for b in list(neigh[a]):
            if a not in neigh.get(b, []):
                neigh.setdefault(b, []).append(a)
    return neigh

@lru_cache(maxsize=128)
def _edge_weight(A, B, portals, safe_dist):
    pa = portals.get(A) or []
    pb = portals.get(B) or []
    if not pa or not pb:
        return 10_000_000
    best_ab = None
    for p in pa:
        for q in pb:
            d = safe_dist(int(p), int(q))
            if best_ab is None or d < best_ab:
                best_ab = d
    best_ba = None
    for q in pb:
        for p in pa:
            d = safe_dist(int(q), int(p))
            if best_ba is None or d < best_ba:
                best_ba = d
    if best_ab is None and best_ba is None:
        return 10_000_000
    if best_ab is None:
        return int(best_ba)
    if best_ba is None:
        return int(best_ab)
    return int(min(best_ab, best_ba))


def _build_mpid_graph(portals, neighbors, safe_dist):
    adj = {pid: [] for pid in neighbors.keys()}
    pbar = tqdm(total=len(neighbors), desc="Stage1: build Mp graph", unit="mp")
    for a, lst in neighbors.items():
        for b in lst:
            w = _edge_weight(a, b, portals, safe_dist)
            adj[a].append((b, w))
        pbar.update(1)
    pbar.close()
    for a in list(adj.keys()):
        for b, _ in list(adj[a]):
            w_ab = min(w for x, w in adj[a] if x == b)
            w_ba = min((w for x, w in adj.get(b, []) if x == a), default=w_ab)
            w_sym = min(w_ab, w_ba)
            adj[a] = [(x, (w_sym if x == b else w)) for x, w in adj[a]]
            if b in adj:
                adj[b] = [(x, (w_sym if x == a else w)) for x, w in adj[b]]
    return adj


def _build_mpid_graph_sql(portals, neighbors, safe_dist, warehouse_id, d0_to, d_from0, hot_db_path):
    """
    Быстрое построение MpId-графа через батч-агрегации в hot.sqlite.
    Логика порталов/соседей/симметризации совпадает с _build_mpid_graph.
    """
    # 1) Предподготовка: min_to_wh / min_from_wh и пары соседей
    t0 = time.time()
    min_to_wh = {}
    min_from_wh = {}
    for mp, plist in portals.items():
        best_to = None
        best_from = None
        for pid in plist:
            # p -> WH
            v1 = d_from0.get(int(pid))
            if v1 is None:
                v1 = safe_dist(int(pid), int(warehouse_id))
            if best_to is None or (v1 is not None and v1 < best_to):
                best_to = v1
            # WH -> p
            v2 = d0_to.get(int(pid))
            if v2 is None:
                v2 = safe_dist(int(warehouse_id), int(pid))
            if best_from is None or (v2 is not None and v2 < best_from):
                best_from = v2
        if best_to is not None:
            min_to_wh[int(mp)] = int(best_to)
        if best_from is not None:
            min_from_wh[int(mp)] = int(best_from)

    pairs = set()
    for a, lst in neighbors.items():
        for b in lst:
            x, y = (int(a), int(b))
            if x == y:
                continue
            if x > y:
                x, y = y, x
            pairs.add((x, y))

    # 2) Загрузка во временные таблицы и SQL-агрегации
    t_load0 = time.time()
    con = sqlite3.connect(hot_db_path)
    cur = con.cursor()
    cur.execute("CREATE TEMP TABLE IF NOT EXISTS tmp_portals(mp INTEGER, pid INTEGER, PRIMARY KEY(mp,pid));")
    cur.execute("CREATE TEMP TABLE IF NOT EXISTS tmp_pairs(a INTEGER, b INTEGER, PRIMARY KEY(a,b));")
    cur.execute("CREATE TEMP TABLE IF NOT EXISTS tmp_min_wh_to(mp INTEGER PRIMARY KEY, val INTEGER);")
    cur.execute("CREATE TEMP TABLE IF NOT EXISTS tmp_min_wh_from(mp INTEGER PRIMARY KEY, val INTEGER);")
    con.commit()

    cur.execute("DELETE FROM tmp_portals;")
    cur.execute("DELETE FROM tmp_pairs;")
    cur.execute("DELETE FROM tmp_min_wh_to;")
    cur.execute("DELETE FROM tmp_min_wh_from;")
    con.commit()

    # portals
    data_portals = [(int(mp), int(pid)) for mp, lst in portals.items() for pid in lst]
    if data_portals:
        cur.executemany("INSERT OR IGNORE INTO tmp_portals(mp,pid) VALUES(?,?);", data_portals)
    # pairs
    data_pairs = [(a, b) for (a, b) in pairs]
    if data_pairs:
        cur.executemany("INSERT OR IGNORE INTO tmp_pairs(a,b) VALUES(?,?);", data_pairs)
    # wh mins
    if min_to_wh:
        cur.executemany("INSERT OR REPLACE INTO tmp_min_wh_to(mp,val) VALUES(?,?);", list(min_to_wh.items()))
    if min_from_wh:
        cur.executemany("INSERT OR REPLACE INTO tmp_min_wh_from(mp,val) VALUES(?,?);", list(min_from_wh.items()))
    con.commit()
    t_load1 = time.time()

    # агрегации
    t_sql0 = time.time()
    cur.execute("DROP TABLE IF EXISTS tmp_min_ab;")
    cur.execute("DROP TABLE IF EXISTS tmp_min_ba;")
    cur.execute(
        """
        CREATE TEMP TABLE tmp_min_ab AS
        SELECT tp.a, tp.b, MIN(dh.d) AS dmin
        FROM tmp_pairs tp
        JOIN tmp_portals pa ON pa.mp = tp.a
        JOIN tmp_portals pb ON pb.mp = tp.b
        JOIN dists_hot dh ON dh.f = pa.pid AND dh.t = pb.pid
        GROUP BY tp.a, tp.b;
        """
    )
    cur.execute(
        """
        CREATE TEMP TABLE tmp_min_ba AS
        SELECT tp.a, tp.b, MIN(dh.d) AS dmin
        FROM tmp_pairs tp
        JOIN tmp_portals pa ON pa.mp = tp.a
        JOIN tmp_portals pb ON pb.mp = tp.b
        JOIN dists_hot dh ON dh.f = pb.pid AND dh.t = pa.pid
        GROUP BY tp.a, tp.b;
        """
    )

    # финальная выборка
    cur.execute(
        """
        SELECT
          p.a, p.b,
          MIN(
            COALESCE(ab.dmin, 1000000000),
            COALESCE(ba.dmin, 1000000000),
            COALESCE(wh_to.val, 1000000000) + COALESCE(wh_from.val, 1000000000)
          ) AS w
        FROM tmp_pairs p
        LEFT JOIN tmp_min_ab  ab     ON ab.a = p.a AND ab.b = p.b
        LEFT JOIN tmp_min_ba  ba     ON ba.a = p.a AND ba.b = p.b
        LEFT JOIN tmp_min_wh_to   wh_to   ON wh_to.mp  = p.a
        LEFT JOIN tmp_min_wh_from wh_from ON wh_from.mp = p.b;
        """
    )
    rows = cur.fetchall()
    t_sql1 = time.time()

    # очистка temp таблиц
    cur.execute("DROP TABLE IF EXISTS tmp_min_ab;")
    cur.execute("DROP TABLE IF EXISTS tmp_min_ba;")
    cur.execute("DELETE FROM tmp_portals;")
    cur.execute("DELETE FROM tmp_pairs;")
    cur.execute("DELETE FROM tmp_min_wh_to;")
    cur.execute("DELETE FROM tmp_min_wh_from;")
    con.commit()
    con.close()

    # сборка adj и симметризация
    adj = {pid: [] for pid in neighbors.keys()}
    E = 0
    for a, b, w in rows:
        w = int(w)
        adj.setdefault(int(a), []).append((int(b), w))
        adj.setdefault(int(b), []).append((int(a), w))
        E += 2

    print(f"Stage1: build Mp graph — edges={E}, tmp_load={int(t_load1 - t_load0)}s, sql_agg={int(t_sql1 - t_sql0)}s")
    return adj


def _components(adj):
    nodes = list(adj.keys())
    seen = set()
    comps = []
    for s in nodes:
        if s in seen:
            continue
        stack = [s]
        comp = []
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v, _ in adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps


def _ensure_connected(adj, portals, safe_dist):
    # Добавляет недостающие рёбра между компонентами, выбирая min по _edge_weight
    comps = _components(adj)
    if len(comps) <= 1:
        return adj
    pbar = tqdm(total=len(comps) - 1, desc="Stage1: connect components", unit="link")
    while len(comps) > 1:
        best = None
        best_pair = None
        # Выбираем 2 компоненты и лучшее ребро между ними
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                Ci = comps[i]
                Cj = comps[j]
                for u in Ci:
                    for v in Cj:
                        w = _edge_weight(u, v, portals, safe_dist)
                        if best is None or w < best:
                            best = w
                            best_pair = (u, v)
        if best_pair is None:
            break
        u, v = best_pair
        # добавляем симметрично
        adj.setdefault(u, []).append((v, int(best)))
        adj.setdefault(v, []).append((u, int(best)))
        # пересчитать компоненты
        comps = _components(adj)
        pbar.update(1)
    pbar.close()
    return adj


def _make_inter_cost_fn_metric_on_mpid_graph(adj, portals, safe_dist, warehouse_id, lru_maxsize=200_000):
    all_pids = list(adj.keys())

    def iter_neighbors(node):
        if node == warehouse_id:
            for b in all_pids:
                pb = portals.get(b) or []
                if not pb:
                    continue
                best_fw = None
                for q in pb:
                    d = safe_dist(int(warehouse_id), int(q))
                    if best_fw is None or d < best_fw:
                        best_fw = d
                best_bw = None
                for q in pb:
                    d = safe_dist(int(q), int(warehouse_id))
                    if best_bw is None or d < best_bw:
                        best_bw = d
                if best_fw is None and best_bw is None:
                    continue
                w = min(x for x in [best_fw, best_bw] if x is not None)
                yield (b, int(w))
        else:
            for nb, w in adj.get(node, []):
                yield (nb, int(w))

    def _dijkstra_from_source(src):
        dist = {src: 0}
        h = [(0, src)]
        visited = set()
        while h:
            d, u = heapq.heappop(h)
            if u in visited:
                continue
            visited.add(u)
            for v, w in iter_neighbors(u):
                nd = d + int(w)
                if nd < dist.get(v, 1 << 60):
                    dist[v] = nd
                    heapq.heappush(h, (nd, v))
        return dist

    dist_cache = {}

    def get_inter_cost(a, b):
        if a == b:
            return 0
        D = dist_cache.get(a)
        if D is None:
            D = _dijkstra_from_source(a)
            dist_cache[a] = D
        return int(D.get(b, 10_000_000))

    return get_inter_cost


def _mst_over_adj(adj, start):
    """
    Строит MST по уже готовым рёбрам adj (MpId-граф) алгоритмом Крускала.
    Возвращает дерево как adjacency: tree[u] = [v1, v2, ...].
    """
    # Список вершин
    nodes = list(adj.keys())
    index = {v: i for i, v in enumerate(nodes)}

    # Собрать список рёбер (u<v) с минимальным весом между каждой парой
    edge_map = {}
    for u, lst in adj.items():
        for v, w in lst:
            a, b = (u, v) if u < v else (v, u)
            cur = edge_map.get((a, b))
            if cur is None or w < cur:
                edge_map[(a, b)] = int(w)
    edges = [(w, a, b) for (a, b), w in edge_map.items()]
    edges.sort(key=lambda x: x[0])

    # DSU (union-find)
    parent = list(range(len(nodes)))
    rank = [0] * len(nodes)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[ry] < rank[rx]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1
        return True

    # Набираем |V|-1 рёбер
    tree = {v: [] for v in nodes}
    need = max(0, len(nodes) - 1)
    added = 0
    pbar = tqdm(total=need, desc="Stage1: MST (Kruskal)", unit="edge")
    for w, a, b in edges:
        ia, ib = index[a], index[b]
        if union(ia, ib):
            tree[a].append(b)
            tree[b].append(a)
            added += 1
            pbar.update(1)
            if added >= need:
                break
    pbar.close()
    return tree


def _tour_from_tree(tree, root):
    order = []
    visited = set()
    def dfs(u):
        visited.add(u)
        order.append(u)
        for w in tree.get(u, []):
            if w not in visited:
                dfs(w)
    dfs(root)
    seen = set()
    tour = []
    for v in order:
        if v in seen:
            continue
        tour.append(v)
        seen.add(v)
    return tour


def _two_opt_improve(tour, get_cost, timecap_s=0.5):
    if len(tour) <= 3:
        return tour
    t0 = time.time()
    def seg_cost(a, b):
        return get_cost(a, b)
    improved = True
    while improved and (time.time() - t0) < timecap_s:
        improved = False
        for i in range(1, len(tour) - 2):
            a, b = tour[i - 1], tour[i]
            for j in range(i + 1, len(tour) - 1):
                c, d = tour[j], tour[j + 1]
                delta = (seg_cost(a, b) + seg_cost(c, d)) - (seg_cost(a, c) + seg_cost(b, d))
                if delta > 0:
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    improved = True
                    if (time.time() - t0) >= timecap_s:
                        break
            if (time.time() - t0) >= timecap_s:
                break
    return tour


def _solve_cluster_tsp_mstdouble(adj, portals, warehouse_id, get_inter_cost, safe_dist, two_opt_timecap_s=0.5):
    # Корень для DFS: MpId, ближайший к складу по порталам
    nodes = list(adj.keys())
    if not nodes:
        return [warehouse_id, warehouse_id]
    def dist_wh_to_pid(pid):
        ps = portals.get(pid) or []
        best = None
        for q in ps:
            d = safe_dist(int(warehouse_id), int(q))
            if best is None or d < best:
                best = d
        return best if best is not None else 10_000_000
    root = min(nodes, key=lambda pid: dist_wh_to_pid(pid))

    tree = _mst_over_adj(adj, root)
    order = _tour_from_tree(tree, root)
    # строим тур с началом в складе
    tour = [warehouse_id] + order + [warehouse_id]
    tour2 = _two_opt_improve(tour, get_inter_cost, timecap_s=two_opt_timecap_s)
    if tour2[0] != warehouse_id:
        try:
            idx = tour2.index(warehouse_id)
            tour2 = tour2[idx:] + tour2[1:idx + 1]
        except ValueError:
            tour2 = [warehouse_id] + [x for x in tour2 if x != warehouse_id] + [warehouse_id]
    if tour2[-1] != warehouse_id:
        tour2.append(warehouse_id)
    return tour2


def run(ctx):
    orders = ctx.get("orders", {})
    warehouse_id = ctx.get("warehouse_id", 0)

    fast_dist = ctx.get("fast_dist")
    safe_dist = ctx.get("safe_dist")
    if fast_dist is None or safe_dist is None:
        d0_to = ctx.get("d0_to", {})
        d_from0 = ctx.get("d_from0", {})
        cur = ctx.get("cur")
        hot_path = ctx.get("options", {}).get("hot_db_path", "hot.sqlite")
        try:
            hot_con = sqlite3.connect(hot_path)
            hot_cur = hot_con.cursor()
        except Exception:
            hot_cur = None

        @lru_cache(maxsize=200_000)
        def _direct(a, b):
            if a == b:
                return 0
            if hot_cur is not None:
                row = hot_cur.execute("SELECT d FROM dists_hot WHERE f=? AND t=?;", (a, b)).fetchone()
                if row:
                    return int(row[0])
            if cur is not None:
                row2 = cur.execute("SELECT d FROM dists WHERE f=? AND t=?;", (a, b)).fetchone()
                if row2:
                    return int(row2[0])
            return None

        @lru_cache(maxsize=200_000)
        def _safe(a, b):
            if a == b:
                return 0
            d = _direct(int(a), int(b))
            if d is not None:
                return d
            d1 = d_from0.get(int(a))
            d2 = d0_to.get(int(b))
            if d1 is not None and d2 is not None:
                return int(d1) + int(d2)
            return 10_000_000

        @lru_cache(maxsize=200_000)
        def _fast(a, b):
            if a == b:
                return 0
            d1 = d_from0.get(int(a))
            d2 = d0_to.get(int(b))
            if d1 is not None and d2 is not None:
                return int(d1) + int(d2)
            return 10_000_000

        fast_dist = _fast
        safe_dist = _safe

    orders_by_mpid = _group_orders_by_mpid(orders)
    closeness = _closeness_scores(orders_by_mpid, warehouse_id, fast_dist)
    k_by_mpid = _pick_k_policy(closeness, tau=0.7)
    portals = _select_portals(orders_by_mpid, warehouse_id, fast_dist, k_by_mpid)

    H0 = 20
    H_by_mpid = {pid: int(math.floor(H0 * (10.0 / max(1, int(k_by_mpid.get(pid, 10)))) ** 2)) for pid in orders_by_mpid.keys()}
    centroids = _compute_centroids(orders_by_mpid, orders)
    neighbors = _neighbors_by_centroid(centroids, H_by_mpid)

    # Быстрое построение через SQL поверх hot.sqlite
    hot_db_path = ctx.get("options", {}).get("hot_db_path", "hot.sqlite")
    d0_to = ctx.get("d0_to", {})
    d_from0 = ctx.get("d_from0", {})
    adj = _build_mpid_graph_sql(portals, neighbors, safe_dist, warehouse_id, d0_to, d_from0, hot_db_path)
    # (C) обеспечить связность графа MpId
    adj = _ensure_connected(adj, portals, safe_dist)

    get_inter_cost = _make_inter_cost_fn_metric_on_mpid_graph(adj, portals, safe_dist, warehouse_id, lru_maxsize=200_000)

    giant = _solve_cluster_tsp_mstdouble(adj, portals, warehouse_id, get_inter_cost, safe_dist, two_opt_timecap_s=0.5)

    try:
        print(f"Stage1: clusters={len(orders_by_mpid)}, portals_avg={int(sum(len(portals.get(mp, [])) for mp in orders_by_mpid)/max(1,len(orders_by_mpid)))}")
        print(f"Stage1: giant_tour_len={len(giant)} (incl WH), first={giant[:10] if isinstance(giant, list) else giant}")
    except Exception:
        pass

    return {
        "portals": portals,
        "k_by_mpid": k_by_mpid,
        "H_by_mpid": H_by_mpid,
        "get_inter_cost": get_inter_cost,
        "giant_tour": giant,
        "params": {"tau": 0.7, "H0": 20, "two_opt_timecap_s": 0.5},
    }
