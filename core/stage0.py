import sqlite3
import random
from functools import lru_cache
from tqdm import tqdm

_MAIN_CUR = None  # будет установлен при run(ctx)


def prefetch_along_tour(hot_con, portals, tour):
    """
    Опциональный префетч: для дуг тура (A,B) подтягивает пары портал×портал (в обе стороны)
    из основной БД в dists_hot. Использует _MAIN_CUR, установленный в run(ctx).
    """
    if _MAIN_CUR is None or hot_con is None or not tour:
        return
    hot_cur = hot_con.cursor()
    to_insert = []
    for a, b in zip(tour[:-1], tour[1:]):
        pa = portals.get(a) or []
        pb = portals.get(b) or []
        for x in pa:
            for y in pb:
                row1 = _MAIN_CUR.execute("SELECT d FROM dists WHERE f=? AND t=?;", (x, y)).fetchone()
                if row1:
                    to_insert.append((int(x), int(y), int(row1[0])))
                row2 = _MAIN_CUR.execute("SELECT d FROM dists WHERE f=? AND t=?;", (y, x)).fetchone()
                if row2:
                    to_insert.append((int(y), int(x), int(row2[0])))
    if to_insert:
        hot_cur.execute("BEGIN;")
        hot_cur.executemany("INSERT OR REPLACE INTO dists_hot(f,t,d) VALUES(?,?,?);", to_insert)
        hot_con.commit()


def run(ctx):
    """
    Stage 0 (A+C): горячий кэш рёбер в локальной БД + LRU-обёртки.
    Возвращает словарь с тремя функциями: direct, safe_dist, fast_dist.
    Контракт неизменен.
    """
    global _MAIN_CUR

    # Входные объекты и опции
    cur = ctx["cur"]
    _MAIN_CUR = cur
    orders = ctx.get("orders", {})
    d0_to = ctx.get("d0_to", {})
    d_from0 = ctx.get("d_from0", {})
    polygons = ctx.get("polygons", {})
    warehouse_id = ctx.get("warehouse_id", 0)

    options = ctx.get("options", {})
    hot_db_path = options.get("hot_db_path", "hot.sqlite")
    k_portals = int(options.get("k_portals", 4))
    H_neighbors = int(options.get("H_neighbors", 20))
    R_far = int(options.get("R_far", 3))
    K_intra = int(options.get("K_intra", 10))
    LRU_MAXSIZE = int(options.get("LRU_MAXSIZE", 500_000))
    RANDOM_SEED = int(options.get("RANDOM_SEED", 42))

    rnd = random.Random(RANDOM_SEED)

    # Инициализация hot-DB
    hot_con = sqlite3.connect(hot_db_path)
    hot_con.execute("PRAGMA journal_mode=OFF;")
    hot_con.execute("PRAGMA synchronous=OFF;")
    hot_con.execute("PRAGMA temp_store=MEMORY;")
    hot_con.execute("PRAGMA cache_size=-131072;")
    hot_cur = hot_con.cursor()
    hot_cur.execute("DROP TABLE IF EXISTS dists_hot;")
    hot_cur.execute("CREATE TABLE dists_hot (f INTEGER, t INTEGER, d INTEGER, PRIMARY KEY(f,t));")
    hot_con.commit()

    # A) Собираем wishlist пар (u,v)
    wishlist = set()

    # 1) склад ↔ заказ (оба направления)
    pbar_wh = tqdm(total=len(orders), desc="Stage0: WH<->orders", unit="order")
    for oid in orders.keys():
        wishlist.add((int(warehouse_id), int(oid)))
        wishlist.add((int(oid), int(warehouse_id)))
        pbar_wh.update(1)
    pbar_wh.close()

    # Вспомогательные: порталы по полигону (ближайшие к складу по d0_to)
    def select_portals(nodes_list):
        if not nodes_list:
            return []
        if len(nodes_list) <= k_portals:
            return list(nodes_list)
        return sorted(nodes_list, key=lambda oid: d0_to.get(int(oid), 10_000_000))[:k_portals]

    polygon_to_portals = {pid: select_portals(nodes) for pid, nodes in polygons.items()}

    # 2) Intra-MpId: для каждого заказа K_intra ближайших соседей по порядку d0_to (скользящее окно)
    pbar_intra = tqdm(total=len(polygons), desc="Stage0: intra-Mp kNN", unit="mp")
    for pid, nodes in polygons.items():
        if not nodes:
            continue
        arr = sorted(nodes, key=lambda oid: d0_to.get(int(oid), 10_000_000))
        n = len(arr)
        for i, u in enumerate(arr):
            # соседи в окне [i-K_intra, i+K_intra]
            lo = max(0, i - K_intra)
            hi = min(n, i + K_intra + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                v = arr[j]
                wishlist.add((int(u), int(v)))
                wishlist.add((int(v), int(u)))
        pbar_intra.update(1)
    pbar_intra.close()

    # 3) Межполигонные соседства: H ближайших + R дальних случайных по порталам (fast-оценка)
    poly_ids = list(polygons.keys())
    def poly_cost_fast(a, b):
        pa = polygon_to_portals.get(a) or []
        pb = polygon_to_portals.get(b) or []
        if not pa or not pb:
            return 10_000_000
        best = None
        for x in pa:
            dx = d_from0.get(int(x))
            if dx is None:
                continue
            for y in pb:
                dy = d0_to.get(int(y))
                if dy is None:
                    continue
                val = dx + dy
                if best is None or val < best:
                    best = val
        return best if best is not None else 10_000_000

    pbar_inter = tqdm(total=len(poly_ids), desc="Stage0: inter-Mp portals", unit="mp")
    for a in poly_ids:
        # Сортировка соседей по fast-стоимости
        neigh = sorted((b for b in poly_ids if b != a), key=lambda b: poly_cost_fast(a, b))
        close = neigh[:H_neighbors]
        far_pool = neigh[H_neighbors:]
        far = rnd.sample(far_pool, min(R_far, len(far_pool))) if far_pool else []
        cand = close + far
        pa = polygon_to_portals.get(a) or []
        for b in cand:
            pb = polygon_to_portals.get(b) or []
            for x in pa:
                for y in pb:
                    wishlist.add((int(x), int(y)))
                    wishlist.add((int(y), int(x)))
        pbar_inter.update(1)
    pbar_inter.close()

    # B) Наполняем dists_hot одним батчем запросов к основной БД (эмулируем join)
    #    Из-за query_only на главном соединении используем последовательные SELECT и батч-вставку
    batch = []
    hot_cur.execute("BEGIN;")
    wl_list = list(wishlist)
    pbar_join = tqdm(total=len(wl_list), desc="Stage0: fill hot.sqlite", unit="edge")
    for (f, t) in wl_list:
        row = cur.execute("SELECT d FROM dists WHERE f=? AND t=?;", (f, t)).fetchone()
        if row:
            batch.append((f, t, int(row[0])))
            if len(batch) >= 10_000:
                hot_cur.executemany("INSERT OR REPLACE INTO dists_hot(f,t,d) VALUES(?,?,?);", batch)
                hot_con.commit()
                hot_cur.execute("BEGIN;")
                batch.clear()
        pbar_join.update(1)
    if batch:
        hot_cur.executemany("INSERT OR REPLACE INTO dists_hot(f,t,d) VALUES(?,?,?);", batch)
        hot_con.commit()
    pbar_join.close()

    # C) Определяем direct/safe_dist/fast_dist с LRU
    def make_direct(maxsize):
        @lru_cache(maxsize=maxsize)
        def _direct(a, b):
            if a == b:
                return 0
            # 1) пробуем hot
            row = hot_cur.execute("SELECT d FROM dists_hot WHERE f=? AND t=?;", (a, b)).fetchone()
            if row:
                return int(row[0])
            # 2) пробуем основную БД (строгое направление)
            row2 = cur.execute("SELECT d FROM dists WHERE f=? AND t=?;", (a, b)).fetchone()
            if row2:
                val = int(row2[0])
                # запись в hot для будущих обращений
                hot_cur.execute("BEGIN;")
                hot_cur.execute("INSERT OR REPLACE INTO dists_hot(f,t,d) VALUES(?,?,?);", (int(a), int(b), val))
                hot_con.commit()
                return val
            return None
        return _direct

    direct = make_direct(LRU_MAXSIZE)

    def safe_dist(a, b):
        if a == b:
            return 0
        d = direct(a, b)
        if d is not None:
            return d
        # через склад: сначала из словарей, затем из hot (без обращения к main)
        d1 = d_from0.get(int(a))
        if d1 is None:
            row_a = hot_cur.execute("SELECT d FROM dists_hot WHERE f=? AND t=?;", (int(a), int(warehouse_id))).fetchone()
            d1 = int(row_a[0]) if row_a else None
        d2 = d0_to.get(int(b))
        if d2 is None:
            row_b = hot_cur.execute("SELECT d FROM dists_hot WHERE f=? AND t=?;", (int(warehouse_id), int(b))).fetchone()
            d2 = int(row_b[0]) if row_b else None
        if d1 is not None and d2 is not None:
            return d1 + d2
        return 10_000_000

    def fast_dist(a, b):
        if a == b:
            return 0
        d1 = d_from0.get(int(a))
        d2 = d0_to.get(int(b))
        if d1 is not None and d2 is not None:
            return d1 + d2
        return 10_000_000

    return {
        "direct": direct,
        "safe_dist": safe_dist,
        "fast_dist": fast_dist,
    }


