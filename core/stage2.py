import math
import sqlite3
import time
from tqdm import tqdm
import numpy as np
from functools import lru_cache



def _normalize_orders(orders):
    if isinstance(orders, dict):
        return list(orders.values())
    return list(orders or [])


def _group_orders_by_mpid(orders_list):
    groups = {}
    for o in orders_list:
        pid = o.get("MpId")
        if pid is None:
            continue
        groups.setdefault(int(pid), []).append(int(o["ID"]))
    # стабильный порядок по ID
    for pid in groups:
        groups[pid].sort()
    return groups


def _make_intra_cost_fn(orders_by_mpid, sd, small_exact_limit=8, two_opt_iters=200):
    @lru_cache(maxsize=200_000)
    def get_intra_cost(mp, s, q):
        path = tuple(get_intra_path(mp, s, q))
        # стоимость по sd вдоль пути
        c = 0
        for a, b in zip(path[:-1], path[1:]):
            c += int(sd(int(a), int(b)))
        return int(c)

    @lru_cache(maxsize=200_000)
    def get_intra_path(mp, s, q):
        nodes = list(orders_by_mpid.get(int(mp), []))
        if not nodes:
            return (int(s), int(q)) if s != q else (int(s),)
        if int(s) not in nodes:
            nodes = [int(s)] + [x for x in nodes if x != int(s)]
        if int(q) not in nodes:
            nodes.append(int(q))
        s = int(s)
        q = int(q)
        nodes = list(dict.fromkeys(nodes))
        if len(nodes) <= 1:
            return (nodes[0],)
        if len(nodes) == 2:
            return (s, q) if s != q else (s,)
        if len(nodes) - 0 <= small_exact_limit:
            return tuple(_held_karp_path(nodes, s, q, sd))

        # быстрый базовый NN+2-opt
        base = _nn_two_opt_path(nodes, s, q, sd, two_opt_iters)

        # короткий ГА для «средних» кластеров (держим бюджет в десятках мс)
        if len(nodes) <= 40:
            ga_path = _ga_optimize_path(
                nodes, s, q, sd,
                base_path=base,
                time_budget_ms=25,   # можно поднять до 40–60 при наличии бюджета
                pop=12,
                iters=40,
                mut_rate=0.25,
                seed=0
            )
            return tuple(ga_path)

        return tuple(base)


   
    def _held_karp_path(nodes, s, q, edge):
        inners = [v for v in nodes if v not in (s, q)]
        pos = {v: i for i, v in enumerate(inners)}
        full_mask = (1 << len(inners)) - 1
        dp = {}
        parent = {}
        dp[(0, s)] = 0
        # переходы
        for mask in range(full_mask + 1):
            for u in [s] + [v for v in inners if (mask & (1 << pos[v]))]:
                key = (mask, u)
                if key not in dp:
                    continue
                base = dp[key]
                # добавляем следующий w из inners, которого нет в mask
                for w in inners:
                    bit = 1 << pos[w]
                    if mask & bit:
                        continue
                    nmask = mask | bit
                    nd = base + int(edge(int(u), int(w)))
                    k2 = (nmask, w)
                    if nd < dp.get(k2, 1 << 60):
                        dp[k2] = nd
                        parent[k2] = (mask, u)
        # завершение на q
        best = None
        last = None
        for u in [s] + inners:
            key = (full_mask, u)
            if key not in dp:
                continue
            cand = dp[key] + int(edge(int(u), int(q)))
            if best is None or cand < best:
                best = cand
                last = u
        # восстановление
        if last is None:
            # fallback: прямой путь s->...->q NN
            return _nn_two_opt_path(nodes, s, q, edge, 0)
        path_rev = [q]
        mask = full_mask
        u = last
        while True:
            path_rev.append(u)
            if mask == 0 and u == s:
                break
            prev = parent.get((mask, u))
            if prev is None:
                break
            mask, u = prev
        path_rev.reverse()
        return path_rev
    

    def _nn_two_opt_path(nodes, s, q, edge, two_opt_iters):
        others = [v for v in nodes if v not in (s, q)]
        seq = [s]
        cur = s
        remaining = set(others)
        while remaining:
            nxt = min(remaining, key=lambda x: edge(int(cur), int(x)))
            remaining.remove(nxt)
            seq.append(nxt)
            cur = nxt
        seq.append(q)
        # 2-opt с фиксированными концами

        def seg_cost(a, b):
            return int(edge(int(a), int(b)))
        
        it = 0
        improved = True
        while improved and it < two_opt_iters:
            improved = False
            it += 1
            for i in range(1, len(seq) - 2):
                a, b = seq[i - 1], seq[i]
                for k in range(i + 1, len(seq) - 1):
                    c, d = seq[k], seq[k + 1]
                    delta = (seg_cost(a, b) + seg_cost(c, d)) - (seg_cost(a, c) + seg_cost(b, d))
                    if delta > 0:
                        seq[i:k + 1] = reversed(seq[i:k + 1])
                        improved = True
                        break
                if improved:
                    break
        return seq
    
    def _ga_optimize_path(nodes, s, q, edge, base_path=None,
                          time_budget_ms=25, pop=12, iters=40, mut_rate=0.2, seed=0):
        """Короткий ГА по внутренним вершинам (s,q фиксированные). Возвращает путь [s, ... , q]."""
        import random, time as _t
        rnd = random.Random(seed)

        inner = [v for v in nodes if v not in (s, q)]
        n = len(inner)
        if n <= 2:
            return [s] + inner + [q]

        def cost_inner(order):
            c = int(edge(int(s), int(order[0])))
            for a, b in zip(order[:-1], order[1:]):
                c += int(edge(int(a), int(b)))
            c += int(edge(int(order[-1]), int(q)))
            return c

        # Инициализация популяции около base_path
        if base_path is None:
            base_path = _nn_two_opt_path(nodes, s, q, edge, 0)
        base_inner = [v for v in base_path if v not in (s, q)]

        population = [base_inner[:]]
        while len(population) < pop:
            cand = base_inner[:]
            if n >= 2:
                i, j = rnd.randrange(n), rnd.randrange(n)
                if i > j: i, j = j, i
                cand[i:j+1] = reversed(cand[i:j+1])
            population.append(cand)

        best = min(population, key=cost_inner)
        best_cost = cost_inner(best)

        def ox(p1, p2):
            """Order crossover для массива inner."""
            ln = len(p1)
            if ln < 2:
                return p1[:]
            i, j = rnd.randrange(ln), rnd.randrange(ln)
            if i > j: i, j = j, i
            child = [None] * ln
            child[i:j+1] = p1[i:j+1]
            fill = [x for x in p2 if x not in child]
            ptr = 0
            for k in range(ln):
                if child[k] is None:
                    child[k] = fill[ptr]
                    ptr += 1
            return child

        def mutate(order):
            ln = len(order)
            if ln >= 2 and rnd.random() < mut_rate:
                i, j = rnd.randrange(ln), rnd.randrange(ln)
                if i > j: i, j = j, i
                order[i:j+1] = reversed(order[i:j+1])

        t_end = _t.perf_counter() + time_budget_ms / 1000.0
        k = 0
        while _t.perf_counter() < t_end and k < iters:
            k += 1
            new_pop = []
            # турнир + кроссовер + мутация
            for _ in range(pop):
                a, b = rnd.choice(population), rnd.choice(population)
                mom = a if cost_inner(a) < cost_inner(b) else b
                a, b = rnd.choice(population), rnd.choice(population)
                dad = a if cost_inner(a) < cost_inner(b) else b
                child = ox(mom, dad)
                mutate(child)
                new_pop.append(child)
            population = new_pop
            for cand in population:
                c = cost_inner(cand)
                if c < best_cost:
                    best_cost = c
                    best = cand[:]

        return [s] + best + [q]


    return get_intra_cost, get_intra_path


# Глобальные таблицы предвычисленных внутрикластерных стоимостей
_intra_tbl_cost = {}
_intra_tbl_ready = set()


def _precompute_intra_table_for_mp(mp, portals, get_intra_cost):
    if mp in _intra_tbl_ready:
        return
    P = portals.get(mp, []) or []
    mp_tbl = {}
    for s in P:
        row = {}
        for q in P:
            row[int(q)] = int(get_intra_cost(int(mp), int(s), int(q)))
        mp_tbl[int(s)] = row
    _intra_tbl_cost[int(mp)] = mp_tbl
    _intra_tbl_ready.add(int(mp))


def _choose_entry_exit_along_tour(giant_tour, portals, sd, get_intra_cost, warehouse_id):
    if not giant_tour or len(giant_tour) <= 2:
        return {}, []
    seq = [mp for mp in giant_tour if mp != warehouse_id]
    if not seq:
        return {}, []
    # если в начале/конце остался склад, удалим
    if seq and seq[0] == warehouse_id:
        seq = seq[1:]
    if seq and seq[-1] == warehouse_id:
        seq = seq[:-1]
    if not seq:
        return {}, []

    back = {}
    # Глобальный прогресс по всем порталам всех кластеров
    total_portals = sum(len(portals.get(mp, [])) for mp in seq)
    pbar_dp = tqdm(total=total_portals, desc="Stage2: DP portals", unit="portal")
    # База для первого кластера
    C1 = seq[0]
    P1 = portals.get(C1, [])
    _precompute_intra_table_for_mp(int(C1), portals, get_intra_cost)
    F_prev = {}
    for q in P1:
        best = None
        best_s = None
        for s in P1:
            c_intra = _intra_tbl_cost[int(C1)][int(s)][int(q)] if int(C1) in _intra_tbl_cost else int(get_intra_cost(int(C1), int(s), int(q)))
            val = int(sd(int(warehouse_id), int(s))) + int(c_intra)
            if best is None or val < best:
                best = val
                best_s = s
        F_prev[q] = best if best is not None else 10_000_000
        back[(0, q)] = (None, best_s)
        pbar_dp.update(1)

    # Шаги для остальных кластеров
    for i in range(1, len(seq)):
        Ci = seq[i]
        Pi = portals.get(Ci, [])
        _precompute_intra_table_for_mp(int(Ci), portals, get_intra_cost)
        F_next = {}
        for q in Pi:
            best = None
            best_prev_p = None
            best_s = None
            for p in portals.get(seq[i - 1], []):
                # min_s ( time(p,s) + intra_i(s,q) )
                inner_best = None
                inner_s = None
                for s in Pi:
                    c_intra = _intra_tbl_cost[int(Ci)][int(s)][int(q)]
                    v = int(sd(int(p), int(s))) + int(c_intra)
                    if inner_best is None or v < inner_best:
                        inner_best = v
                        inner_s = s
                cand = F_prev.get(p, 10_000_000) + (inner_best if inner_best is not None else 10_000_000)
                if best is None or cand < best:
                    best = cand
                    best_prev_p = p
                    best_s = inner_s
            F_next[q] = best if best is not None else 10_000_000
            back[(i, q)] = (best_prev_p, best_s)
            pbar_dp.update(1)
        F_prev = F_next
    pbar_dp.close()

    # Завершение: + dist(q, W)
    last_idx = len(seq) - 1
    Pn = portals.get(seq[-1], [])
    best_total = None
    best_q = None
    for q in Pn:
        cand = F_prev.get(q, 10_000_000) + int(sd(int(q), int(warehouse_id)))
        if best_total is None or cand < best_total:
            best_total = cand
            best_q = q

    # Реконструкция
    entry_exit = {}
    q_cur = best_q
    for i in range(last_idx, -1, -1):
        p_prev, s_i = back.get((i, q_cur), (None, None))
        mp = seq[i]
        entry_exit[int(mp)] = {"entry": int(s_i) if s_i is not None else int(q_cur), "exit": int(q_cur)}
        q_cur = p_prev if p_prev is not None else q_cur

    return entry_exit, seq


def _expand_clusters_to_orders(entry_exit, get_intra_path):
    paths = {}
    pbar = tqdm(total=len(entry_exit), desc="Stage2: expand clusters", unit="mp")
    for mp, doors in entry_exit.items():
        s = int(doors["entry"])
        q = int(doors["exit"])
        path = list(get_intra_path(int(mp), s, q))
        paths[int(mp)] = path
        pbar.update(1)
    pbar.close()
    return paths


def run(ctx, distances=None):

    safe_dist = ctx.get("safe_dist")

    @lru_cache(maxsize=1_500_000)  # памяти хватает (32 ГБ), это критично ускоряет портальные DP
    def sd(a: int, b: int) -> int:
        return int(safe_dist(int(a), int(b)))
    
    orders_list = _normalize_orders(ctx.get("orders", []))
    orders_by_mpid = _group_orders_by_mpid(orders_list)
    warehouse_id = int(ctx.get("warehouse_id", 0))


    # взять giant_tour и portals; если нет — получим из Stage1
    giant_tour = ctx.get("giant_tour")
    portals = ctx.get("portals")
    if giant_tour is None or portals is None:
        try:
            from core.stage1 import run as stage1_run
            s1 = stage1_run(ctx)
            if giant_tour is None:
                giant_tour = s1.get("giant_tour")
            if portals is None:
                portals = s1.get("portals")
        except Exception:
            pass
    if giant_tour is None:
        giant_tour = [warehouse_id, warehouse_id]
    if portals is None:
        portals = {mp: orders_by_mpid.get(mp, [])[:1] for mp in orders_by_mpid.keys()}

    get_intra_cost, get_intra_path = _make_intra_cost_fn(orders_by_mpid, sd, small_exact_limit=12, two_opt_iters=200)
    entry_exit, seq = _choose_entry_exit_along_tour(giant_tour, portals, sd, get_intra_cost, warehouse_id)
    paths_by_mpid = _expand_clusters_to_orders(entry_exit, get_intra_path)

    # Fallback: если DP не смог выбрать двери (entry_exit пустой), зададим простой план —
    # брать первый портал как entry и последний (или тот же) как exit, и построим путь.
    if not entry_exit:
        seq2 = [mp for mp in giant_tour if mp != warehouse_id]
        for mp in seq2:
            P = portals.get(int(mp), []) or orders_by_mpid.get(int(mp), [])[:1]
            if not P:
                continue
            s = int(P[0])
            q = int(P[-1])
            entry_exit[int(mp)] = {"entry": s, "exit": q}
        # построим пути по кластерам для выбранных дверей
        paths_by_mpid = _expand_clusters_to_orders(entry_exit, get_intra_path)

    try:
        print(f"Stage2: seq_len={len(seq) if isinstance(seq,list) else 0}, entry_exit={len(entry_exit)}, paths={len(paths_by_mpid)}")
    except Exception:
        pass

    return {
        "entry_exit": entry_exit,
        "paths_by_mpid": paths_by_mpid,
        "giant_tour": giant_tour,
        "params": {"small_exact_limit": 12, "two_opt_iters": 200},
    }


def prefetch_after_stage2(ctx, entry_exit, paths_by_mpid, giant_tour):
    try:
        from core.stage0 import prefetch_along_tour as _prefetch_tour
    except Exception:
        _prefetch_tour = None

    # 1) Межкластерные стыки по giant_tour: prev_exit -> entry
    pairs = set()
    wh = int(ctx.get("warehouse_id", 0))
    seq = [mp for mp in giant_tour if mp != wh]
    if _prefetch_tour is not None and "portals" in ctx:
        try:
            _prefetch_tour(None, ctx.get("portals", {}), seq)
        except Exception:
            pass
    # точечные пары по выбранным дверям
    for a, b in zip(seq[:-1], seq[1:]):
        ex = entry_exit.get(int(a), {}).get("exit")
        en = entry_exit.get(int(b), {}).get("entry")
        if ex is not None and en is not None:
            pairs.add((int(ex), int(en)))

    # 2) Внутрикластерные рёбра путей
    for mp, path in paths_by_mpid.items():
        for u, v in zip(path[:-1], path[1:]):
            pairs.add((int(u), int(v)))

    # 3) Батч дозаливка в dists_hot через pairs-временку
    hot_db_path = ctx.get("options", {}).get("hot_db_path", "hot.sqlite")
    durations_db = ctx.get("options", {}).get("durations_db", None)
    # если нет путей к БД — ничего не делаем
    if not hot_db_path:
        return
    con_hot = sqlite3.connect(hot_db_path)
    cur_hot = con_hot.cursor()
    cur_hot.execute("CREATE TEMP TABLE IF NOT EXISTS pairs(f INTEGER, t INTEGER, PRIMARY KEY(f,t));")
    cur_hot.execute("DELETE FROM pairs;")
    if pairs:
        cur_hot.executemany("INSERT OR IGNORE INTO pairs(f,t) VALUES(?,?);", list(pairs))
    con_hot.commit()
    # вставляем найденные из dists_hot самого же файла (если уже есть) — noop, затем из main.dists
    # Здесь предполагается, что основная БД доступна как отдельный файл durations_db
    if durations_db:
        con_main = sqlite3.connect(durations_db)
        cur_main = con_main.cursor()
        # выбираем существующие
        rows = []
        for f, t in pairs:
            row = cur_main.execute("SELECT d FROM dists WHERE f=? AND t=?;", (f, t)).fetchone()
            if row:
                rows.append((int(f), int(t), int(row[0])))
        if rows:
            cur_hot.execute("BEGIN;")
            cur_hot.executemany("INSERT OR REPLACE INTO dists_hot(f,t,d) VALUES(?,?,?);", rows)
            con_hot.commit()
        con_main.close()

    cur_hot.execute("DROP TABLE IF EXISTS pairs;")
    con_hot.commit()
    con_hot.close()


