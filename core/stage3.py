import time
from tqdm import tqdm

LIMIT = 12 * 3600
BIG = 10 ** 9


def _sequence_of_clusters(giant_tour, warehouse_id):
    if not giant_tour:
        return []
    return [mp for mp in giant_tour if mp != warehouse_id]


def _cluster_internal_move_time(path, safe_dist):
    if not path or len(path) <= 1:
        return 0
    t = 0
    for a, b in zip(path[:-1], path[1:]):
        t += int(safe_dist(int(a), int(b)))
    return t


def _courier_service_map(couriers):
    svc = {}
    for c in couriers:
        cid = int(c.get("ID")) if isinstance(c, dict) else int(c)
        m = {}
        if isinstance(c, dict):
            for s in c.get("ServiceTimeInMps", []) or []:
                mp = int(s.get("MpID"))
                st = s.get("ServiceTime", 300)
                try:
                    st = int(st)
                except Exception:
                    st = 300
                if st < 0:
                    st = 300
                m[mp] = st
        svc[cid] = m
    return svc


def _avg_service_by_mpid(couriers, orders_by_mpid):
    svc_map = _courier_service_map(couriers)
    mp_ids = list(orders_by_mpid.keys())
    avg_by_mp = {}
    if not svc_map:
        for mp in mp_ids:
            avg_by_mp[mp] = 300
        return avg_by_mp
    for mp in mp_ids:
        vals = []
        for cid, m in svc_map.items():
            vals.append(int(m.get(mp, 300)))
        avg_by_mp[mp] = int(sum(vals) / max(1, len(vals)))
    return avg_by_mp


def _alpha_beta(i, j, C_order, entry_exit, safe_dist, warehouse_id):
    mp_i = C_order[i - 1]
    mp_j = C_order[j - 1]
    s = entry_exit.get(int(mp_i), {}).get("entry")
    e = entry_exit.get(int(mp_j), {}).get("exit")
    a = int(safe_dist(int(warehouse_id), int(s))) if s is not None else BIG
    b = int(safe_dist(int(e), int(warehouse_id))) if e is not None else BIG
    return a, b


def _build_prefixes(C_order, L, S_cluster, B, entry_exit, safe_dist, warehouse_id):
    n = len(C_order)
    PL = [0] * (n + 1)
    PS = [0] * (n + 1)
    PB = [0] * (n + 1)
    for j in range(1, n + 1):
        mp = C_order[j - 1]
        PL[j] = PL[j - 1] + int(L[mp])
        PS[j] = PS[j - 1] + int(S_cluster[mp])
        if j - 1 >= 1:
            PB[j] = PB[j - 1] + int(B[j - 1])
        else:
            PB[j] = 0

    def cost_avg(i, j):
        a, b = _alpha_beta(i, j, C_order, entry_exit, safe_dist, warehouse_id)
        return int(a + (PL[j] - PL[i - 1]) + (PS[j] - PS[i - 1]) + (PB[j] - PB[i]) + b)

    return cost_avg


def _split_dp_basic(n, cost_avg, limit=LIMIT):
    dp = [(BIG, -1)] * (n + 1)
    dp[0] = (0, -1)
    pbar = tqdm(total=n, desc="Stage3: Split-DP", unit="i")
    for i in range(0, n):
        base = dp[i][0]
        if base >= BIG:
            pbar.update(1)
            continue
        j = i + 1
        while j <= n:
            c = cost_avg(i + 1, j)
            if c > limit:
                break
            new_cost = base + c
            if new_cost < dp[j][0]:
                dp[j] = (new_cost, i)
            j += 1
        pbar.update(1)
    pbar.close()
    segs = []
    cur = n
    while cur > 0:
        prev = dp[cur][1]
        if prev < 0:
            break
        segs.append((prev + 1, cur))
        cur = prev
    segs.reverse()
    return segs


def _split_dp_with_cap(n, cost_avg, max_segments, limit=LIMIT):
    dp = [[BIG] * (max_segments + 1) for _ in range(n + 1)]
    back = [[-1] * (max_segments + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    pbar = tqdm(total=max_segments * max(1, n), desc="Stage3: Split-DP (cap)", unit="step")
    for m in range(1, max_segments + 1):
        for j in range(1, n + 1):
            i = j - 1
            while i >= 0:
                c = cost_avg(i + 1, j)
                if c <= limit:
                    val = dp[i][m - 1] + c
                    if val < dp[j][m]:
                        dp[j][m] = val
                        back[j][m] = i
                i -= 1
            pbar.update(1)
    pbar.close()
    best_m = None
    best_val = BIG
    for m in range(1, max_segments + 1):
        if dp[n][m] < best_val:
            best_val = dp[n][m]
            best_m = m
    if best_m is None:
        return []
    segs = []
    cur = n
    m = best_m
    while cur > 0 and m > 0:
        i = back[cur][m]
        if i < 0:
            break
        segs.append((i + 1, cur))
        cur = i
        m -= 1
    segs.reverse()
    return segs


def _route_time_for_courier(segment, C_order, L, B, entry_exit, paths_by_mpid, service_map, courier_id, safe_dist, warehouse_id):
    i, j = segment
    a, b = _alpha_beta(i, j, C_order, entry_exit, safe_dist, warehouse_id)
    Lsum = 0
    Ssum = 0
    for k in range(i, j + 1):
        mp = C_order[k - 1]
        Lsum += int(L[mp])
        st = int(service_map.get(courier_id, {}).get(int(mp), 300))
        cnt = len(paths_by_mpid.get(int(mp), []) or [])
        Ssum += st * cnt
    Bsum = 0
    for k in range(i, j):
        mp_prev = C_order[k - 1]
        mp_next = C_order[k]
        ex = entry_exit.get(int(mp_prev), {}).get("exit")
        en = entry_exit.get(int(mp_next), {}).get("entry")
        if ex is None or en is None:
            return BIG
        Bsum += int(safe_dist(int(ex), int(en)))
    total = a + Lsum + Ssum + Bsum + b
    return int(total)


def _build_cost_matrix(segments, couriers, C_order, L, B, entry_exit, paths_by_mpid, service_map, safe_dist, warehouse_id):
    K = len(segments)
    C = len(couriers)
    cost = [[BIG] * C for _ in range(K)]
    pbar = tqdm(total=K, desc="Stage3: Build cost matrix", unit="seg")
    for r, seg in enumerate(segments):
        for cidx, c in enumerate(couriers):
            cid = int(c)
            t = _route_time_for_courier(seg, C_order, L, B, entry_exit, paths_by_mpid, service_map, cid, safe_dist, warehouse_id)
            cost[r][cidx] = t if t <= LIMIT else BIG
        pbar.update(1)
    pbar.close()
    return cost


# === INSERT: жадное слияние, если сегментов больше, чем курьеров ===
def _greedy_merge_segments(segments, cost_avg, time_limit, max_segments):
    """
    Сливает соседние отрезки [i..j] + [j+1..k] в [i..k], если merged <= time_limit.
    Каждый шаг выбирает пару с минимальным ростом стоимости. Останавливается, когда len<=max_segments
    или больше слить нельзя.
    """
    if max_segments is None or len(segments) <= max_segments:
        return segments
    segs = list(segments)
    cost_cache = {}

    def seg_cost(a, b):
        key = (a, b)
        if key not in cost_cache:
            cost_cache[key] = cost_avg(a, b)
        return cost_cache[key]

    while len(segs) > max_segments:
        best_k = None
        best_increase = None
        for k in range(len(segs) - 1):
            i1, j1 = segs[k]
            i2, j2 = segs[k + 1]
            if j1 + 1 != i2:
                continue
            c1 = seg_cost(i1, j1)
            c2 = seg_cost(i2, j2)
            c12 = seg_cost(i1, j2)
            if c12 <= time_limit:
                inc = c12 - (c1 + c2)
                if (best_increase is None) or (inc < best_increase):
                    best_increase = inc
                    best_k = k
        if best_k is None:
            break
        i1, _ = segs[best_k]
        _, j2 = segs[best_k + 1]
        segs[best_k] = (i1, j2)
        del segs[best_k + 1]
    return segs


def _hungarian(cost):
    n_rows = len(cost)
    n_cols = len(cost[0]) if cost else 0
    n = max(n_rows, n_cols)
    cost2 = [row + [0] * (n - n_cols) for row in cost]
    for _ in range(n - n_rows):
        cost2.append([0] * n)

    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [BIG] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = BIG
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost2[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    ans = [-1] * n_rows
    for j in range(1, n_cols + 1):
        if p[j] <= n_rows:
            ans[p[j] - 1] = j - 1
    return [(i, ans[i]) for i in range(n_rows) if ans[i] != -1]


def _build_routes(segments, assignment, C_order, paths_by_mpid, warehouse_id):
    routes = []
    pbar = tqdm(total=len(assignment), desc="Stage3: Build routes", unit="route")
    for (r, cidx, cid, t) in assignment:
        i, j = segments[r]
        seq = [warehouse_id]
        for k in range(i, j + 1):
            mp = C_order[k - 1]
            seq.extend(paths_by_mpid.get(int(mp), []))
        seq.append(warehouse_id)
        routes.append({"courier_id": cid, "route": seq})
        pbar.update(1)
    pbar.close()
    return routes


# ================= Greedy repair utilities =================
TIME_LIMIT = LIMIT


def _unassigned_or_overlimit_segments(segments, assignment_dicts, cost_avg, couriers, C_order):
    assigned_map = {a["segment_id"]: int(a.get("time_sec", BIG)) for a in assignment_dicts}
    problem_ids = []
    for seg_id in range(len(segments)):
        t = assigned_map.get(seg_id)
        if t is None or t > TIME_LIMIT:
            i, j = segments[seg_id]
            problem_ids.append((seg_id, cost_avg(i, j)))
    # sort by neutral duration desc
    problem_ids.sort(key=lambda x: -x[1])
    problem_ids = [seg_id for seg_id, _ in problem_ids]
    used_couriers = set(int(a["courier_id"]) for a in assignment_dicts if a.get("courier_id") is not None)
    return problem_ids, used_couriers


def _best_free_for_segment(segment, free_couriers, service_map, safe_dist, entry_exit, paths_by_mpid, C_order, L, warehouse_id):
    i, j = segment
    best = None
    best_c = None
    for cid in list(free_couriers):
        t = _route_time_for_courier((i, j), C_order, L, {}, entry_exit, paths_by_mpid, service_map, int(cid), safe_dist, warehouse_id)
        # Для скорости внутренняя длина/стыки уже учтутся выше при пустых L/B? Лучше посчитать корректно:
        # Пересчёт корректный использует L/B из вне; здесь fallback: если нет, считаем напрямую
        if t > TIME_LIMIT:
            continue
        if best is None or t < best:
            best = t
            best_c = int(cid)
    if best_c is None:
        return None
    return best_c, int(best)


def _try_split_two_way(segment, free_couriers, service_map, safe_dist, entry_exit, paths_by_mpid, C_order, L, warehouse_id, max_scan=None):
    i, j = segment
    points = list(range(i, j))
    if max_scan is not None and max_scan > 0 and len(points) > max_scan:
        # равномерно выберем max_scan точек
        step = max(1, len(points) // max_scan)
        points = points[::step][:max_scan]
    best_plan = None
    best_sum = None
    free = list(dict.fromkeys(int(c) for c in free_couriers))
    for m in points:
        r1 = (i, m)
        r2 = (m + 1, j)
        # найдём лучших двух разных курьеров
        cand1 = []
        for cid in free:
            t1 = _route_time_for_courier(r1, C_order, L, {}, entry_exit, paths_by_mpid, service_map, int(cid), safe_dist, warehouse_id)
            if t1 <= TIME_LIMIT:
                cand1.append((t1, int(cid)))
        if not cand1:
            continue
        cand1.sort()
        for t1, c1 in cand1[:5]:
            best_t2 = None
            best_c2 = None
            for cid2 in free:
                if int(cid2) == c1:
                    continue
                t2 = _route_time_for_courier(r2, C_order, L, {}, entry_exit, paths_by_mpid, service_map, int(cid2), safe_dist, warehouse_id)
                if t2 <= TIME_LIMIT and (best_t2 is None or t2 < best_t2):
                    best_t2 = t2
                    best_c2 = int(cid2)
            if best_t2 is None:
                continue
            s = int(t1) + int(best_t2)
            if best_sum is None or s < best_sum:
                best_sum = s
                best_plan = {"split": [(i, m, c1, int(t1)), (m + 1, j, best_c2, int(best_t2))]}
    return best_plan


def _try_boundary_shift(problem_seg, neighbor_seg, neighbor_courier, free_couriers, direction,
                        service_map, safe_dist, entry_exit, paths_by_mpid, C_order, L, warehouse_id, max_shift=3):
    ip, jp = problem_seg
    inb, jnb = neighbor_seg
    free = list(dict.fromkeys(int(c) for c in free_couriers))
    if direction == "left":
        # neighbor on left; try moving s from neighbor right end to problem left
        best = None
        for s in range(1, max_shift + 1):
            if jnb - s < inb or ip - s < 1 or jnb - s != ip - 1:
                continue
            new_nb = (inb, jnb - s)
            new_pb = (ip - s, jp)
            t_nb = _route_time_for_courier(new_nb, C_order, L, {}, entry_exit, paths_by_mpid, service_map, int(neighbor_courier), safe_dist, warehouse_id)
            if t_nb > TIME_LIMIT:
                continue
            bf = _best_free_for_segment(new_pb, free, service_map, safe_dist, entry_exit, paths_by_mpid, C_order, L, warehouse_id)
            if not bf:
                continue
            c_id, t_pb = bf
            total = int(t_nb) + int(t_pb)
            if best is None or total < best[0]:
                best = (total, new_nb, new_pb, c_id, int(t_nb), int(t_pb))
        if best:
            _, new_nb, new_pb, c_id, t_nb, t_pb = best
            return {"left": {"neighbor": new_nb, "problem": new_pb, "neighbor_time": t_nb, "problem_courier": c_id, "problem_time": t_pb}}
    else:
        # direction == right; neighbor on right; move s from neighbor left to problem right
        best = None
        for s in range(1, max_shift + 1):
            if inb + s > jnb or jp + s > len(C_order) or inb + s != jp + 1:
                continue
            new_nb = (inb + s, jnb)
            new_pb = (ip, jp + s)
            t_nb = _route_time_for_courier(new_nb, C_order, L, {}, entry_exit, paths_by_mpid, service_map, int(neighbor_courier), safe_dist, warehouse_id)
            if t_nb > TIME_LIMIT:
                continue
            bf = _best_free_for_segment(new_pb, free, service_map, safe_dist, entry_exit, paths_by_mpid, C_order, L, warehouse_id)
            if not bf:
                continue
            c_id, t_pb = bf
            total = int(t_nb) + int(t_pb)
            if best is None or total < best[0]:
                best = (total, new_nb, new_pb, c_id, int(t_nb), int(t_pb))
        if best:
            _, new_nb, new_pb, c_id, t_nb, t_pb = best
            return {"right": {"neighbor": new_nb, "problem": new_pb, "neighbor_time": t_nb, "problem_courier": c_id, "problem_time": t_pb}}
    return None


def _apply_assignment_update(assignment_dicts, seg_id, courier_id, time_sec):
    found = False
    for a in assignment_dicts:
        if a["segment_id"] == seg_id:
            a["courier_id"] = int(courier_id)
            a["time_sec"] = int(time_sec)
            found = True
            break
    if not found:
        assignment_dicts.append({"segment_id": seg_id, "courier_id": int(courier_id), "time_sec": int(time_sec)})


def _greedy_repair_unassigned(segments, assignment_dicts, couriers_list, service_map,
                              safe_dist, entry_exit, paths_by_mpid, C_order, L, cost_avg, warehouse_id):
    problem_ids, used = _unassigned_or_overlimit_segments(segments, assignment_dicts, cost_avg, couriers_list, C_order)
    free_set = set(int(c.get("ID")) for c in couriers_list if int(c.get("ID")) not in used)

    repaired = 0
    split_cnt = 0
    shift_cnt = 0
    pbar = tqdm(total=len(problem_ids), desc="Stage3: Greedy repair", unit="seg")

    # 1) целиком на свободного
    for seg_id in list(problem_ids):
        seg = segments[seg_id]
        best = _best_free_for_segment(seg, free_set, service_map, safe_dist, entry_exit, paths_by_mpid, C_order, L, warehouse_id)
        if best:
            c_id, t_sec = best
            _apply_assignment_update(assignment_dicts, seg_id, c_id, t_sec)
            if c_id in free_set:
                free_set.remove(c_id)
            problem_ids.remove(seg_id)
            repaired += 1
            pbar.update(1)

    # 2) разрез на два
    for seg_id in list(problem_ids):
        if len(free_set) < 2:
            break
        seg = segments[seg_id]
        plan = _try_split_two_way(seg, list(free_set), service_map, safe_dist, entry_exit, paths_by_mpid, C_order, L, warehouse_id, max_scan=30)
        if not plan:
            continue
        (i1, j1, c1, t1), (i2, j2, c2, t2) = plan["split"]
        # заменить сегмент и вставить второй, поддерживая порядок
        segments[seg_id] = (i1, j1)
        insert_pos = seg_id + 1
        segments.insert(insert_pos, (i2, j2))
        # обновить назначения
        _apply_assignment_update(assignment_dicts, seg_id, c1, t1)
        _apply_assignment_update(assignment_dicts, insert_pos, c2, t2)
        if c1 in free_set:
            free_set.remove(c1)
        if c2 in free_set:
            free_set.remove(c2)
        # пересдвинуть segment_id у последующих назначений (сдвинуты вправо)
        for a in assignment_dicts:
            if a["segment_id"] >= insert_pos and (a["courier_id"] not in (c1, c2) or a["segment_id"] != insert_pos):
                a["segment_id"] += 1
        problem_ids.remove(seg_id)
        split_cnt += 1
        pbar.update(1)

    # 3) сдвиг границы с соседями
    for seg_id in list(problem_ids):
        left_id = seg_id - 1 if seg_id - 1 >= 0 else None
        right_id = seg_id + 1 if seg_id + 1 < len(segments) else None
        ok = False
        if left_id is not None:
            left_assign = next((a for a in assignment_dicts if a["segment_id"] == left_id), None)
            if left_assign:
                plan = _try_boundary_shift(segments[seg_id], segments[left_id], left_assign["courier_id"], list(free_set), "left",
                                           service_map, safe_dist, entry_exit, paths_by_mpid, C_order, L, warehouse_id)
                if plan and "left" in plan:
                    new_nb = plan["left"]["neighbor"]
                    new_pb = plan["left"]["problem"]
                    c_id = plan["left"]["problem_courier"]
                    t_nb = plan["left"]["neighbor_time"]
                    t_pb = plan["left"]["problem_time"]
                    segments[left_id] = new_nb
                    segments[seg_id] = new_pb
                    _apply_assignment_update(assignment_dicts, left_id, left_assign["courier_id"], t_nb)
                    _apply_assignment_update(assignment_dicts, seg_id, c_id, t_pb)
                    if c_id in free_set:
                        free_set.remove(c_id)
                    problem_ids.remove(seg_id)
                    ok = True
                    shift_cnt += 1
                    pbar.update(1)
        if (not ok) and (right_id is not None):
            right_assign = next((a for a in assignment_dicts if a["segment_id"] == right_id), None)
            if right_assign:
                plan = _try_boundary_shift(segments[seg_id], segments[right_id], right_assign["courier_id"], list(free_set), "right",
                                           service_map, safe_dist, entry_exit, paths_by_mpid, C_order, L, warehouse_id)
                if plan and "right" in plan:
                    new_nb = plan["right"]["neighbor"]
                    new_pb = plan["right"]["problem"]
                    c_id = plan["right"]["problem_courier"]
                    t_nb = plan["right"]["neighbor_time"]
                    t_pb = plan["right"]["problem_time"]
                    segments[right_id] = new_nb
                    segments[seg_id] = new_pb
                    _apply_assignment_update(assignment_dicts, right_id, right_assign["courier_id"], t_nb)
                    _apply_assignment_update(assignment_dicts, seg_id, c_id, t_pb)
                    if c_id in free_set:
                        free_set.remove(c_id)
                    problem_ids.remove(seg_id)
                    shift_cnt += 1
                    pbar.update(1)

    print(f"Stage3: greedy repair — repaired={repaired}, split={split_cnt}, shift={shift_cnt}, remaining_problems={len(problem_ids)}")
    pbar.close()
    return segments, assignment_dicts


def run(ctx):
    orders = ctx.get("orders", {})
    orders_list = list(orders.values()) if isinstance(orders, dict) else (orders or [])
    orders_by_mpid = {}
    for o in orders_list:
        orders_by_mpid.setdefault(int(o.get("MpId")), []).append(int(o.get("ID")))

    warehouse_id = int(ctx.get("warehouse_id", 0))
    safe_dist = ctx.get("safe_dist") or (lambda a, b: BIG)

    giant_tour = ctx.get("giant_tour") or []
    paths_by_mpid = ctx.get("paths_by_mpid") or {}
    entry_exit = ctx.get("entry_exit") or {}
    couriers_json = ctx.get("couriers_json")
    couriers_list = couriers_json.get("Couriers") if isinstance(couriers_json, dict) else []
    couriers_ids = [int(c.get("ID")) for c in couriers_list]

    C_order = _sequence_of_clusters(giant_tour, warehouse_id)
    n = len(C_order)
    if n == 0:
        return {"segments": [], "assignment": [], "routes": [], "stats": {}}

    L = {int(mp): _cluster_internal_move_time(paths_by_mpid.get(int(mp), []), safe_dist) for mp in C_order}
    svc_avg_by_mp = _avg_service_by_mpid(couriers_list, orders_by_mpid)
    # Кластерный нейтральный сервис: avg_service(mp) * |cluster|
    cluster_sizes = {int(mp): len(paths_by_mpid.get(int(mp), [])) for mp in C_order}
    S_cluster_by_mp = {int(mp): int(svc_avg_by_mp.get(int(mp), 0) * cluster_sizes.get(int(mp), 0)) for mp in C_order}
    print("Stage3: neutral service uses cluster-level values (avg*count)")

    B = {k: 0 for k in range(1, n)}
    for k in range(1, n):
        mp_prev = C_order[k - 1]
        mp_next = C_order[k]
        ex = entry_exit.get(int(mp_prev), {}).get("exit")
        en = entry_exit.get(int(mp_next), {}).get("entry")
        if ex is None or en is None:
            B[k] = BIG
        else:
            B[k] = int(safe_dist(int(ex), int(en)))

    cost_avg = _build_prefixes(C_order, L, S_cluster_by_mp, B, entry_exit, safe_dist, warehouse_id)

    # Базовый сплит
    # Диагностика: сколько одиночных кластеров укладывается в лимит
    try:
        single_costs = []
        within = 0
        for idx in range(1, n + 1):
            cst = cost_avg(idx, idx)
            single_costs.append(cst)
            if cst <= LIMIT:
                within += 1
        if single_costs:
            print(f"Stage3: single-block cost min/avg/max = {min(single_costs)}/{int(sum(single_costs)/len(single_costs))}/{max(single_costs)}, within_limit={within}/{n}")
    except Exception:
        pass

    segs = _split_dp_basic(n, cost_avg, limit=LIMIT)
    K = len(segs)
    C = len(couriers_ids)
    # Если сегментов больше, чем курьеров — пробуем ограниченный сплит, затем план Б (жадное слияние)
    if C and K > C:
        segs_cap = _split_dp_with_cap(n, cost_avg, max_segments=C, limit=LIMIT)
        if segs_cap:
            segs = segs_cap
        if len(segs) > C:
            segs = _greedy_merge_segments(segs, cost_avg, LIMIT, C)
        K = len(segs)

    service_map = _courier_service_map(couriers_list)
    # Если нет сегментов — нечего назначать
    if not segs:
        return {"segments": [], "assignment": [], "routes": [], "stats": {}}

    cost = _build_cost_matrix(segs, couriers_ids, C_order, L, B, entry_exit, paths_by_mpid, service_map, safe_dist, warehouse_id)
    matches = _hungarian(cost)

    assignment = []
    for r, cidx in matches:
        cid = couriers_ids[cidx] if cidx < len(couriers_ids) else None
        t = cost[r][cidx] if cidx < len(couriers_ids) else BIG
        if cid is not None and t < BIG:
            assignment.append((r, cidx, int(cid), int(t)))

    # Преобразуем assignment в словарный вид для ремонта
    assignment_dicts = [{"courier_id": cid, "segment_id": r, "time_sec": t} for (r, _, cid, t) in assignment]

    # --- Greedy repair of unassigned/over-limit segments ---
    segments, assignment_dicts = _greedy_repair_unassigned(
        segments=segs,
        assignment_dicts=assignment_dicts,
        couriers_list=couriers_list,
        service_map=service_map,
        safe_dist=safe_dist,
        entry_exit=entry_exit,
        paths_by_mpid=paths_by_mpid,
        C_order=C_order,
        L=L,
        cost_avg=cost_avg,
        warehouse_id=warehouse_id,
    )

    # Назад в кортежный вид для дальнейшей сборки
    assignment = [(a["segment_id"], 0, int(a["courier_id"]), int(a["time_sec"])) for a in assignment_dicts]

    routes = _build_routes(segments, assignment, C_order, paths_by_mpid, warehouse_id)

    # === расширенные метрики ===
    TIME_LIMIT = LIMIT
    assigned_mps = set()
    for (r, _, cid, t) in assignment:
        i, j = segments[r]
        for k in range(i, j + 1):
            assigned_mps.add(C_order[k - 1])

    total_orders = len(orders_list)
    assigned_orders = 0
    for mp in assigned_mps:
        assigned_orders += len(paths_by_mpid.get(int(mp), []))
    unassigned_orders = total_orders - assigned_orders

    route_times = [t for (_, _, _, t) in assignment]
    over_limit = [t for t in route_times if t > TIME_LIMIT]

    stats = {
        "num_clusters": n,
        "num_segments": len(segments),
        "num_couriers": C,
        "num_assigned_routes": len(assignment),
        "total_time_assigned": int(sum(route_times)),
        "min_route_time": int(min(route_times)) if route_times else 0,
        "max_route_time": int(max(route_times)) if route_times else 0,
        "ratio_max_min": (max(route_times) / max(1, min(route_times))) if route_times else 0.0,
        "assigned_orders": int(assigned_orders),
        "unassigned_orders": int(unassigned_orders),
        "over_limit_routes": int(len(over_limit)),
    }

    return {
        "segments": [{"start_idx": i, "end_idx": j, "mps": C_order[i - 1:j]} for (i, j) in segments],
        "assignment": [{"courier_id": cid, "segment_id": r, "time_sec": t} for (r, _, cid, t) in assignment],
        "routes": routes,
        "stats": stats,
    }


