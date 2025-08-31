import time
import random
from copy import deepcopy
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from functools import lru_cache


# Debug facilities
DEBUG = False

def dbg(msg, *a):
    if DEBUG:
        try:
            print("Stage5[dbg] " + str(msg).format(*a))
        except Exception:
            try:
                print("Stage5[dbg] " + str(msg))
            except Exception:
                pass


MAX_WORK_TIME = 12 * 3600


def run(ctx, stage4_out, method: str = "alns", time_cap_sec: Optional[int] = None, seed: int = 42):
    """
    ctx: dict, как в baseline (safe_dist/paths/orders/couriers/и т.д.)
    stage4_out: dict результата Stage 4 (обязательно содержит хотя бы routes_orders или routes)
    method: "alns" по умолчанию
    time_cap_sec: бюджет времени на Stage 5 или None
    seed: фиксированный сид для воспроизводимости
    """
    # Обязательные элементы контекста
    safe_dist = ctx.get("safe_dist")
    # поддержка alias: 'paths' (новый контракт) или 'paths_by_mpid' (старый)
    paths_map = ctx.get("paths") or ctx.get("paths_by_mpid")
    entry_exit = ctx.get("entry_exit") or {}
    orders_json = ctx.get("orders_json") or {}
    couriers_json = ctx.get("couriers_json") or {}

    svc_map: Dict[int, Dict[int, int]] = defaultdict(dict)
    for c in couriers_json.get("Couriers", []) or []:
        try:
            cid = int(c.get("ID"))
        except Exception:
            continue
        for s in c.get("ServiceTimeInMps", []) or []:
            try:
                mp = int(s.get("MpID"))
                st = int(s.get("ServiceTime"))
            except Exception:
                continue
            if st >= 0:
                svc_map[cid][mp] = st

    def service_time_fn(cid: int, mp: int) -> int:
        # Фолбэк 300, как в скоринге
        return int(svc_map.get(cid, {}).get(mp, 300))
    
    # --- Precompute orders per MP and service contribution per (courier, mp) ---
    orders_per_mp: Dict[int, int] = {
        int(k): len(v or [])
        for k, v in (paths_map or {}).items()
        if isinstance(k, int) or (isinstance(k, str) and str(k).isdigit())
    }

    SVC_CONTRIB: Dict[Tuple[int, int], int] = {}
    all_courier_ids = [int(c.get("ID")) for c in (couriers_json.get("Couriers", []) or []) if c.get("ID") is not None]
    for cid in all_courier_ids:
        for mp, cnt in orders_per_mp.items():
            SVC_CONTRIB[(int(cid), int(mp))] = cnt * int(service_time_fn(int(cid), int(mp))) if cnt else 0

    def svc_contrib(cid: int, mp: int) -> int:
        return int(SVC_CONTRIB.get((int(cid), int(mp)), 0))

    def assigned_orders_count_fast(S_loc: Dict[int, List[int]]) -> int:
        used_mps: Set[int] = set()
        for seq in S_loc.values():
            for mp in seq:
                used_mps.add(int(mp))
        return sum(orders_per_mp.get(int(mp), 0) for mp in used_mps)


    t_start_total = time.time()
    # set DEBUG and seed
    global DEBUG
    DEBUG = bool(ctx.get("debug", DEBUG))
    try:
        import numpy as _np  # type: ignore
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch  # type: ignore
        try:
            _torch.manual_seed(seed)
        except Exception:
            pass
    except Exception:
        pass
    try:
        random.seed(seed)
    except Exception:
        pass

    # Валидация входа
    if not isinstance(stage4_out, dict):
        return stage4_out
    routes_orders_s4 = stage4_out.get("routes_orders", {}) or {}

    # ---- быстрый вклад сервиса для (courier, mp) ----
    orders_per_mp = {
        int(k): len(v or [])
        for k, v in (paths_map or {}).items()
        if isinstance(k, int) or (isinstance(k, str) and k.isdigit())
    }

    all_courier_ids = [int(c.get("ID")) for c in (couriers_json.get("Couriers", []) or []) if c.get("ID") is not None]

    SVC_CONTRIB = {}
    for cid in all_courier_ids:
        for mp, cnt in orders_per_mp.items():
            if cnt:
                SVC_CONTRIB[(int(cid), int(mp))] = cnt * int(service_time_fn(int(cid), int(mp)))
            else:
                SVC_CONTRIB[(int(cid), int(mp))] = 0

    def svc_contrib(cid: int, mp: int) -> int:
        return int(SVC_CONTRIB.get((int(cid), int(mp)), 0))

    def assigned_orders_count_fast(S_loc: Dict[int, List[int]]) -> int:
        """Быстро: сумма по уникальным MP (у нас глобальная уникальность соблюдается)."""
        used_mps = set()
        for seq in S_loc.values():
            for mp in seq:
                used_mps.add(int(mp))
        return sum(orders_per_mp.get(int(mp), 0) for mp in used_mps)

    # Склад: по условию — ctx["warehouse_node"] или ctx["warehouse_id"], дефолт 1 (не 0)
    W = int(ctx.get("warehouse_node", ctx.get("warehouse_id", 0)))
    dbg("using warehouse W={}", W)

    # Паспорт контекста
    has_sd = bool(callable(safe_dist))
    has_paths = isinstance(paths_map, dict)
    has_orders = isinstance(orders_json, dict) and bool(orders_json)
    has_couriers = isinstance(couriers_json, dict) and bool(couriers_json)
    num_couriers = len((couriers_json or {}).get("Couriers", []) or [])
    num_orders = len((orders_json or {}).get("Orders", []) or [])
    num_paths_mp = len((paths_map or {})) if has_paths else 0
    dbg(
        "ctx: W={}, time_cap={}, seed={}, has(safe_dist)={}, has(paths)={}, has(orders_json)={}, has(couriers_json)={}, |Couriers|={}, |Orders|={}, |Mp(paths)|={}",
        W, time_cap_sec, seed, has_sd, has_paths, has_orders, has_couriers, num_couriers, num_orders, num_paths_mp,
    )

    if not has_sd or not has_paths:
        print("[Stage5] failover to Stage4: missing safe_dist or paths")
        return stage4_out

    rnd = random.Random(seed)

    # Базовые мапы
    def _build_order_to_mp(ctx_local):
        order_to_mp_local = {}
        orders_dict = ctx_local.get("orders")
        if isinstance(orders_dict, dict) and orders_dict:
            for k, v in orders_dict.items():
                try:
                    oid = int(k) if not isinstance(k, int) else k
                    mpv = v.get("MpId") if isinstance(v, dict) else None
                    if mpv is not None:
                        order_to_mp_local[int(oid)] = int(mpv)
                except Exception:
                    pass
        if len(order_to_mp_local) < 100:
            oj = ctx_local.get("orders_json", {})
            candidates = []
            if isinstance(oj, list):
                candidates = oj
            elif isinstance(oj, dict):
                for key in ("Orders", "orders", "items", "data"):
                    if key in oj and isinstance(oj[key], list):
                        candidates = oj[key]
                        break
                if not candidates and oj and all(isinstance(v, dict) for v in oj.values()):
                    candidates = list(oj.values())
            for o in candidates:
                try:
                    oid = o.get("ID") if "ID" in o else o.get("Id", o.get("OrderId"))
                    mpv = o.get("MpId") if "MpId" in o else o.get("mpId")
                    if oid is not None and mpv is not None:
                        order_to_mp_local[int(oid)] = int(mpv)
                except Exception:
                    pass
        dbg("order_to_mp built: size={}", len(order_to_mp_local))
        return order_to_mp_local

    order_to_mp: Dict[int, int] = _build_order_to_mp(ctx)
    # фильтруем псевдополигон склада (MpId == W)
    filtered_warehouse = 0
    if isinstance(W, int):
        cleaned = {}
        for oid, mp in order_to_mp.items():
            if int(mp) == int(W):
                filtered_warehouse += 1
                continue
            cleaned[int(oid)] = int(mp)
        order_to_mp = cleaned
    dbg("filtered warehouse-like orders (MpId==W): {}", filtered_warehouse)
    # Mp mismatch diagnostics (orders vs paths)
    try:
        mp_in_orders = set(int(x) for x in set(order_to_mp.values()))
        mp_in_paths = set(int(x) for x in (paths_map or {}).keys()) if has_paths else set()
        missing_in_paths = sorted(list(mp_in_orders - mp_in_paths))[:5]
        missing_in_orders = sorted(list(mp_in_paths - mp_in_orders))[:5]
        if missing_in_paths or missing_in_orders:
            dbg("mp mismatch: missing_in_paths={}, missing_in_orders={}", missing_in_paths, missing_in_orders)
    except Exception:
        pass


    # Кэш seg_time: (courier_id, tuple(mp_seq)) -> total_time_with_services
    seg_time_cache: Dict[tuple, int] = {}

    def route_orders_from_mp_seq(mp_seq: List[int]) -> List[int]:
        orders: List[int] = []
        for mp in mp_seq:
            core = (paths_map or {}).get(int(mp)) or []
            orders.extend(int(x) for x in core)
        return orders

    # --- Portal-based time primitives ---
    INF = 10 ** 12

    @lru_cache(maxsize=None)
    def _get_entries(mp: int) -> List[int]:
        rec = entry_exit.get(int(mp)) or {}
        entries = list(rec.get("entries", []) or [])
        if not entries:
            core = (paths_map or {}).get(int(mp)) or []
            if core:
                entries = [int(core[0])]
        return [int(x) for x in entries]

    @lru_cache(maxsize=None)
    def _get_exits(mp: int) -> List[int]:
        rec = entry_exit.get(int(mp)) or {}
        exits = list(rec.get("exits", []) or [])
        if not exits:
            core = (paths_map or {}).get(int(mp)) or []
            if core:
                exits = [int(core[-1])]
        return [int(x) for x in exits]

    ENTRY_CACHE: Dict[int, Tuple[int, int]] = {}
    EXIT_CACHE: Dict[int, Tuple[int, int]] = {}
    INTRA_CACHE: Dict[int, int] = {}
    PORTAL_CACHE: Dict[Tuple[int, int], Tuple[int, int, int]] = {}

    @lru_cache(maxsize=None)
    def _entry_cached(mp: int) -> Tuple[int, int]:
        mp = int(mp)
        v = ENTRY_CACHE.get(mp)
        if v is not None:
            return v
        best_t, best_e = INF, -1
        for e in _get_entries(mp):
            d = int(safe_dist(int(W), int(e)))
            if d < best_t:
                best_t, best_e = d, int(e)
        ENTRY_CACHE[mp] = (best_t, best_e)
        return ENTRY_CACHE[mp]

    def _exit_cached(mp: int) -> Tuple[int, int]:
        mp = int(mp)
        v = EXIT_CACHE.get(mp)
        if v is not None:
            return v
        best_t, best_x = INF, -1
        for x in _get_exits(mp):
            d = int(safe_dist(int(x), int(W)))
            if d < best_t:
                best_t, best_x = d, int(x)
        EXIT_CACHE[mp] = (best_t, best_x)
        return EXIT_CACHE[mp]

    @lru_cache(maxsize=None)
    def _intra_cached(mp: int) -> int:
        mp = int(mp)
        v = INTRA_CACHE.get(mp)
        if v is not None:
            return v
        core = (paths_map or {}).get(mp) or []
        if not core or len(core) <= 1:
            INTRA_CACHE[mp] = 0
            return 0
        t = 0
        for a, b in zip(core[:-1], core[1:]):
            t += int(safe_dist(int(a), int(b)))
        INTRA_CACHE[mp] = int(t)
        return INTRA_CACHE[mp]

    @lru_cache(maxsize=None)
    def _portal_cached(a: int, b: int) -> Tuple[int, int, int]:
        a = int(a); b = int(b)
        key = (a, b)
        v = PORTAL_CACHE.get(key)
        if v is not None:
            return v
        best_t, best_x, best_e = INF, -1, -1
        for x in _get_exits(a):
            for e in _get_entries(b):
                d = int(safe_dist(int(x), int(e)))
                if d < best_t:
                    best_t, best_x, best_e = d, int(x), int(e)
        PORTAL_CACHE[key] = (best_t, best_x, best_e)
        return PORTAL_CACHE[key]

    def route_travel_time_portal(mp_seq: List[int]) -> int:
        if not mp_seq:
            return 0
        t0, _ = _entry_cached(int(mp_seq[0]))
        if t0 >= INF:
            return INF
        t = t0 + _intra_cached(int(mp_seq[0]))
        for a, b in zip(mp_seq[:-1], mp_seq[1:]):
            pt, _, _ = _portal_cached(int(a), int(b))
            if pt >= INF:
                return INF
            t += pt + _intra_cached(int(b))
        te, _ = _exit_cached(int(mp_seq[-1]))
        if te >= INF:
            return INF
        return int(t + te)

    def _delta_travel_insert_portal(seq: List[int], pos: int, mp: int) -> int:
        """
        Изменение travel-времени при вставке mp в позицию pos (0..len).
        Использует _entry_cached/_exit_cached/_portal_cached/_intra_cached.
        """
        mp = int(mp)
        if not seq:
            t0, _ = _entry_cached(mp)
            te, _ = _exit_cached(mp)
            if t0 >= INF or te >= INF:
                return INF
            return int(t0 + _intra_cached(mp) + te)

        # вставка в начало
        if pos == 0:
            first = int(seq[0])
            t_new_first, _ = _entry_cached(mp)
            t_old_first, _ = _entry_cached(first)
            pt, _, _ = _portal_cached(mp, first)
            if t_new_first >= INF or pt >= INF or t_old_first >= INF:
                return INF
            return int((t_new_first + _intra_cached(mp) + pt) - t_old_first)

        # вставка в конец
        if pos == len(seq):
            last = int(seq[-1])
            pt, _, _ = _portal_cached(last, mp)
            te_new, _ = _exit_cached(mp)
            te_old, _ = _exit_cached(last)
            if pt >= INF or te_new >= INF or te_old >= INF:
                return INF
            return int(pt + _intra_cached(mp) + te_new - te_old)

        # вставка между A и B
        A = int(seq[pos - 1])
        B = int(seq[pos])
        pt1, _, _ = _portal_cached(A, mp)
        pt2, _, _ = _portal_cached(mp, B)
        pt_old, _, _ = _portal_cached(A, B)
        if pt1 >= INF or pt2 >= INF or pt_old >= INF:
            return INF
        return int(pt1 + _intra_cached(mp) + pt2 - pt_old)
    
    # --- Delta helpers for insert ---
    K_POS = 24  # можно тюнить 12..32

    def _delta_travel_insert_portal(seq: List[int], pos: int, mp: int) -> int:
        mp = int(mp)
        if not seq:
            t0, _ = _entry_cached(mp)
            te, _ = _exit_cached(mp)
            if t0 >= INF or te >= INF:
                return INF
            return int(t0 + _intra_cached(mp) + te)

        if pos == 0:  # before first
            first = int(seq[0])
            t_new_first, _ = _entry_cached(mp)
            t_old_first, _ = _entry_cached(first)
            pt, _, _ = _portal_cached(mp, first)
            if t_new_first >= INF or pt >= INF or t_old_first >= INF:
                return INF
            return int((t_new_first + _intra_cached(mp) + pt) - t_old_first)

        if pos == len(seq):  # after last
            last = int(seq[-1])
            pt, _, _ = _portal_cached(last, mp)
            te_new, _ = _exit_cached(mp)
            te_old, _ = _exit_cached(last)
            if pt >= INF or te_new >= INF or te_old >= INF:
                return INF
            return int(pt + _intra_cached(mp) + te_new - te_old)

        A = int(seq[pos - 1]); B = int(seq[pos])
        pt1, _, _ = _portal_cached(A, mp)
        pt2, _, _ = _portal_cached(mp, B)
        pt_old, _, _ = _portal_cached(A, B)
        if pt1 >= INF or pt2 >= INF or pt_old >= INF:
            return INF
        return int(pt1 + _intra_cached(mp) + pt2 - pt_old)

    def _candidate_positions_for_insert(seq: List[int], mp: int, k: int = K_POS) -> List[int]:
        n = len(seq)
        if n <= 2:
            return list(range(0, n + 1))

        uniq = list(dict.fromkeys(int(x) for x in seq))
        pairs = []
        for m in uniq:
            d, _, _ = _portal_cached(int(mp), int(m))
            pairs.append((d, m))
        pairs.sort(key=lambda x: x[0])
        cand = set([0, n])  # пробуем крайние позиции

        for _, near_mp in pairs[:max(1, k // 2)]:
            for i, val in enumerate(seq):
                if int(val) == int(near_mp):
                    for p in (i, i + 1, i - 1, i + 2):
                        if 0 <= p <= n:
                            cand.add(p)
            if len(cand) >= k:
                break

        if len(cand) < k:
            step = max(1, n // max(1, (k - len(cand))))
            for p in range(0, n + 1, step):
                cand.add(p)
                if len(cand) >= k:
                    break

        out = sorted(cand)
        return out[:k] if len(out) > k else out


    def _entry_cached(mp: int) -> Tuple[int, int]:
        mp = int(mp)
        if mp in ENTRY_CACHE:
            return ENTRY_CACHE[mp]
        best_t, best_e = INF, -1
        for e in _get_entries(mp):
            d = int(safe_dist(int(W), int(e)))
            if d < best_t:
                best_t, best_e = d, int(e)
        ENTRY_CACHE[mp] = (best_t, best_e)
        return ENTRY_CACHE[mp]

    def _exit_cached(mp: int) -> Tuple[int, int]:
        mp = int(mp)
        if mp in EXIT_CACHE:
            return EXIT_CACHE[mp]
        best_t, best_x = INF, -1
        for x in _get_exits(mp):
            d = int(safe_dist(int(x), int(W)))
            if d < best_t:
                best_t, best_x = d, int(x)
        EXIT_CACHE[mp] = (best_t, best_x)
        return EXIT_CACHE[mp]

    def _intra_cached(mp: int) -> int:
        mp = int(mp)
        if mp in INTRA_CACHE:
            return INTRA_CACHE[mp]
        core = (paths_map or {}).get(mp) or []
        if not core or len(core) <= 1:
            INTRA_CACHE[mp] = 0
            return 0
        t = 0
        for a, b in zip(core[:-1], core[1:]):
            t += int(safe_dist(int(a), int(b)))
        INTRA_CACHE[mp] = int(t)
        return INTRA_CACHE[mp]

    def _portal_cached(a: int, b: int) -> Tuple[int, int, int]:
        a = int(a); b = int(b)
        key = (a, b)
        if key in PORTAL_CACHE:
            return PORTAL_CACHE[key]
        best_t, best_x, best_e = INF, -1, -1
        for x in _get_exits(a):
            for e in _get_entries(b):
                d = int(safe_dist(int(x), int(e)))
                if d < best_t:
                    best_t, best_x, best_e = d, int(x), int(e)
        PORTAL_CACHE[key] = (best_t, best_x, best_e)
        return PORTAL_CACHE[key]


    def route_travel_time_portal(mp_seq: List[int]) -> int:
        if not mp_seq:
            return 0
        # W -> first mp entry
        t0, _ = _entry_cached(int(mp_seq[0]))
        if t0 >= INF:
            return INF
        t = t0 + _intra_cached(int(mp_seq[0]))
        # inter-mp
        for a, b in zip(mp_seq[:-1], mp_seq[1:]):
            pt, _, _ = _portal_cached(int(a), int(b))
            if pt >= INF:
                return INF
            t += pt + _intra_cached(int(b))
        # last mp exit -> W
        te, _ = _exit_cached(int(mp_seq[-1]))
        if te >= INF:
            return INF
        t += te
        return int(t)

    # Кэш на (courier_id, tuple(seq))
    seg_time_cache: Dict[Tuple[int, Tuple[int, ...]], int] = {}

    def T_c(cid: int, mp_seq: List[int]) -> int:
        key = (int(cid), tuple(int(x) for x in mp_seq))
        v = seg_time_cache.get(key)
        if v is not None:
            return v
        if not mp_seq:
            seg_time_cache[key] = 0
            return 0
        travel = route_travel_time_portal(mp_seq)
        if travel >= INF:
            seg_time_cache[key] = INF
            return INF
        s = 0
        # последовательности мы обычно «сжимаем», но даже если повтор есть — сервис-вклад суммируется корректно
        for mp in mp_seq:
            s += svc_contrib(int(cid), int(mp))
        res = int(travel + s)
        seg_time_cache[key] = res
        return res



    def T_c_raw(cid: int, mp_seq: List[int]) -> int:
        travel = route_travel_time_portal(mp_seq)
        if travel >= INF:
            return INF
        s = 0
        for mp in mp_seq:
            num_orders = len((paths_map or {}).get(int(mp), []) or [])
            if num_orders:
                s += num_orders * service_time_fn(int(cid), int(mp))
        return int(travel + s)


    # --- Gentle trimming to 12h limit ---
    SHIFT_LIMIT = 12 * 3600

    def trim_to_shift_limit(cid: int, mp_seq: List[int], time_fn, limit: int = SHIFT_LIMIT, max_block: int = 1) -> Tuple[List[int], List[int], int]:
        """Remove minimal tail blocks until route time <= limit. Returns (new_seq, removed_tail, new_time)."""
        removed: List[int] = []
        if not mp_seq:
            return mp_seq, removed, 0
        seq = [int(x) for x in mp_seq]
        cur_time = int(time_fn(cid, seq))
        if cur_time <= limit:
            return seq, removed, cur_time
        # Greedy tail removal
        while seq and cur_time > limit:
            b = min(max_block, len(seq))
            best_b = 1
            best_time = cur_time
            trimmed = False
            for bb in range(1, b + 1):
                cand = seq[:-bb]
                t = int(time_fn(cid, cand))
                if t <= limit:
                    seq = cand
                    cur_time = t
                    trimmed = True
                    break
                if t < best_time:
                    best_time = t
                    best_b = bb
            if trimmed:
                break
            # none achieved the limit; cut the best_b and continue
            seq = seq[:-best_b]
            cur_time = best_time
        if not seq:
            # everything removed; all original become removed
            return [], [int(x) for x in mp_seq], 0
        removed = [int(x) for x in mp_seq[len(seq):]]
        return seq, removed, cur_time

    def compress_seq(seq: List[int]) -> List[int]:
        if not seq:
            return []
        out = [int(seq[0])]
        for x in seq[1:]:
            x = int(x)
            if x != out[-1]:
                out.append(x)
        return out

    def insert_mp_compressed(seq: List[int], pos: int, mp: int) -> List[int]:
        cand = list(seq[:pos]) + [int(mp)] + list(seq[pos:])
        return compress_seq(cand)

    # Инициализация из Stage 4 (сбор сегментов вдоль C_order)
    try:
        if "routes_orders" in stage4_out and isinstance(routes_orders_s4, dict):
            assigned_s4 = sum(len(v or []) for v in routes_orders_s4.values())
            dbg("stage4 routes_orders present: couriers={}, assigned_orders={}", len(routes_orders_s4), assigned_s4)
        else:
            dbg("stage4 routes_orders missing")
    except Exception:
        pass

    # order_to_mp already built; get C_order
    C_order = ctx.get("C_order") or (paths_map or {}).get("C_order") or []
    if not isinstance(C_order, list):
        C_order = []
    C_order = [int(x) for x in C_order if isinstance(x, (int, str))]

    # mp ownership by couriers from Stage4 orders
    from collections import defaultdict as _dd, Counter as _Counter
    mp_counts = _dd(_Counter)
    for cid, oids in (routes_orders_s4 or {}).items():
        try:
            icid = int(cid)
        except Exception:
            continue
        for oid in (oids or []):
            mp = order_to_mp.get(int(oid))
            if mp is None:
                continue
            mp_counts[int(mp)][icid] += 1

    mp_owner: Dict[int, int] = {}
    for mp, cnt in mp_counts.items():
        if not cnt:
            continue
        max_val = max(cnt.values())
        cands = [c for c, k in cnt.items() if k == max_val]
        owner = int(min(cands))
        mp_owner[int(mp)] = owner

    segments_by_cid: Dict[int, List[List[int]]] = defaultdict(list)
    n = len(C_order)
    k = 0
    while k < n:
        mp = int(C_order[k])
        owner = mp_owner.get(mp)
        if owner is None:
            k += 1
            continue
        start = k
        while k + 1 < n and mp_owner.get(int(C_order[k + 1])) == owner:
            k += 1
        end = k
        seg_mps = [int(x) for x in C_order[start:end + 1]]
        if len(seg_mps) > 0:
            segments_by_cid[int(owner)].append(seg_mps)
        k += 1

    # Build initial S from Stage4 routes_orders directly with int casting and miss counting
    miss = 0
    S: Dict[int, List[int]] = {}
    for cid_any, core_orders in (routes_orders_s4 or {}).items():
        try:
            cid = int(cid_any)
        except Exception:
            continue
        mp_seq: List[int] = []
        last = None
        for oid_any in (core_orders or []):
            try:
                oid = int(oid_any)
            except Exception:
                miss += 1
                continue
            mp = order_to_mp.get(oid)
            if mp is None:
                miss += 1
                continue
            if mp != last:
                mp_seq.append(int(mp))
                last = mp
        if mp_seq:
            S[int(cid)] = mp_seq
    dbg("init mp-seqs: couriers_with_route={}, total_mp_in_routes={}, unique_mp_used={}, misses={}",
        sum(1 for s in S.values() if s),
        sum(len(s) for s in S.values()),
        len({mp for seq in S.values() for mp in seq}),
        miss)
    # Если внезапно пусто — фоллбэк: перестраиваем order_to_mp и повторяем
    if not any(S.values()) and routes_orders_s4:
        order_to_mp = _build_order_to_mp(ctx)
        miss = 0
        S = {}
        for cid_any, core_orders in (routes_orders_s4 or {}).items():
            try:
                cid = int(cid_any)
            except Exception:
                continue
            mp_seq = []
            last = None
            for oid_any in (core_orders or []):
                try:
                    oid = int(oid_any)
                except Exception:
                    miss += 1
                    continue
                mp = order_to_mp.get(oid)
                if mp is None:
                    miss += 1
                    continue
                if mp != last:
                    mp_seq.append(int(mp))
                    last = mp
            if mp_seq:
                S[int(cid)] = mp_seq
        dbg("fallback init mp-seqs: couriers_with_route={}, total_mp_in_routes={}, unique_mp_used={}, misses={}",
            sum(1 for s in S.values() if s),
            sum(len(s) for s in S.values()),
            len({mp for seq in S.values() for mp in seq}),
            miss)

    # diagnostics on segments
    try:
        num_cid_with_segs = sum(1 for segs in segments_by_cid.values() if segs)
        total_segments = sum(len(segs) for segs in segments_by_cid.values())
        all_mps = []
        for segs in segments_by_cid.values():
            for s in segs:
                all_mps.extend(s)
        covered_unique = len(set(all_mps))
        owners_unique = len(set(mp_owner.keys()))
        dbg("segments init: couriers_with_segs={}, total_segments={}, covered_mp_unique={}, owned_mp={} ", num_cid_with_segs, total_segments, covered_unique, owners_unique)
        assert len(all_mps) == len(set(all_mps)), "duplicate mp in segments"
        assert covered_unique == sum(1 for mp in C_order if int(mp) in mp_owner), "coverage mismatch"
    except Exception as _e:
        try:
            dbg("segments init diagnostics failed: {}", str(_e))
        except Exception:
            pass

    # enforce unique ownership of each MpId across all couriers
    owner_by_mp = {}
    duplicates = 0
    CR: List[int] = []
    for cid, seq in list(S.items()):
        new_seq = []
        for mp in seq:
            if mp not in owner_by_mp:
                owner_by_mp[mp] = cid
                new_seq.append(mp)
            else:
                CR.append(int(mp))
                duplicates += 1
        S[cid] = new_seq
    dbg("init unique-ownership: removed_dups={}, CR_size_after_dedup={}", duplicates, len(CR))

    # init mp seqs summary
    try:
        couriers_with_route = sum(1 for v in S.values() if v)
        total_mp_in_routes = sum(len(v) for v in S.values())
        unique_mp_used = len(set(x for seq in S.values() for x in seq))
        dbg("init mp-seqs: couriers_with_route={}, total_mp_in_routes={}, unique_mp_used={}", couriers_with_route, total_mp_in_routes, unique_mp_used)
        # top by orders and by mp
        orders_count = {int(cid): len(routes_orders_s4.get(cid, [])) for cid in routes_orders_s4}
        mp_count = {int(cid): len(seq) for cid, seq in S.items()}
        top_orders = sorted(orders_count.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_mp = sorted(mp_count.items(), key=lambda kv: kv[1], reverse=True)[:3]
        fmt_orders = ", ".join([f"{cid}:{orders_count.get(cid,0)}/{mp_count.get(cid,0)}" for cid, _ in top_orders])
        fmt_mp = ", ".join([f"{cid}:{orders_count.get(cid,0)}/{mp_count.get(cid,0)}" for cid, _ in top_mp])
        dbg("init top by orders: [{}]", fmt_orders)
        dbg("init top by mp:     [{}]", fmt_mp)
    except Exception:
        pass

    # Список всех курьеров из stage4_out (сохраняем пустые маршруты)
    for cid_str in routes_orders_s4.keys():
        cid = int(cid_str)
        if cid not in S:
            S[cid] = []

    # Trim к 12 часам и собрать CR (пул снятых MpId)
    # init T_c stats (before trim)
    try:
        vals = []
        for cid, mp_seq in S.items():
            vals.append((cid, T_c_raw(cid, mp_seq)))
        if vals:
            mins = min(v for _, v in vals)
            maxs = max(v for _, v in vals)
            max_cid = max(vals, key=lambda x: x[1])[0]
            avgs = int(sum(v for _, v in vals) / max(1, len(vals)))
            over_limit = sum(1 for _, v in vals if v > MAX_WORK_TIME)
            dbg("init T_c stats: min={}s, max={}s (cid={}), avg={}s, over_limit={}", mins, maxs, max_cid, avgs, over_limit)
    except Exception:
        pass
    # Gentle trimming pass to fit 12h without wiping routes
    total_mp_before = sum(len(v) for v in S.values())
    trimmed_couriers = 0
    trimmed_mps = 0
    emptied = 0
    for cid, mp_seq in S.items():
        # normalize seq types
        if mp_seq and not isinstance(mp_seq[0], int):
            mp_seq = [int(x) for x in mp_seq]
            S[cid] = mp_seq
        t0 = T_c(cid, mp_seq)
        if t0 > SHIFT_LIMIT:
            new_seq, removed_tail, t1 = trim_to_shift_limit(cid, mp_seq, T_c, SHIFT_LIMIT, max_block=1)
            S[cid] = new_seq
            if removed_tail:
                CR.extend(int(x) for x in removed_tail)
                trimmed_couriers += 1
                trimmed_mps += len(removed_tail)
            if not new_seq:
                emptied += 1
            try:
                dbg("trim> cid={}, before={}s, after={}s, removed_mp={}, CR_size={}", cid, t0, t1, len(removed_tail), len(CR))
            except Exception:
                pass
    try:
        dbg("trim summary: couriers={}, removed_mp={}, pool={}, emptied_routes={}", trimmed_couriers, trimmed_mps, len(CR), emptied)
        total_mp_after = sum(len(v) for v in S.values())
        if total_mp_before != total_mp_after + len(CR):
            dbg("WARNING: mass mismatch: before={} after={} pool={}", total_mp_before, total_mp_after, len(CR))
        else:
            dbg("mass preserved: before={} == after={} + pool={}", total_mp_before, total_mp_after, len(CR))
    except Exception:
        pass

    # Жадная реинсерция из CR
    def best_insertion_for_mp(mp: int, S_loc: Dict[int, List[int]]):
        best = None  # (delta, cid, pos, new_seq)
        for cid, seq in S_loc.items():
            base = T_c(cid, seq)
            if base >= INF:
                continue
            for pos in range(0, len(seq) + 1):
                cand_seq = insert_mp_compressed(seq, pos, mp)
                t_new = T_c(cid, cand_seq)
                if t_new >= INF or t_new > MAX_WORK_TIME:
                    continue
                delta = t_new - base
                if (best is None) or (delta < best[0]):
                    best = (delta, cid, pos, cand_seq)
        return best

    if CR:
        rescued = 0
        for mp in list(CR):
            cand = best_insertion_for_mp(mp, S)
            if cand is not None:
                _, cid, pos, new_seq = cand
                S[cid] = new_seq
                CR.remove(mp)
                rescued += 1
        try:
            dbg("greedy reinsertion: inserted={}, still_in_CR={}", rescued, len(CR))
        except Exception:
            pass

    # precompute orders-per-mp for fast counts
    orders_per_mp = {
        int(k): len(v or [])
        for k, v in (paths_map or {}).items()
        if isinstance(k, int) or (isinstance(k, str) and k.isdigit())
    }


    def get_svc_contrib(cid: int, mp: int) -> int:
        """Быстрый доступ к вкладу времени обслуживания данного MP у курьера."""
        key = (int(cid), int(mp))
        v = svc_contrib.get(key)
        if v is not None:
            return v
        cnt = int(orders_per_mp.get(int(mp), 0))
        if cnt == 0:
            return 0
        return cnt * int(service_time_fn(int(cid), int(mp)))


    TOTAL_ORDERS = len(orders_json.get("Orders", []) or [])
    PENALTY = 3000

    def objective(S_loc: Dict[int, List[int]]) -> int:
        travel_plus_service = 0
        for cid, seq in S_loc.items():
            t = T_c(cid, seq)
            if t >= INF:
                return INF
            travel_plus_service += t
        covered = assigned_orders_count_fast(S_loc)
        assigned = min(TOTAL_ORDERS, covered)
        remaining = TOTAL_ORDERS - assigned
        return int(travel_plus_service + PENALTY * remaining)



    # Быстрая проверка допустимости
    def is_feasible(S_loc: Dict[int, List[int]]) -> bool:
        for cid, seq in S_loc.items():
            if T_c(cid, seq) > MAX_WORK_TIME:
                return False
        return True

    # ––– ALNS: операторы
    class SequenceRemoval:
        def __init__(self, rnd):
            self.rnd = rnd
        def remove_polygons(self, S_loc: Dict[int, List[int]]) -> Set[int]:
            # выбираем случайный маршрут с непустой последовательностью
            non_empty = [(cid, seq) for cid, seq in S_loc.items() if seq]
            if not non_empty:
                return set()
            cid, seq = self.rnd.choice(non_empty)
            if not seq:
                return set()
            length = len(seq)
            # окно 1..min(ceil(0.3*len), 5)
            max_k = max(1, min(max(1, (length + 2) // 3), 5))
            k = self.rnd.randint(1, max_k)
            start = self.rnd.randint(0, max(0, length - k))
            taken = set(int(x) for x in seq[start:start + k])
            # снять из маршрута
            new_seq = list(seq[:start]) + list(seq[start + k:])
            S_loc[cid] = compress_seq(new_seq)
            return taken

    class WorstRouteRemoval:
        def __init__(self, rnd):
            self.rnd = rnd
        def remove_polygons(self, S_loc: Dict[int, List[int]]) -> Set[int]:
            if not S_loc:
                return set()
            worst_cid = None
            worst_val = -1
            for cid, seq in S_loc.items():
                val = T_c(cid, seq)
                if val > worst_val and seq:
                    worst_val = val
                    worst_cid = cid
            if worst_cid is None:
                return set()
            seq = S_loc[worst_cid]
            if not seq:
                return set()
            # снять 1..min(5, len(seq)) из хвоста
            k = self.rnd.randint(1, min(5, len(seq)))
            taken = set(int(x) for x in seq[-k:])
            S_loc[worst_cid] = list(seq[:-k])
            return taken

    # Insertion: Regret-2
    class RegretKInsertion:
        def __init__(self, rnd, warehouse_id, k=2, eps=0.01):
            self.rnd = rnd
            self.warehouse_id = warehouse_id
            self.k = k
            self.eps = eps
            self.last_fail_no_pos = 0
            self.last_fail_lim_viol = 0
            self.last_fail_feas_sum = 0
            self.last_cr_size = 0

        def all_feasible_positions(self, mp: int, S_loc: Dict[int, List[int]]) -> List[Tuple[int,int,int]]:
            """Возвращает отсортированные кандидаты (delta_total, cid, pos) БЕЗ сборки new_seq."""
            cands = []
            for cid, seq in S_loc.items():
                base = T_c(cid, seq)
                if base >= INF:
                    continue
                for pos in _candidate_positions_for_insert(seq, mp, K_POS):
                    d_travel = _delta_travel_insert_portal(seq, pos, mp)
                    if d_travel >= INF:
                        continue
                    new_total = base + d_travel + svc_contrib(int(cid), int(mp))
                    if new_total > MAX_WORK_TIME:
                        continue
                    cands.append((int(new_total - base), int(cid), int(pos)))
            cands.sort(key=lambda x: x[0])
            return cands

        def insert_all(self, CR_iter: Set[int], S_loc: Dict[int, List[int]], prev_owner: Optional[Dict[int, int]] = None) -> bool:
            remaining = set(int(x) for x in CR_iter)
            while remaining:
                best_choice = None  # (regret, -best_delta, mp, best_tuple)
                feas_sum = 0; no_pos = 0; lim_viol = 0
                for mp in list(remaining):
                    potential_positions = sum(len(S_loc[cid]) + 1 for cid in S_loc)  # только для статистики
                    cands = self.all_feasible_positions(mp, S_loc)
                    feas_sum += len(cands)
                    if not cands:
                        if potential_positions > 0: lim_viol += 1
                        else: no_pos += 1
                        continue
                    best = cands[0]
                    second = cands[1] if len(cands) > 1 else (best[0], None, None)
                    regret = (second[0] - best[0]) if second[1] is not None else 0
                    cand = (regret, -best[0], mp, best)
                    if (best_choice is None) or (cand > (best_choice[0], best_choice[1], best_choice[2], best_choice[3])):
                        best_choice = cand
                if best_choice is None:
                    self.last_fail_no_pos = int(no_pos)
                    self.last_fail_lim_viol = int(lim_viol)
                    self.last_fail_feas_sum = int(feas_sum)
                    self.last_cr_size = int(len(CR_iter))
                    return False

                _, _, mp, (delta, cid, pos) = best_choice
                # формируем new_seq один раз — только для победителя
                new_seq = insert_mp_compressed(S_loc[cid], pos, mp)
                S_loc[cid] = new_seq
                if DEBUG:
                    src = prev_owner.get(int(mp)) if isinstance(prev_owner, dict) else None
                    dbg("[ALNS] move mp={} from cid={} to cid={} at pos={}, Δ={}", int(mp), src, cid, pos, int(delta))
                remaining.remove(mp)
            return True


    # Настройки ALNS
    removals = [SequenceRemoval(rnd), WorstRouteRemoval(rnd)]
    insertions = [RegretKInsertion(rnd, warehouse_id=W)]

    rem_weights = [1.0 for _ in removals]
    ins_weights = [1.0 for _ in insertions]
    rem_scores = [0.0 for _ in removals]
    ins_scores = [0.0 for _ in insertions]
    accept_improve_reward = 6.0
    accept_worse_reward = 3.0
    reject_reward = 0.0
    reaction = 0.2

    # Приёмка: LAHC
    L = 96
    lahc_buf = [objective(S)] * L
    iter_idx = 0
    best = deepcopy(S)
    best_obj = objective(best)
    cur = deepcopy(S)
    cur_obj = objective(cur)

    # Печать старта
    try:
        non_empty_routes = sum(1 for v in S.values() if v)
        dbg("init: routes={}, assigned_orders={}, obj={}", non_empty_routes, sum(len(route_orders_from_mp_seq(seq)) for seq in S.values()), best_obj)
        try:
            init_assigned = assigned_orders_count_fast(S)
            dbg("init objective: travel+svc={} penalty={}*{} => obj={}",
                sum(T_c(cid, seq) for cid, seq in S.items()),
                PENALTY, max(0, TOTAL_ORDERS - init_assigned),
                objective(S))
        except Exception:
            pass
        if sum(len(route_orders_from_mp_seq(seq)) for seq in S.values()) == 0:
            dbg("WARNING: init assigned=0 — ALNS will start from empty state")
    except Exception:
        pass

    # безопасный дедлайн даже если time_cap_sec=None/<=0
    if time_cap_sec is None or float(time_cap_sec) <= 0:
        time_deadline = time.time() + 30.0
    else:
        time_deadline = time.time() + float(time_cap_sec)

    accepted = 0
    iterations = 0
    total_removed = 0
    # window diagnostics
    WIN = 500
    win_acc = 0
    win_rej = 0
    win_imp = 0
    win_no_pos = 0
    win_lim_viol = 0
    win_empty_cr = 0
    last_rem_name = ""
    last_ins_name = ""

    def sample_by_weights(items, weights):
        s = sum(weights)
        if s <= 0:
            return rnd.choice(items)
        r = rnd.random() * s
        acc = 0.0
        for it, w in zip(items, weights):
            acc += w
            if r <= acc:
                return it
        return items[-1]

    # Главный цикл
    while time.time() < time_deadline:
        iterations += 1

        # выбрать операторы
        rem = sample_by_weights(removals, rem_weights)
        ins = sample_by_weights(insertions, ins_weights)
        last_rem_name = rem.__class__.__name__
        last_ins_name = ins.__class__.__name__

        # разрушение: снять подпоследовательность
        S_work = deepcopy(cur)
        removed_set = rem.remove_polygons(S_work)
        if not removed_set and CR:
            removed_set = set(int(x) for x in CR)
            CR.clear()
        elif removed_set and CR:
            removed_set |= set(int(x) for x in CR)
            CR.clear()
        total_removed += len(removed_set)
        if not removed_set:
            win_empty_cr += 1

        # восстановление Regret-2
        S_new = deepcopy(S_work)
        feasible = True
        if removed_set:
            # prev_owner map for logs
            prev_owner = {}
            for cid_k, seq_k in S_work.items():
                for mp in seq_k:
                    prev_owner[int(mp)] = int(cid_k)
            ok = insertions[0].insert_all(removed_set, S_new, prev_owner)  # единственный вставщик: Regret2
            feasible = bool(ok)

        if not feasible:
            # штраф за отклонение
            i_r = removals.index(rem)
            i_i = insertions.index(insertions[0])
            rem_scores[i_r] += reject_reward
            ins_scores[i_i] += reject_reward
            # window counters
            win_rej += 1
            try:
                total_potential = sum(len(S_work[cid]) + 1 for cid in S_work)
                dbg("reject: CR_iter={} potential_positions={} feas_positions_sum={} no_pos={} lim_viol={} op={}/{}",
                    insertions[0].last_cr_size, total_potential, insertions[0].last_fail_feas_sum, insertions[0].last_fail_no_pos, insertions[0].last_fail_lim_viol, last_rem_name, last_ins_name)
            except Exception:
                pass
            continue

        # опциональная локальная оптимизация: простая межмаршрутная swap 1-1
        def try_swap(cur_S: Dict[int, List[int]]):
            # одно случайное улучшение
            cids = [cid for cid in cur_S.keys()]
            if len(cids) < 2:
                return cur_S
            a, b = rnd.sample(cids, 2)
            ra = cur_S[a]
            rb = cur_S[b]
            if not ra or not rb:
                return cur_S
            ia = rnd.randrange(0, len(ra))
            ib = rnd.randrange(0, len(rb))
            if ra[ia] == rb[ib]:
                return cur_S
            new_ra = list(ra)
            new_rb = list(rb)
            new_ra[ia], new_rb[ib] = new_rb[ib], new_ra[ia]
            new_ra = compress_seq(new_ra)
            new_rb = compress_seq(new_rb)
            if T_c(a, new_ra) <= MAX_WORK_TIME and T_c(b, new_rb) <= MAX_WORK_TIME:
                out = deepcopy(cur_S)
                out[a] = new_ra
                out[b] = new_rb
                return out
            return cur_S

        S_new = try_swap(S_new)

        new_obj = objective(S_new)

        # Приёмка LAHC
        buf_val = lahc_buf[iter_idx % L]
        accepted_move = (new_obj <= cur_obj) or (new_obj <= buf_val)
        if new_obj < best_obj:
            best = deepcopy(S_new)
            best_obj = int(new_obj)
            cur = deepcopy(S_new)
            cur_obj = int(new_obj)
            accepted += 1
            win_acc += 1
            win_imp += 1
            # награда за глобальное улучшение
            i_r = removals.index(rem)
            i_i = insertions.index(insertions[0])
            rem_scores[i_r] += accept_improve_reward
            ins_scores[i_i] += accept_improve_reward
            try:
                # touched routes count
                touched = sum(1 for cid in S_work if S_work.get(cid) != S_new.get(cid))
                dbg("IMPROVE Δ={}, best_obj={} <- {}, moved_mp={}, touched_routes={}, ops={}/{}", (cur_obj - new_obj), best_obj, cur_obj, len(removed_set), touched, last_rem_name, last_ins_name)
            except Exception:
                pass
        elif accepted_move:
            cur = deepcopy(S_new)
            cur_obj = int(new_obj)
            accepted += 1
            win_acc += 1
            i_r = removals.index(rem)
            i_i = insertions.index(insertions[0])
            rem_scores[i_r] += accept_worse_reward
            ins_scores[i_i] += accept_worse_reward
        else:
            i_r = removals.index(rem)
            i_i = insertions.index(insertions[0])
            rem_scores[i_r] += reject_reward
            ins_scores[i_i] += reject_reward
            win_rej += 1

        lahc_buf[iter_idx % L] = cur_obj
        iter_idx += 1

        # обновление весов раз в 16 итераций
        if iter_idx % 16 == 0:
            for i in range(len(rem_weights)):
                rem_weights[i] = (1 - reaction) * rem_weights[i] + reaction * max(1e-6, rem_scores[i])
                rem_scores[i] = 0.0
            for i in range(len(ins_weights)):
                ins_weights[i] = (1 - reaction) * ins_weights[i] + reaction * max(1e-6, ins_scores[i])
                ins_scores[i] = 0.0

        # периодический лог (окно)
        if iterations % WIN == 0:
            try:
                elapsed = int(time.time() - t_start_total)
                acc_rate = win_acc / max(1, WIN)
                dbg("iters={}, elapsed={}s, best_obj={}, cur_obj={}, acc_rate={:.2f}, removed={}, infeasible(no_pos={} lim={} emptyCR={}), last_ops={}/{}, assigned={} remaining={}",
                    iterations, elapsed, best_obj, cur_obj, acc_rate, total_removed, win_no_pos, win_lim_viol, win_empty_cr, last_rem_name, last_ins_name,
                    assigned_orders_count_fast(cur), max(0, TOTAL_ORDERS - assigned_orders_count_fast(cur)))
            except Exception:
                pass
            win_acc = 0
            win_rej = 0
            win_imp = 0
            win_no_pos = 0
            win_lim_viol = 0
            win_empty_cr = 0

    # Сбор лучшего решения
    S_fin = best if is_feasible(best) else cur if is_feasible(cur) else S

    # Если не удалось собрать допустимое — возврат stage4_out
    if not is_feasible(S_fin):
        try:
            dbg("WARNING: empty result — routes={}, assigned=0")
        except Exception:
            pass
        return stage4_out

    # Сборка выходных структур
    routes_orders: Dict[int, List[int]] = {}
    for cid, mp_seq in S_fin.items():
        core = route_orders_from_mp_seq(mp_seq)
        routes_orders[int(cid)] = core

    # Метрики
    total_orders = len(orders_json.get("Orders", []) or [])
    assigned_orders = sum(len(v) for v in routes_orders.values())
    remaining_orders = max(0, total_orders - assigned_orders)

    # times: distance-only per route via portals (travel-only)
    times = []
    for cid, mp_seq in S_fin.items():
        if not mp_seq:
            continue
        times.append(route_travel_time_portal(mp_seq))

    # routes: with warehouse
    routes = []
    for cid, core in routes_orders.items():
        if not core:
            continue
        routes.append({"courier_id": int(cid), "route": [int(W)] + [int(x) for x in core] + [int(W)]})

    # Статистика
    stats = {
        "iters": int(iterations),
        "accepted": int(accepted),
        "best_obj": int(best_obj),
        "cur_obj": int(cur_obj),
        "removed": int(total_removed),
        "seed": seed,
        "time_cap_used": float(max(0.0, time.time() - t_start_total)),
        "method": method,
    }

    try:
        elapsed_total = int(time.time() - t_start_total)
        dbg("finished: iters_total={}, time={}s, best_obj={}, assigned={}/{}, routes={}", iterations, elapsed_total, best_obj, assigned_orders, total_orders, sum(1 for v in routes_orders.values() if v))
        # T_c stats on final
        vals = []
        for cid, mp_seq in S_fin.items():
            vals.append(T_c_raw(cid, mp_seq))
        if vals:
            mins = min(vals)
            maxs = max(vals)
            avgs = int(sum(vals) / max(1, len(vals)))
            violators = sum(1 for v in vals if v > MAX_WORK_TIME)
            dbg("T_c stats: min={}s, max={}s, avg={}s, violators={}", mins, maxs, avgs, violators)
        # Post metric: sum(times) + 3000*remaining
        sum_times = sum(times)
        penalty = remaining_orders * 3000
        dbg("post_metric: sum_times={} + penalty({}*3000)={} => {}", sum_times, remaining_orders, penalty, sum_times + penalty)
        try:
            assigned_final = sum(len(route_orders_from_mp_seq(seq)) for seq in S_fin.values())
            dbg("final assigned_orders={} of {}", assigned_final, TOTAL_ORDERS)
        except Exception:
            pass
        if assigned_orders == 0:
            dbg("WARNING: empty result — routes={}, assigned=0")
    except Exception:
        pass

    return {
        "routes_orders": routes_orders,
        "routes": routes,
        "times": times,
        "assigned_orders": assigned_orders,
        "total_orders": total_orders,
        "remaining_orders": remaining_orders,
        "stats": stats,
    }


