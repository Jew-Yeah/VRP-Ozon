def run(ctx, distances, stage2_data, stage3_data):
    """
    Stage 4: пассивный этап. Не меняет решение Stage 3, просто адаптирует его к контракту постпроцессинга.
    """
    orders = ctx.get("orders", {})
    total_orders = len(orders)
    couriers = ctx.get("couriers", [])
    routes_stage3 = []
    if isinstance(stage3_data, dict):
        routes_stage3 = stage3_data.get("routes", []) or []

    # Преобразуем routes (со складами) в routes_orders для постпроцессинга
    routes_orders = {int(c): [] for c in couriers}
    assigned_orders = 0
    times = []
    W = ctx.get("warehouse_id", 0)

    def route_time_no_service(seq, safe_dist):
        if not seq:
            return 0
        t = 0
        for a, b in zip(seq[:-1], seq[1:]):
            t += int(safe_dist(int(a), int(b)))
        return t

    safe_dist = ctx.get("safe_dist") or (lambda a, b: 0)

    # Глобальный cleanup дублей: один заказ может быть только у одного курьера
    seen_orders = set()
    dup_removed = 0
    from tqdm import tqdm as _tqdm
    for r in _tqdm(routes_stage3, desc="Stage4: cleanup duplicates", unit="route"):
        cid = int(r.get("courier_id"))
        path = list(r.get("route", []))
        # убрать склад из концов, оставить только заказы
        core = [oid for oid in path if oid in orders and oid != W]
        cleaned = []
        for oid in core:
            if oid in seen_orders:
                dup_removed += 1
                continue
            seen_orders.add(oid)
            cleaned.append(oid)
        routes_orders[cid] = cleaned
        assigned_orders += len(cleaned)
        times.append(int(route_time_no_service([W] + cleaned + [W], safe_dist)))

    remaining_orders = max(0, total_orders - assigned_orders)

    try:
        print(f"Stage4: cleanup -> unique_assigned={assigned_orders}/{total_orders}, duplicates_removed={dup_removed}")
    except Exception:
        pass

    # Дополнительная попытка пристроить оставшиеся заказы:
    # 1) пробуем вставить в маршрут того же MpId
    # 2) если не получилось — в любой маршрут с минимальным дельта-временем
    # 3) если есть свободные курьеры — создаём одиночный маршрут [W, oid, W]
    if remaining_orders > 0:
        MAX_WORK_TIME = 12 * 3600
        # индекс: заказ -> MpId
        order_to_mp = {}
        for oid, o in orders.items():
            try:
                order_to_mp[int(oid)] = int(o.get("MpId"))
            except Exception:
                continue
        all_ids = set(int(k) for k in orders.keys())
        missing = [oid for oid in all_ids if oid not in seen_orders]
        # свободные курьеры (без маршрута)
        used_couriers = set(cid for cid, r in routes_orders.items() if r)
        free_couriers = [cid for cid in (int(c) for c in couriers) if cid not in used_couriers]

        rescued = 0
        for oid in _tqdm(missing, desc="Stage4: rescue unassigned", unit="order"):
            mp = order_to_mp.get(int(oid))
            # 1) кандидаты маршруты с тем же MpId (содержат хотя бы один заказ этого MpId)
            same_mp_routes = []
            for cid, core in routes_orders.items():
                if not core:
                    continue
                if any(order_to_mp.get(int(x)) == mp for x in core):
                    same_mp_routes.append((cid, core))
            candidate_routes = same_mp_routes if same_mp_routes else list(routes_orders.items())

            best = None
            best_cid = None
            best_route = None
            for cid, core in candidate_routes:
                base_t = route_time_no_service([W] + core + [W], safe_dist)
                # вставка на позицию с минимальным дельта-временем
                for pos in range(0, len(core) + 1):
                    cand = core[:pos] + [oid] + core[pos:]
                    t = route_time_no_service([W] + cand + [W], safe_dist)
                    if t <= MAX_WORK_TIME:
                        delta = t - base_t
                        if best is None or delta < best:
                            best = delta
                            best_cid = cid
                            best_route = cand
            if best_cid is not None:
                routes_orders[best_cid] = best_route
                seen_orders.add(oid)
                assigned_orders += 1
                rescued += 1
                continue
            # 3) создадим одиночный маршрут, если есть свободный курьер и вписывается
            if free_couriers:
                cid = free_couriers.pop(0)
                single_t = route_time_no_service([W, oid, W], safe_dist)
                if single_t <= MAX_WORK_TIME:
                    routes_orders[cid] = [oid]
                    seen_orders.add(oid)
                    assigned_orders += 1
                    rescued += 1
                else:
                    # вернуть курьера, если не удалось
                    free_couriers.insert(0, cid)
        # пересчёт
        remaining_orders = max(0, total_orders - assigned_orders)
        times = [int(route_time_no_service([W] + routes_orders[cid] + [W], safe_dist)) for cid in routes_orders if routes_orders[cid]]
        try:
            print(f"Stage4: extra insert -> rescued={rescued}, remaining={remaining_orders}")
        except Exception:
            pass

    return {
        "routes_orders": routes_orders,
        "assigned_orders": assigned_orders,
        "total_orders": total_orders,
        "remaining_orders": remaining_orders,
        "times": times,
    }


