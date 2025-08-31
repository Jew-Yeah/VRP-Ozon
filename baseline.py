# python baseline.py --orders ml_ozon_logistic_dataSetOrders.json --couriers ml_ozon_logistic_dataSetCouriers.json --durations_json ml_ozon_logistic_dataDurations.json --durations_db durations.sqlite --output solution.json

#!/usr/bin/env python3
import argparse
import json
import sqlite3
import time
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
import ijson
from tqdm import tqdm
import psutil

WAREHOUSE_ID = 0
MAX_WORK_TIME = 12 * 3600
PENALTY = 3000
COURIERS_TO_USE = 280
BATCH = 500000
# STAGE5_TIME_CAP_SEC = 60 # время работы этапа 5

def ram_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def build_sqlite_stream(durations_json, db_path, max_rows=0):
    size = Path(durations_json).stat().st_size
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=OFF;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA cache_size=-1048576;")
    cur.execute("DROP TABLE IF EXISTS dists;")
    cur.execute("CREATE TABLE dists (f INTEGER, t INTEGER, d INTEGER);")
    conn.commit()
    inserted = 0
    t0 = time.time()
    with Path(durations_json).open("rb") as f, tqdm(total=size, unit="B", unit_scale=True, desc="Build SQLite (read)") as pbar:
        parser = ijson.items(f, "item")
        batch = []
        last_pos = 0
        for rec in parser:
            batch.append((int(rec["from"]), int(rec["to"]), int(rec["dist"])))
            if len(batch) >= BATCH:
                cur.execute("BEGIN;")
                cur.executemany("INSERT INTO dists(f,t,d) VALUES(?,?,?)", batch)
                conn.commit()
                inserted += len(batch)
                batch.clear()
                now = f.tell()
                pbar.update(now - last_pos)
                last_pos = now
                if max_rows and inserted >= max_rows:
                    break
        if batch and (not max_rows or inserted < max_rows):
            need = max_rows - inserted if max_rows else len(batch)
            cur.execute("BEGIN;")
            cur.executemany("INSERT INTO dists(f,t,d) VALUES(?,?,?)", batch[:need])
            conn.commit()
            inserted += min(len(batch), need)
            now = f.tell()
            pbar.update(now - last_pos)
    cur.execute("CREATE INDEX idx_ft ON dists(f,t);")
    conn.commit()
    conn.close()
    tqdm.write(f"Inserted rows: {inserted}, RAM: {ram_mb():.1f} MB")
    tqdm.write(f"SQLite ready in {int(time.time()-t0)}s, RAM: {ram_mb():.1f} MB")

def connect_db(db_path):
    uri = f"file:{Path(db_path).as_posix()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=OFF;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-524288;")
    conn.execute("PRAGMA query_only=ON;")
    return conn

def preload_warehouse_edges(cur, relevant_ids):
    d0_to = {}
    for t,d in cur.execute("SELECT t,d FROM dists WHERE f=?;", (WAREHOUSE_ID,)):
        d0_to[int(t)] = int(d)
    d_from0 = {}
    def pair_lookup(a, b):
        row = cur.execute("SELECT d FROM dists WHERE f=? AND t=?;", (a, b)).fetchone()
        if row:
            return int(row[0])
        row2 = cur.execute("SELECT d FROM dists WHERE f=? AND t=?;", (b, a)).fetchone()
        if row2:
            return int(row2[0])
        return None
    from tqdm import tqdm as _tqdm  # локальный алиас, чтобы избежать конфликтов имён
    for a in _tqdm(relevant_ids, desc="Preload *->0 edges", unit="id"):
        if a == WAREHOUSE_ID:
            d_from0[a] = 0
            continue
        val = pair_lookup(a, WAREHOUSE_ID)
        if val is not None:
            d_from0[int(a)] = val
    # добиваем отсутствующие 0->id обратным ребром id->0, если есть
    for b in _tqdm(relevant_ids, desc="Preload 0->* edges", unit="id"):
        if b == WAREHOUSE_ID:
            d0_to[b] = 0
            continue
        if b not in d0_to:
            val = pair_lookup(WAREHOUSE_ID, b)
            if val is not None:
                d0_to[int(b)] = val
    return d0_to, d_from0

def preprocessing(args):
    """
    Препроцессинг и инфраструктура:
    - Загружает входные JSON (заказы, курьеры)
    - Ограничивает набор курьеров по COURIERS_TO_USE
    - Готовит/подключает SQLite c длительностями (build при отсутствии)
    - Предзагружает рёбра со складом для ускоренных fallback'ов
    Возвращает контекст с объектами и мапами для последующих шагов.
    """
    # Загрузка заказов
    print("Load Orders...")
    orders_json = load_json(args.orders)
    orders = {o["ID"]: o for o in orders_json["Orders"]}
    print(f"Orders: {len(orders)}, RAM: {ram_mb():.1f} MB")

    # Загрузка курьеров
    print("Load Couriers...")
    couriers_json = load_json(args.couriers)
    couriers_all = [c["ID"] for c in couriers_json["Couriers"]]
    couriers = couriers_all[:COURIERS_TO_USE]
    print(f"Couriers used: {len(couriers)}/{len(couriers_all)}, RAM: {ram_mb():.1f} MB")

    # Подготовка/подключение БД длительностей
    db_path = Path(args.durations_db)
    if not db_path.exists():
        print("Build durations SQLite...")
        build_sqlite_stream(args.durations_json, db_path, max_rows=args.build_rows)
        print("Build durations SQLite... done")
    else:
        print(f"Use existing SQLite: {db_path}")

    # Подключение к БД (только чтение)
    print("Connect DB...")
    conn = connect_db(db_path)
    cur = conn.cursor()

    # Предзагрузка рёбер со складом (out: 0->*, in: *->0)
    d0_to, d_from0 = preload_warehouse_edges(cur, orders.keys())
    print(f"Warehouse edges: out={len(d0_to)}, in={len(d_from0)}")

    # Группировка заказов по полигонам (MpId) — это часть препроцессинга, данность из данных
    print("Build polygons...")
    polygons = defaultdict(list)
    for oid, o in tqdm(orders.items(), desc="Group by MpId", unit="order"):
        polygons[o["MpId"]].append(oid)
    print(f"Polygons: {len(polygons)}, RAM: {ram_mb():.1f} MB")

    # Возвращаем контекст для ядра алгоритма
    return {
        "orders_json": orders_json,
        "orders": orders,
        "couriers_json": couriers_json,
        "couriers_all": couriers_all,
        "couriers": couriers,
        "conn": conn,
        "cur": cur,
        "d0_to": d0_to,
        "d_from0": d_from0,
        "polygons": polygons,
        "warehouse_id": WAREHOUSE_ID,
    }

def core_algorithm(ctx):
    start_time = time.time()
    FULL_TIME_CORE = 3500
    # Этап 0: дистанции и кэши
    from core.stage0 import run as stage0_run
    distances = stage0_run(ctx)
    try:
        if isinstance(distances, dict):
            # Пробросить функции дистанций в ctx для последующих этапов
            if "safe_dist" in distances:
                ctx["safe_dist"] = distances["safe_dist"]
            if "fast_dist" in distances:
                ctx["fast_dist"] = distances["fast_dist"]
            if "direct" in distances:
                ctx["direct"] = distances["direct"]
            print("Stage0: distances bound into ctx")
    except Exception:
        pass
    print("Stage 0 completed")

    # Этапы 1–4: заглушки (реализация по PDF будет добавлена поэтапно)
    from core.stage1 import run as stage1_run
    stage1_out = stage1_run(ctx)
    try:
        if isinstance(stage1_out, dict):
            ctx.update(stage1_out)
    except Exception:
        pass
    print("Stage 1 completed")
    from core.stage2 import run as stage2_run
    stage2_out = stage2_run(ctx, distances)
    try:
        if isinstance(stage2_out, dict):
            ctx.update(stage2_out)
    except Exception:
        pass
    print("Stage 2 completed")
    from core.stage3 import run as stage3_run
    stage3_out = stage3_run(ctx)
    # Обновим контекст данными Stage 3 (без сохранения на этом этапе)
    try:
        if isinstance(stage3_out, dict):
            ctx.update(stage3_out)
    except Exception:
        pass
    print("Stage 3 completed")
    from core.stage4 import run as stage4_run
    stage4_out = stage4_run(ctx, distances, stage2_out, stage3_out)
    print("Stage 4 completed")

    # Сохраняем промежуточное решение после Stage 4 в solution_test.json
    try:
        W = ctx.get("warehouse_id", 0)
        routes_orders = stage4_out.get("routes_orders", {}) if isinstance(stage4_out, dict) else {}
        routes = []
        for cid, core in routes_orders.items():
            if not core:
                continue
            route_with_wh = [int(W)] + [int(x) for x in core] + [int(W)]
            routes.append({"courier_id": int(cid), "route": route_with_wh})
        Path("solution_test.json").write_text(json.dumps({"routes": routes}, indent=2), encoding="utf-8")
        print("Saved solution_test.json (from Stage 4)")
    except Exception as e:
        print(f"Failed to save solution_test.json after Stage 4: {e}")

    STAGE5_TIME_CAP_SEC = max(0,3500-(time.time()-start_time))
    # Этап 5: ALNS — улучшение маршрутов на уровне MpId с жёстким лимитом 12ч
    try:
        from core.stage5 import run as stage5_run
        budget = STAGE5_TIME_CAP_SEC
        result = stage5_run(ctx, stage4_out, method="alns", time_cap_sec=budget, seed=ctx.get("seed", 42))
    except Exception as e:
        print(f"[Stage5] skipped due to error: {e}")
        result = stage4_out
    print("Stage 5 completed")

    # # Этап 6: GRASP — улучшение маршрутов на уровне MpId с жёстким лимитом 12ч
    # from core.stage6 import run as stage6_run
    # result = stage6_run(ctx, result)
    # print("Stage 6 completed")

    # # Этап 7: Балансировка маршрутов с поправкой по времени сервиса
    # from core.stage7 import run as stage7_run
    # result = stage7_run(ctx, result)
    # print("Stage 7 completed")

    return result

def postprocessing(ctx, algo_res, output_path):
    """
    Постпроцессинг и отчёт:
    - Формирует JSON сабмит с маршрутом каждого курьера [0, ..., 0]
    - Считает и печатает метрики (времена, штрафы, итоговый скор)
    - Записывает файл решения
    """
    orders = ctx["orders"]
    routes_orders = algo_res["routes_orders"]
    times = algo_res["times"]
    assigned_orders = algo_res["assigned_orders"]
    total_orders = algo_res["total_orders"]
    remaining_orders = algo_res["remaining_orders"]

    # Сборка финальных маршрутов с явным добавлением склада в начало/конец
    routes_with_wh = []
    cleaned = 0
    from tqdm import tqdm as _tqdm
    for cid, r in _tqdm(routes_orders.items(), desc="Postprocess: build routes", unit="route"):
        if not r:
            continue
        rr = [oid for oid in r if oid in orders and oid != WAREHOUSE_ID]
        cleaned += len(r) - len(rr)
        routes_with_wh.append({"courier_id": cid, "route": [WAREHOUSE_ID] + rr + [WAREHOUSE_ID]})

    # Финальный скор: сумма времени маршрутов + штрафы за неназначенные
    total_work_time = sum(times)
    penalty = remaining_orders * PENALTY
    final_score = total_work_time + penalty

    solution = {"routes": routes_with_wh}
    Path(output_path).write_text(json.dumps(solution, indent=2), encoding="utf-8")

    # Печать ключевых метрик
    print(f"Saved {output_path}")
    print(f"Assigned orders: {assigned_orders}/{total_orders}")
    print(f"Unassigned orders: {remaining_orders}")
    print(f"Cleaned entries removed: {cleaned}")
    if times:
        print(f"Min route time: {min(times)}s")
        print(f"Max route time: {max(times)}s")
        print(f"Avg route time: {int(sum(times)/len(times))}s")
    print(f"Total work time: {int(total_work_time)}s")
    print(f"Penalty: {int(penalty)}s")
    print(f"Final score: {int(final_score)}s")
    print(f"RAM: {ram_mb():.1f} MB")

def main():
    # Аргументы командной строки
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", required=True)
    ap.add_argument("--couriers", required=True)
    ap.add_argument("--durations_json", required=True)
    ap.add_argument("--durations_db", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--build_rows", type=int, default=0)
    args = ap.parse_args()

    t_all0 = time.time()

    # 1) Препроцессинг (I/O, БД, контекст)
    t_pre0 = time.time()
    ctx = preprocessing(args)
    t_pre1 = time.time()
    print(f"Preprocessing time: {int(t_pre1 - t_pre0)}s")

    # 2) Основной алгоритм (эвристика VRP)
    t_core0 = time.time()
    algo_res = core_algorithm(ctx)
    t_core1 = time.time()
    print(f"Core algorithm time: {int(t_core1 - t_core0)}s")

    # 3) Постпроцессинг (сабмит + отчёт)
    t_post0 = time.time()
    postprocessing(ctx, algo_res, args.output)
    t_post1 = time.time()
    print(f"Postprocessing time: {int(t_post1 - t_post0)}s")

    t_all1 = time.time()
    print(f"Total elapsed: {int(t_all1 - t_all0)}s")

if __name__ == "__main__":
    main()
