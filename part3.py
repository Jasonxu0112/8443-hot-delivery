"""
RSM-8443 Hot Delivery – Part III
Multiple drivers: assign orders to drivers AND route them.

Key change: use order-indexed nodes (R_k, C_k) instead of location names.
This avoids collisions when the same neighborhood appears multiple times.

Objective: minimize total distance across all drivers.
Constraint: global average wait time <= W.
"""

import pulp
import pandas as pd
from datetime import datetime

BASE    = "/sessions/dazzling-gifted-thompson/mnt/RSM8443/Case 3/"
SERVICE = 5   # minutes per customer stop

def parse_time(s):
    dt = datetime.strptime(s.strip(), "%Y-%m-%d %I:%M %p")
    return dt.hour * 60 + dt.minute

def load_distances():
    df = pd.read_csv(BASE + "distances.csv")
    d = {}
    for _, row in df.iterrows():
        d[(row["origin"], row["destination"])] = row["distance"]
        d[(row["destination"], row["origin"])]  = row["distance"]
    return d

def dist_km(d_map, i, j):
    if i == j: return 0.0
    return d_map.get((i, j), 1e9)

# ── solver ───────────────────────────────────────────────────────────────────

def solve_part3(orders_file, drivers_file, dist_map, W=120,
                time_limit=120, verbose=False, n_drivers=None):
    """
    n_drivers: use first n_drivers rows from drivers_file (for Q4 sensitivity).
    """
    orders_df  = pd.read_csv(BASE + orders_file)
    drivers_df = pd.read_csv(BASE + drivers_file)
    if n_drivers is not None:
        drivers_df = drivers_df.iloc[:n_drivers]

    orders = list(zip(
        orders_df["restaurant"],
        orders_df["customer"],
        [parse_time(a) for a in orders_df["estimated availability"]]
    ))
    K = len(orders)
    D = len(drivers_df)

    d_locs   = list(drivers_df["start region"])
    d_speeds = list(drivers_df["velocity"])

    # Node labels (order-indexed to avoid location collisions)
    def dep(d): return f"__DEP_{d}__"
    def snk(d): return f"__SNK_{d}__"
    def R(k):   return f"R{k}"
    def C(k):   return f"C{k}"

    rest_nodes = [R(k) for k in range(K)]
    cust_nodes = [C(k) for k in range(K)]
    real_nodes = rest_nodes + cust_nodes
    cust_set   = set(cust_nodes)

    # Physical location of a node
    def loc(drv, v):
        if v == dep(drv): return d_locs[drv]
        k = int(v[1:])
        return orders[k][0] if v.startswith("R") else orders[k][1]

    def arc_dist(drv, i, j):
        if j == snk(drv): return 0.0
        return dist_km(dist_map, loc(drv, i), loc(drv, j))

    def arc_tt(drv, i, j):
        return arc_dist(drv, i, j) / d_speeds[drv] * 60.0

    # Arcs per driver
    def build_arcs(drv):
        nodes = [dep(drv)] + real_nodes + [snk(drv)]
        return [
            (i, j) for i in nodes for j in nodes
            if i != j
            and j != dep(drv)
            and i != snk(drv)
            and not (j == snk(drv) and i not in cust_set)
        ]

    arcs_d = {drv: build_arcs(drv) for drv in range(D)}

    # ── model ────────────────────────────────────────────────────────────────
    prob = pulp.LpProblem("Part3", pulp.LpMinimize)

    M_pos  = 2 * K + 2
    M_time = 24 * 60

    y = {(drv, k): pulp.LpVariable(f"y_{drv}_{k}", cat="Binary")
         for drv in range(D) for k in range(K)}

    x = {(drv, i, j): pulp.LpVariable(f"x_{drv}_{i}_{j}", cat="Binary")
         for drv in range(D) for (i, j) in arcs_d[drv]}

    u = {(drv, v): pulp.LpVariable(f"u_{drv}_{v}", lowBound=1, upBound=2*K)
         for drv in range(D) for v in real_nodes}

    t = {(drv, v): pulp.LpVariable(f"t_{drv}_{v}", lowBound=0, upBound=M_time)
         for drv in range(D)
         for v in [dep(drv)] + real_nodes}

    # wait auxiliary: w[drv,k] = wait for order k by driver drv (0 if not assigned)
    w = {(drv, k): pulp.LpVariable(f"w_{drv}_{k}", lowBound=0, upBound=M_time)
         for drv in range(D) for k in range(K)}

    # ── objective ─────────────────────────────────────────────────────────────
    prob += pulp.lpSum(
        arc_dist(drv, i, j) * x[drv, i, j]
        for drv in range(D) for (i, j) in arcs_d[drv]
    )

    # ── each order assigned to exactly one driver ─────────────────────────────
    for k in range(K):
        prob += pulp.lpSum(y[drv, k] for drv in range(D)) == 1

    # ── per-driver constraints ────────────────────────────────────────────────
    for drv in range(D):
        arcs = arcs_d[drv]

        # leave depot once, arrive at sink once
        prob += pulp.lpSum(x[drv,i,j] for (i,j) in arcs if i == dep(drv)) == 1
        prob += pulp.lpSum(x[drv,i,j] for (i,j) in arcs if j == snk(drv)) == 1

        # flow for each order's restaurant and customer node
        for k in range(K):
            r, c = R(k), C(k)
            prob += pulp.lpSum(x[drv,i,j] for (i,j) in arcs if j == r) == y[drv,k]
            prob += pulp.lpSum(x[drv,i,j] for (i,j) in arcs if i == r) == y[drv,k]
            prob += pulp.lpSum(x[drv,i,j] for (i,j) in arcs if j == c) == y[drv,k]
            prob += pulp.lpSum(x[drv,i,j] for (i,j) in arcs if i == c) == y[drv,k]

        # MTZ subtour elimination
        for (i, j) in arcs:
            if i in real_nodes and j in real_nodes:
                prob += u[drv,j] >= u[drv,i] + 1 - M_pos*(1 - x[drv,i,j])

        # pickup before delivery (relaxed when not assigned)
        for k in range(K):
            prob += u[drv,C(k)] >= u[drv,R(k)] + 1 - M_pos*(1 - y[drv,k])

        # time propagation
        for (i, j) in arcs:
            if j != snk(drv) and (drv, j) in t:
                svc = SERVICE if i in cust_set else 0
                prob += (t[drv,j] >= t[drv,i] + svc + arc_tt(drv,i,j)
                         - M_time*(1 - x[drv,i,j]))

        # food availability (relaxed when not assigned)
        for k in range(K):
            a = orders[k][2]
            prob += t[drv, R(k)] >= a - M_time*(1 - y[drv,k])

        # wait time linearization: w[drv,k] = y[drv,k] * (t[drv,C_k] - a_k)
        for k in range(K):
            a = orders[k][2]
            prob += w[drv,k] >= t[drv,C(k)] - a - M_time*(1 - y[drv,k])
            prob += w[drv,k] <= M_time * y[drv,k]

    # ── global average wait time <= W ─────────────────────────────────────────
    prob += pulp.lpSum(w[drv,k] for drv in range(D) for k in range(K)) <= W * K

    # ── solve ─────────────────────────────────────────────────────────────────
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))

    status = pulp.LpStatus[prob.status]
    if prob.status != 1:
        return None, None, None, status

    total_dist = pulp.value(prob.objective)

    # compute actual waits
    waits = []
    for k in range(K):
        for drv in range(D):
            if pulp.value(y[drv,k]) > 0.5:
                a = orders[k][2]
                waits.append(pulp.value(t[drv, C(k)]) - a)
                break

    avg_wait = sum(waits) / K
    max_wait = max(waits)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Instance : {orders_file}  |  W={W}  |  {D} drivers")
        print(f"Status   : {status}")
        print(f"Total distance : {total_dist:.2f} km")
        print(f"Avg wait       : {avg_wait:.1f} min   Max wait: {max_wait:.1f} min")

        for drv in range(D):
            assigned = [k for k in range(K) if pulp.value(y[drv,k]) > 0.5]
            arcs = arcs_d[drv]

            if not assigned:
                print(f"\n  Driver {drv+1} ({d_locs[drv]}, {d_speeds[drv]}km/h): IDLE")
                continue

            # trace route
            route, cur = [dep(drv)], dep(drv)
            while True:
                nxt = next(
                    (j for (i,j) in arcs if i==cur and pulp.value(x.get((drv,i,j),0))>0.5),
                    None
                )
                if nxt is None or nxt == snk(drv): break
                route.append(nxt); cur = nxt

            route_dist = sum(arc_dist(drv, route[i], route[i+1]) for i in range(len(route)-1))
            print(f"\n  Driver {drv+1} ({d_locs[drv]}, {d_speeds[drv]}km/h) | "
                  f"orders {[k+1 for k in assigned]} | dist={route_dist:.2f} km")

            for v in route:
                if v == dep(drv):
                    arr_str = ""
                    print(f"    [START   ] {d_locs[drv]}")
                elif v.startswith("R"):
                    k = int(v[1:])
                    arr = pulp.value(t[drv,v])
                    print(f"    [PICKUP  {k+1}] {orders[k][0]}  (arrive={arr:.0f}min, ready={orders[k][2]}min)")
                elif v.startswith("C"):
                    k = int(v[1:])
                    arr  = pulp.value(t[drv,v])
                    wait = arr - orders[k][2]
                    print(f"    [DELIVER {k+1}] {orders[k][1]}  (arrive={arr:.0f}min, wait={wait:.1f}min)")

    return total_dist, avg_wait, max_wait, status


# ── Q4: sensitivity to number of drivers ─────────────────────────────────────

def driver_sensitivity(orders_file, drivers_file, dist_map, W=120, max_drivers=None):
    drivers_df = pd.read_csv(BASE + drivers_file)
    if max_drivers is None:
        max_drivers = len(drivers_df)

    print(f"\n{'─'*60}")
    print(f"Driver sensitivity: {orders_file}  (W={W})")
    print(f"{'Drivers':>8}  {'Distance (km)':>14}  {'Avg wait (min)':>15}  {'Max wait':>9}  {'Status':>10}")
    print(f"{'─'*60}")

    for nd in range(1, max_drivers + 1):
        d, avg, mx, status = solve_part3(orders_file, drivers_file, dist_map,
                                          W=W, n_drivers=nd, time_limit=180)
        if d is not None:
            print(f"{nd:>8}  {d:>14.2f}  {avg:>15.1f}  {mx:>9.1f}  {status:>10}")
        else:
            print(f"{nd:>8}  {'—':>14}  {'—':>15}  {'—':>9}  {status:>10}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dist_map = load_distances()

    # Full solution with all 3 drivers
    solve_part3("part3_small.csv", "part3_drivers.csv", dist_map,
                W=120, verbose=True)

    # Q4: how does solution change with fewer drivers?
    driver_sensitivity("part3_small.csv", "part3_drivers.csv", dist_map, W=120)
