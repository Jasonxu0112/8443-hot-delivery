"""
RSM-8443 Hot Delivery – Part II
Single driver, minimize total distance, with average wait time <= W.

New elements vs Part I:
  - t[v]  : arrival time at node v (continuous, minutes from midnight)
  - avail[r]: food-ready time for restaurant r
  - Customer wait time for order k = t[customer_k] - avail[restaurant_k]
  - Constraint: average wait time <= W
  - Driver spends 5 min at each customer location (service time)
  - Speed: 40 km/h
"""

import pulp
import pandas as pd
from datetime import datetime

BASE      = "/sessions/dazzling-gifted-thompson/mnt/RSM8443/Case 3/"
DEPOT     = "__DEPOT__"                      # internal depot node (avoids name clashes)
DEPOT_LOC = "Downtown Toronto (Rosedale)"   # actual neighborhood name for distance lookup
SINK      = "__SINK__"
SPEED   = 40   # km/h
SERVICE =  5   # minutes per customer stop

# ── helpers ──────────────────────────────────────────────────────────────────

def parse_time(s):
    """'2022-04-02 7:27 PM'  →  minutes from midnight (e.g. 19*60+27 = 1167)"""
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
    # map internal DEPOT node to actual location name for distance lookup
    i = DEPOT_LOC if i == DEPOT else i
    j = DEPOT_LOC if j == DEPOT else j
    if i == j: return 0.0
    return d_map.get((i, j), 1e9)

def ttime(d_map, i, j, speed=SPEED):
    """Travel time in minutes."""
    return dist_km(d_map, i, j) / speed * 60.0

# ── solver ───────────────────────────────────────────────────────────────────

def solve_part2(orders_file, dist_map, W=120, use_max_wait=False, time_limit=120, verbose=False):
    """
    Solve single-driver routing with time constraints.

    Parameters
    ----------
    W            : limit on average (or max, if use_max_wait=True) customer wait time (minutes)
    use_max_wait : if True, constrain max wait time instead of average
    """
    orders_df = pd.read_csv(BASE + orders_file)
    orders = list(zip(
        orders_df["restaurant"],
        orders_df["customer"],
        [parse_time(a) for a in orders_df["estimated availability"]]
    ))
    n_orders = len(orders)

    restaurants  = [r for r, c, a in orders]
    customers    = [c for r, c, a in orders]
    avails       = [a for r, c, a in orders]
    customer_set = set(customers)

    # unique non-depot locations (DEPOT is internal "__DEPOT__", not a real neighborhood name,
    # so neighborhood "Rosedale" can appear in orders without conflict)
    seen, locs = set(), []
    for r, c, a in orders:
        for loc in [r, c]:
            if loc not in seen:
                seen.add(loc)
                locs.append(loc)

    all_nodes = [DEPOT] + locs + [SINK]
    real      = locs
    n         = len(real)
    M_pos     = n + 2
    M_time    = 24 * 60  # 1440 min upper bound on times

    arcs = [
        (i, j)
        for i in all_nodes
        for j in all_nodes
        if i != j
        and j != DEPOT
        and i != SINK
        and not (j == SINK and i not in customer_set)
    ]

    # ── model ────────────────────────────────────────────────────────────────
    prob = pulp.LpProblem("Part2", pulp.LpMinimize)

    x = {(i,j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for (i,j) in arcs}
    u = {v: pulp.LpVariable(f"u_{v}", lowBound=1, upBound=n) for v in real}
    t = {v: pulp.LpVariable(f"t_{v}", lowBound=0, upBound=M_time)
         for v in all_nodes if v != SINK}

    # objective: minimize distance
    prob += pulp.lpSum(
        (0.0 if j == SINK else dist_km(dist_map, i, j)) * x[i,j]
        for (i,j) in arcs
    )

    # ── flow constraints ─────────────────────────────────────────────────────
    prob += pulp.lpSum(x[i,j] for (i,j) in arcs if i == DEPOT) == 1
    prob += pulp.lpSum(x[i,j] for (i,j) in arcs if j == SINK)  == 1
    for v in real:
        prob += pulp.lpSum(x[i,j] for (i,j) in arcs if j == v) == 1
        prob += pulp.lpSum(x[i,j] for (i,j) in arcs if i == v) == 1

    # ── MTZ subtour elimination ───────────────────────────────────────────────
    for (i,j) in arcs:
        if i in real and j in real:
            prob += u[j] >= u[i] + 1 - M_pos * (1 - x[i,j])

    # ── pickup before delivery ────────────────────────────────────────────────
    for r, c, a in orders:
        prob += u[c] >= u[r] + 1

    # ── time propagation ──────────────────────────────────────────────────────
    # service time: 5 min at each customer location, 0 at restaurants/depot
    svc = {v: (SERVICE if v in customer_set else 0) for v in all_nodes if v != SINK}

    for (i,j) in arcs:
        if j != SINK and i in t and j in t:
            tt = ttime(dist_map, i, j)
            prob += t[j] >= t[i] + svc.get(i, 0) + tt - M_time * (1 - x[i,j])

    # ── food availability ─────────────────────────────────────────────────────
    for r, c, a in orders:
        if r in t:
            prob += t[r] >= a   # can't pick up before food is ready

    # ── wait time constraint ──────────────────────────────────────────────────
    if use_max_wait:
        # max wait time <= W  (alternative metric)
        for r, c, a in orders:
            prob += t[c] - a <= W
    else:
        # average wait time <= W
        prob += pulp.lpSum(t[c] - a for r, c, a in orders) <= n_orders * W

    # ── solve ─────────────────────────────────────────────────────────────────
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))

    if prob.status not in (1,):   # 1 = Optimal
        return None, None, None, pulp.LpStatus[prob.status]

    total_dist = pulp.value(prob.objective)
    wait_times = [pulp.value(t[c]) - a for r, c, a in orders]
    avg_wait   = sum(wait_times) / n_orders
    max_wait   = max(wait_times)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Instance : {orders_file}  |  W={W}  |  {'max' if use_max_wait else 'avg'} wait")
        print(f"Status   : {pulp.LpStatus[prob.status]}")
        print(f"Distance : {total_dist:.2f} km")
        print(f"Avg wait : {avg_wait:.1f} min   Max wait: {max_wait:.1f} min")

        # extract and print route
        route, current = [DEPOT], DEPOT
        while True:
            nxt = next((j for (i,j) in arcs if i == current and pulp.value(x.get((i,j),0)) > 0.5), None)
            if nxt is None or nxt == SINK:
                break
            route.append(nxt)
            current = nxt

        roles = {}
        for k, (r, c, a) in enumerate(orders):
            roles.setdefault(r, []).append(f"PICKUP  order {k+1} (ready {a}min)")
            roles.setdefault(c, []).append(f"DELIVER order {k+1} (wait {wait_times[k]:.1f}min)")

        print("\nRoute:")
        for step, node in enumerate(route):
            tag = ", ".join(roles.get(node, ["START"]))
            arr = f"arrive {pulp.value(t[node]):.0f}min" if node in t else ""
            print(f"  {step+1:2d}. [{tag}]  {node}  [{arr}]")

    return total_dist, avg_wait, max_wait, pulp.LpStatus[prob.status]


# ── Q1 & Q2: trade-off curve ──────────────────────────────────────────────────

def tradeoff_curve(orders_file, dist_map, W_values=None):
    if W_values is None:
        W_values = list(range(20, 301, 20))

    print(f"\n{'─'*55}")
    print(f"Trade-off curve: {orders_file}")
    print(f"{'W (min)':>10}  {'Distance (km)':>15}  {'Avg wait (min)':>15}  {'Status':>10}")
    print(f"{'─'*55}")

    results = []
    for W in W_values:
        d, avg, mx, status = solve_part2(orders_file, dist_map, W=W)
        if d is not None:
            print(f"{W:>10}  {d:>15.2f}  {avg:>15.1f}  {status:>10}")
            results.append((W, d, avg))
        else:
            print(f"{W:>10}  {'infeasible':>15}  {'—':>15}  {status:>10}")

    return results


# ── Q3: alternative metric (max wait) ────────────────────────────────────────

def compare_metrics(orders_file, dist_map, W=120):
    print(f"\n{'='*60}")
    print(f"Metric comparison on {orders_file}  (W={W})")

    d_avg, avg_avg, max_avg, s1 = solve_part2(orders_file, dist_map, W=W,
                                               use_max_wait=False, verbose=True)
    d_max, avg_max, max_max, s2 = solve_part2(orders_file, dist_map, W=W,
                                               use_max_wait=True,  verbose=True)

    print(f"\n  Average-wait model: dist={d_avg:.2f} km, avg={avg_avg:.1f}, max={max_avg:.1f}")
    print(f"  Max-wait model    : dist={d_max:.2f} km, avg={avg_max:.1f}, max={max_max:.1f}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dist_map = load_distances()

    # ── Part II verbose solutions ─────────────────────────────────────────────
    solve_part2("part2_ordersA.csv", dist_map, W=120, verbose=True)
    solve_part2("part2_ordersB.csv", dist_map, W=120, verbose=True)

    # ── Q2: trade-off curves ──────────────────────────────────────────────────
    tradeoff_curve("part2_ordersA.csv", dist_map, W_values=list(range(20, 301, 20)))
    tradeoff_curve("part2_ordersB.csv", dist_map, W_values=list(range(20, 301, 20)))

    # ── Q3: alternative metric ────────────────────────────────────────────────
    compare_metrics("part2_ordersA.csv", dist_map, W=120)
    compare_metrics("part2_ordersB.csv", dist_map, W=120)
