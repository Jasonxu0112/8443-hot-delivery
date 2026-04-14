"""
RSM-8443 Hot Delivery – Part IV
Scalable heuristic for large instances.

Heuristic: Greedy Insertion + MIP Sub-routing
─────────────────────────────────────────────
Insight from Parts I-III:
  - Drivers close to a restaurant should handle that order (saves distance).
  - Each driver's sub-route can be solved exactly via Part II MIP because
    the number of orders per driver is small in practice.

Algorithm:
  1. Sort orders by availability time (earliest first).
  2. Greedily assign each order to the driver whose current "position"
     is closest to the restaurant (cheapest insertion).
  3. Update the driver's virtual position to the new customer location.
  4. Solve each driver's sub-route independently with the Part II MIP.
  5. Report total distance and global average wait time.

This runs in O(K*D) for assignment + MIP solve per driver (very fast for
small order counts per driver).
"""

import pulp
import pandas as pd
from datetime import datetime
import time as timer

BASE    = "/sessions/dazzling-gifted-thompson/mnt/RSM8443/Case 3/"
SERVICE = 5

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

# ── Part II sub-router (single driver, given fixed order set) ─────────────────

def route_driver(driver_loc, driver_speed, assigned_orders, dist_map, W, time_limit=120):
    """
    Optimally route a single driver over their assigned orders using MIP.
    Returns (distance, wait_list) or (None, None) if infeasible.
    assigned_orders: list of (restaurant_loc, customer_loc, avail_minutes)
    """
    if not assigned_orders:
        return 0.0, []

    DEPOT = "__DEPOT__"
    SINK  = "__SINK__"
    K     = len(assigned_orders)

    # Nodes: depot, R0..R_{K-1}, C0..C_{K-1}, sink
    def R(k): return f"R{k}"
    def C(k): return f"C{k}"

    rest_nodes = [R(k) for k in range(K)]
    cust_nodes = [C(k) for k in range(K)]
    real_nodes = rest_nodes + cust_nodes
    cust_set   = set(cust_nodes)

    def node_loc(v):
        if v == DEPOT: return driver_loc
        k = int(v[1:])
        return assigned_orders[k][0] if v.startswith("R") else assigned_orders[k][1]

    def adist(i, j):
        if j == SINK: return 0.0
        return dist_km(dist_map, node_loc(i), node_loc(j))

    def att(i, j):
        return adist(i, j) / driver_speed * 60.0

    nodes = [DEPOT] + real_nodes + [SINK]
    arcs  = [
        (i, j) for i in nodes for j in nodes
        if i != j and j != DEPOT and i != SINK
        and not (j == SINK and i not in cust_set)
    ]

    prob  = pulp.LpProblem("SubRoute", pulp.LpMinimize)
    M_pos = 2*K + 2; M_t = 24*60

    x = {(i,j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for (i,j) in arcs}
    u = {v: pulp.LpVariable(f"u_{v}", lowBound=1, upBound=2*K) for v in real_nodes}
    t = {v: pulp.LpVariable(f"t_{v}", lowBound=0, upBound=M_t)
         for v in [DEPOT] + real_nodes}

    prob += pulp.lpSum((0.0 if j==SINK else adist(i,j)) * x[i,j] for (i,j) in arcs)

    prob += pulp.lpSum(x[i,j] for (i,j) in arcs if i==DEPOT) == 1
    prob += pulp.lpSum(x[i,j] for (i,j) in arcs if j==SINK)  == 1
    for v in real_nodes:
        prob += pulp.lpSum(x[i,j] for (i,j) in arcs if j==v) == 1
        prob += pulp.lpSum(x[i,j] for (i,j) in arcs if i==v) == 1

    for (i,j) in arcs:
        if i in real_nodes and j in real_nodes:
            prob += u[j] >= u[i] + 1 - M_pos*(1 - x[i,j])

    for k in range(K):
        prob += u[C(k)] >= u[R(k)] + 1

    for (i,j) in arcs:
        if j != SINK and j in t:
            svc = SERVICE if i in cust_set else 0
            prob += t[j] >= t[i] + svc + att(i,j) - M_t*(1 - x[i,j])

    for k in range(K):
        a = assigned_orders[k][2]
        prob += t[R(k)] >= a

    # average wait constraint
    prob += pulp.lpSum(t[C(k)] - assigned_orders[k][2] for k in range(K)) <= W * K

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))

    if prob.status != 1:
        return None, None

    d_val  = pulp.value(prob.objective)
    waits  = [pulp.value(t[C(k)]) - assigned_orders[k][2] for k in range(K)]
    return d_val, waits


# ── greedy assignment heuristic ───────────────────────────────────────────────

def heuristic(orders_file, drivers_file, dist_map, W=120, verbose=False):
    orders_df  = pd.read_csv(BASE + orders_file)
    drivers_df = pd.read_csv(BASE + drivers_file)

    orders = list(zip(
        orders_df["restaurant"],
        orders_df["customer"],
        [parse_time(a) for a in orders_df["estimated availability"]]
    ))
    K = len(orders)
    D = len(drivers_df)

    d_locs   = list(drivers_df["start region"])
    d_speeds = list(drivers_df["velocity"])

    # ── Step 1: greedy assignment ─────────────────────────────────────────────
    # Sort by availability time (earliest first)
    sorted_k = sorted(range(K), key=lambda k: orders[k][2])

    virtual_pos   = list(d_locs)          # driver's estimated current location
    driver_orders = {drv: [] for drv in range(D)}

    for k in sorted_k:
        r_loc = orders[k][0]
        # pick driver with min distance from virtual pos to restaurant
        best = min(range(D), key=lambda d: dist_km(dist_map, virtual_pos[d], r_loc))
        driver_orders[best].append(k)
        virtual_pos[best] = orders[k][1]  # update virtual pos to customer loc

    # ── Step 2: solve each driver's sub-route with MIP ────────────────────────
    t0 = timer.time()
    total_dist = 0.0
    all_waits  = []

    for drv in range(D):
        assigned = [(orders[k][0], orders[k][1], orders[k][2]) for k in driver_orders[drv]]
        d_val, waits = route_driver(d_locs[drv], d_speeds[drv], assigned, dist_map, W)
        if d_val is None:
            print(f"  Driver {drv+1}: infeasible sub-route (W too tight?)")
            continue
        total_dist += d_val
        all_waits.extend(waits)

        if verbose:
            print(f"\n  Driver {drv+1} ({d_locs[drv]}, {d_speeds[drv]}km/h) | "
                  f"orders {[k+1 for k in driver_orders[drv]]} | dist={d_val:.2f} km")

    elapsed   = timer.time() - t0
    avg_wait  = sum(all_waits) / len(all_waits) if all_waits else 0
    max_wait  = max(all_waits) if all_waits else 0

    print(f"\n  Total distance : {total_dist:.2f} km")
    print(f"  Avg wait       : {avg_wait:.1f} min   Max wait: {max_wait:.1f} min")
    print(f"  Runtime        : {elapsed:.1f} s")

    return total_dist, avg_wait, max_wait


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dist_map = load_distances()

    # ── Evaluate heuristic on small instance (compare with Part III optimal) ──
    print("=" * 60)
    print("Small instance (part3_small.csv) – W=120")
    print("Part III optimal: 30.54 km, avg_wait=23.3 min")
    print("\nHeuristic:")
    heuristic("part3_small.csv", "part3_drivers.csv", dist_map, W=120, verbose=True)

    # ── Large instance ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Large instance (part4_large.csv) – W=120")
    heuristic("part4_large.csv", "part4_drivers.csv", dist_map, W=120, verbose=True)
