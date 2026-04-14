"""
RSM-8443 Hot Delivery – Part I
Single driver, minimize total distance.

Model: open-path TSP with pickup-before-delivery constraints.
- Driver starts at Rosedale (depot), ends at the last customer location.
- A dummy SINK node (zero-cost arcs from customers) handles the open path.
- MTZ constraints eliminate subtours.
- Precedence: u[restaurant] + 1 <= u[customer] for each order.
"""

import pulp
import pandas as pd

BASE = "/sessions/dazzling-gifted-thompson/mnt/RSM8443/Case 3/"
DEPOT = "Downtown Toronto (Rosedale)"
SINK  = "__SINK__"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_distances():
    df = pd.read_csv(BASE + "distances.csv")
    d = {}
    for _, row in df.iterrows():
        d[(row["origin"], row["destination"])] = row["distance"]
        d[(row["destination"], row["origin"])]  = row["distance"]
    return d

def dist(d_map, i, j):
    if i == j:
        return 0.0
    return d_map.get((i, j), 1e9)   # 1e9 = unreachable

# ── solver ───────────────────────────────────────────────────────────────────

def solve_part1(orders_file, dist_map):
    orders_df = pd.read_csv(BASE + orders_file)
    orders = list(zip(orders_df["restaurant"], orders_df["customer"]))
    n_orders = len(orders)

    restaurants = [r for r, c in orders]
    customers   = [c for r, c in orders]
    customer_set = set(customers)

    # unique non-depot locations (preserve insertion order)
    seen, locs = {DEPOT}, []
    for r, c in orders:
        for loc in [r, c]:
            if loc not in seen:
                seen.add(loc)
                locs.append(loc)

    all_nodes = [DEPOT] + locs + [SINK]
    real = locs          # non-depot, non-sink
    n    = len(real)
    M    = n + 2

    # valid arcs
    arcs = [
        (i, j)
        for i in all_nodes
        for j in all_nodes
        if i != j
        and j != DEPOT                              # no return to depot
        and i != SINK                               # can't leave sink
        and not (j == SINK and i not in customer_set)  # only customers end route
    ]

    # ── model ────────────────────────────────────────────────────────────────
    prob = pulp.LpProblem("Part1", pulp.LpMinimize)

    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for (i, j) in arcs}
    u = {v: pulp.LpVariable(f"u_{v}", lowBound=1, upBound=n) for v in real}

    # objective: total distance (arcs to SINK cost 0)
    prob += pulp.lpSum(
        (0.0 if j == SINK else dist(dist_map, i, j)) * x[i, j]
        for (i, j) in arcs
    )

    # leave depot once
    prob += pulp.lpSum(x[i, j] for (i, j) in arcs if i == DEPOT) == 1

    # enter sink once
    prob += pulp.lpSum(x[i, j] for (i, j) in arcs if j == SINK) == 1

    # each real node: exactly one in-arc and one out-arc
    for v in real:
        prob += pulp.lpSum(x[i, j] for (i, j) in arcs if j == v) == 1
        prob += pulp.lpSum(x[i, j] for (i, j) in arcs if i == v) == 1

    # MTZ subtour elimination (real-to-real arcs only)
    for (i, j) in arcs:
        if i in real and j in real:
            prob += u[j] >= u[i] + 1 - M * (1 - x[i, j])

    # pickup before delivery
    for r, c in orders:
        prob += u[c] >= u[r] + 1

    # ── solve ────────────────────────────────────────────────────────────────
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))

    print(f"\n{'='*60}")
    print(f"Instance : {orders_file}")
    print(f"Status   : {pulp.LpStatus[prob.status]}")
    print(f"Distance : {pulp.value(prob.objective):.2f} km")

    # extract route
    route, current = [DEPOT], DEPOT
    while True:
        nxt = next(
            (j for (i, j) in arcs if i == current and pulp.value(x[i, j]) > 0.5),
            None
        )
        if nxt is None or nxt == SINK:
            break
        route.append(nxt)
        current = nxt

    # node → role labels
    roles = {}
    for k, (r, c) in enumerate(orders):
        roles.setdefault(r, []).append(f"PICKUP  order {k+1}")
        roles.setdefault(c, []).append(f"DELIVER order {k+1}")

    print("\nRoute:")
    for step, node in enumerate(route):
        tag = ", ".join(roles.get(node, ["START"]))
        print(f"  {step+1:2d}. [{tag}]  {node}")

    return pulp.value(prob.objective), route


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dist_map = load_distances()
    solve_part1("part1_ordersA.csv", dist_map)
    solve_part1("part1_ordersB.csv", dist_map)
