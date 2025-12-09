from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import math
import random


@dataclass
class SkuConfig:
    """Configuración básica de un SKU para el simulador."""
    sku_id: str
    name: str
    annual_demand: float
    demand_std_pct: float = 0.25  # % de variación sobre la demanda diaria media
    lead_time_days: float = 7.0
    lead_time_std_pct: float = 0.25
    unit_cost: float = 10.0
    holding_cost_pct: float = 0.25  # % anual sobre el valor de inventario
    order_cost_fixed: float = 50.0
    stockout_cost_per_unit: float = 5.0
    policy_type: str = "EOQ"  # EOQ | ROP | MINMAX | PERIODIC
    review_period_days: int = 7  # sólo para PERIODIC
    initial_inventory: int = 0

    # Parámetros calculados de política:
    order_qty: int = 0
    reorder_point: float = 0.0
    min_inventory: float = 0.0
    max_inventory: float = 0.0
    target_inventory: float = 0.0  # para política periódica


@dataclass
class DailyState:
    day: int
    on_hand: int
    pipeline: int
    demand: int
    shipped: int
    lost_sales: int
    order_placed: int
    holding_cost: float
    ordering_cost: float
    stockout_cost: float


@dataclass
class SkuResult:
    sku_id: str
    name: str
    config: SkuConfig
    days: List[DailyState] = field(default_factory=list)
    total_demand: int = 0
    total_shipped: int = 0
    total_lost_sales: int = 0
    total_holding_cost: float = 0.0
    total_ordering_cost: float = 0.0
    total_stockout_cost: float = 0.0

    @property
    def fill_rate(self) -> float:
        if self.total_demand <= 0:
            return 1.0
        return self.total_shipped / self.total_demand

    @property
    def total_cost(self) -> float:
        return self.total_holding_cost + self.total_ordering_cost + self.total_stockout_cost

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skuId": self.sku_id,
            "name": self.name,
            "config": asdict(self.config),
            "kpis": {
                "totalDemand": self.total_demand,
                "totalShipped": self.total_shipped,
                "totalLostSales": self.total_lost_sales,
                "fillRate": self.fill_rate,
                "totalHoldingCost": self.total_holding_cost,
                "totalOrderingCost": self.total_ordering_cost,
                "totalStockoutCost": self.total_stockout_cost,
                "totalCost": self.total_cost,
            },
            "dailyStates": [asdict(d) for d in self.days],
        }


# Tabla simple de valores Z para nivel de servicio (aprox.)
Z_TABLE = [
    (0.80, 0.84),
    (0.85, 1.04),
    (0.90, 1.28),
    (0.95, 1.65),
    (0.97, 1.88),
    (0.98, 2.05),
    (0.99, 2.33),
]


def z_from_service(service_level: float) -> float:
    """Devuelve un valor Z aproximado para un nivel de servicio dado (0-1)."""
    service = max(0.80, min(0.999, service_level))
    Z_TABLE_sorted = sorted(Z_TABLE, key=lambda x: x[0])
    for i, (p, z) in enumerate(Z_TABLE_sorted):
        if service == p:
            return z
        if service < p:
            p0, z0 = Z_TABLE_sorted[i - 1]
            factor = (service - p0) / (p - p0)
            return z0 + factor * (z - z0)
    return Z_TABLE_sorted[-1][1]


def _draw_non_negative_normal(rng: random.Random, mean: float, std: float) -> float:
    if std <= 0:
        return max(0.0, mean)
    val = rng.gauss(mean, std)
    return max(0.0, val)


def compute_policy(cfg: SkuConfig, service_level: float) -> None:
    """Calcula parámetros de política (Q, R, s, S, etc.) en función del tipo de política."""
    daily_demand_mean = cfg.annual_demand / 365.0
    daily_demand_std = daily_demand_mean * max(0.0, cfg.demand_std_pct)

    lt_mean = max(0.1, cfg.lead_time_days)
    lt_std = lt_mean * max(0.0, cfg.lead_time_std_pct)

    demand_lt_mean = daily_demand_mean * lt_mean
    # Aproximación clásica de varianza en lead time
    demand_lt_std = math.sqrt(
        (daily_demand_std ** 2) * lt_mean + (daily_demand_mean ** 2) * (lt_std ** 2)
    )
    z = z_from_service(service_level)
    safety = z * demand_lt_std

    # Coste de mantener una unidad durante un año
    h_per_unit_year = cfg.unit_cost * cfg.holding_cost_pct

    if cfg.policy_type.upper() == "EOQ":
        # Política (Q, R)
        if h_per_unit_year <= 0:
            eoq = daily_demand_mean
        else:
            eoq = math.sqrt(
                max(1.0, 2.0 * cfg.annual_demand * cfg.order_cost_fixed / h_per_unit_year)
            )
        cfg.order_qty = max(1, int(round(eoq)))
        cfg.reorder_point = demand_lt_mean + safety
        cfg.min_inventory = cfg.reorder_point
        cfg.max_inventory = cfg.reorder_point + cfg.order_qty

    elif cfg.policy_type.upper() == "ROP":
        # Solo punto de reorden, Q fijo como EOQ
        if h_per_unit_year <= 0:
            eoq = daily_demand_mean
        else:
            eoq = math.sqrt(
                max(1.0, 2.0 * cfg.annual_demand * cfg.order_cost_fixed / h_per_unit_year)
            )
        cfg.order_qty = max(1, int(round(eoq)))
        cfg.reorder_point = demand_lt_mean + safety

    elif cfg.policy_type.upper() == "MINMAX":
        # Política (s, S)
        if h_per_unit_year <= 0:
            base = demand_lt_mean
        else:
            base = demand_lt_mean
        cfg.min_inventory = base + safety
        # S ~ s + un EOQ aproximado
        if h_per_unit_year <= 0:
            eoq = daily_demand_mean
        else:
            eoq = math.sqrt(
                max(1.0, 2.0 * cfg.annual_demand * cfg.order_cost_fixed / h_per_unit_year)
            )
        cfg.max_inventory = cfg.min_inventory + eoq

    elif cfg.policy_type.upper() == "PERIODIC":
        # Política de revisión periódica (R, S)
        P = max(1, cfg.review_period_days)
        target_horizon = P + lt_mean
        cfg.target_inventory = daily_demand_mean * target_horizon + safety
    else:
        # Fallback a EOQ estándar
        if h_per_unit_year <= 0:
            eoq = daily_demand_mean
        else:
            eoq = math.sqrt(
                max(1.0, 2.0 * cfg.annual_demand * cfg.order_cost_fixed / h_per_unit_year)
            )
        cfg.policy_type = "EOQ"
        cfg.order_qty = max(1, int(round(eoq)))
        cfg.reorder_point = demand_lt_mean + safety
        cfg.min_inventory = cfg.reorder_point
        cfg.max_inventory = cfg.reorder_point + cfg.order_qty


def simulate_single_sku(
    cfg: SkuConfig,
    days: int,
    service_level: float,
    rng_seed: Optional[int] = None,
) -> SkuResult:
    """Simula un SKU individual durante 'days' días."""
    rng = random.Random(rng_seed)
    compute_policy(cfg, service_level)

    daily_demand_mean = cfg.annual_demand / 365.0
    daily_demand_std = daily_demand_mean * max(0.0, cfg.demand_std_pct)

    lt_mean = max(0.1, cfg.lead_time_days)
    lt_std = lt_mean * max(0.0, cfg.lead_time_std_pct)

    h_per_unit_year = cfg.unit_cost * cfg.holding_cost_pct
    h_per_unit_day = h_per_unit_year / 365.0

    result = SkuResult(sku_id=cfg.sku_id, name=cfg.name, config=cfg)

    on_hand = max(0, int(cfg.initial_inventory))
    pipeline_orders: List[Dict[str, Any]] = []

    for day in range(days):
        # 1) Llegan pedidos que estaban en tránsito
        arriving_qty = 0
        still_in_pipeline = []
        for order in pipeline_orders:
            if order["arrival_day"] == day:
                arriving_qty += order["qty"]
            else:
                still_in_pipeline.append(order)
        pipeline_orders = still_in_pipeline
        on_hand += arriving_qty

        # 2) Realizamos la demanda del día
        demand = int(round(_draw_non_negative_normal(rng, daily_demand_mean, daily_demand_std)))
        shipped = min(on_hand, demand)
        lost_sales = max(0, demand - shipped)
        on_hand -= shipped

        # 3) Decisión de pedido según política
        pipeline_qty = sum(o["qty"] for o in pipeline_orders)
        inv_position = on_hand + pipeline_qty

        order_placed = 0
        if cfg.policy_type.upper() in ("EOQ", "ROP"):
            if inv_position <= cfg.reorder_point:
                order_placed = cfg.order_qty
        elif cfg.policy_type.upper() == "MINMAX":
            if inv_position <= cfg.min_inventory:
                order_placed = int(max(0.0, cfg.max_inventory - inv_position))
        elif cfg.policy_type.upper() == "PERIODIC":
            if (day % max(1, cfg.review_period_days)) == 0:
                order_placed = int(max(0.0, cfg.target_inventory - inv_position))

        if order_placed > 0:
            lt_real = max(1, int(round(_draw_non_negative_normal(rng, lt_mean, lt_std))))
            arrival_day = day + lt_real
            pipeline_orders.append({"qty": order_placed, "arrival_day": arrival_day})

        # 4) Costes
        holding_cost = on_hand * h_per_unit_day
        ordering_cost = cfg.order_cost_fixed if order_placed > 0 else 0.0
        stockout_cost = lost_sales * cfg.stockout_cost_per_unit

        # 5) Acumulados
        result.total_demand += demand
        result.total_shipped += shipped
        result.total_lost_sales += lost_sales
        result.total_holding_cost += holding_cost
        result.total_ordering_cost += ordering_cost
        result.total_stockout_cost += stockout_cost

        state = DailyState(
            day=day,
            on_hand=on_hand,
            pipeline=pipeline_qty,
            demand=demand,
            shipped=shipped,
            lost_sales=lost_sales,
            order_placed=order_placed,
            holding_cost=holding_cost,
            ordering_cost=ordering_cost,
            stockout_cost=stockout_cost,
        )
        result.days.append(state)

    return result


def simulate_portfolio(
    skus: List[Dict[str, Any]],
    days: int,
    service_level: float,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    """
    Simula un portafolio de SKUs. 'skus' es una lista de diccionarios
    con los campos necesarios para construir SkuConfig.
    """
    results: List[SkuResult] = []
    rng_base = rng_seed

    for i, sku in enumerate(skus):
        cfg = SkuConfig(
            sku_id=str(sku.get("skuId") or sku.get("sku_id") or f"SKU_{i+1}"),
            name=str(sku.get("name") or sku.get("sku_name") or f"SKU {i+1}"),
            annual_demand=float(sku.get("annualDemand", sku.get("annual_demand", 10000))),
            demand_std_pct=float(sku.get("demandStdPct", 0.25)),
            lead_time_days=float(sku.get("leadTimeDays", sku.get("lead_time_days", 7))),
            lead_time_std_pct=float(sku.get("leadTimeStdPct", 0.25)),
            unit_cost=float(sku.get("unitCost", sku.get("unit_cost", 10.0))),
            holding_cost_pct=float(sku.get("holdingCostPct", 0.25)),
            order_cost_fixed=float(sku.get("orderCostFixed", sku.get("order_cost_fixed", 50.0))),
            stockout_cost_per_unit=float(
                sku.get("stockoutCostPerUnit", sku.get("stockout_cost_per_unit", 5.0))
            ),
            policy_type=str(sku.get("policyType", sku.get("policy_type", "EOQ"))).upper(),
            review_period_days=int(sku.get("reviewPeriodDays", sku.get("review_period_days", 7))),
            initial_inventory=int(sku.get("initialInventory", sku.get("initial_inventory", 0))),
        )
        res = simulate_single_sku(cfg, days=days, service_level=service_level, rng_seed=rng_base + i)
        results.append(res)

    portfolio = {
        "skus": [r.to_dict() for r in results],
    }

    total_demand = sum(r.total_demand for r in results)
    total_shipped = sum(r.total_shipped for r in results)
    total_cost = sum(r.total_cost for r in results)

    portfolio["portfolioKpis"] = {
        "totalDemand": total_demand,
        "totalShipped": total_shipped,
        "fillRate": (total_shipped / total_demand) if total_demand > 0 else 1.0,
        "totalCost": total_cost,
        "avgCostPerUnit": (total_cost / total_shipped) if total_shipped > 0 else 0.0,
    }

    portfolio["insights"] = build_insights(results)

    return portfolio


def build_insights(results: List[SkuResult]) -> List[str]:
    """Genera mensajes cortos de insights sobre el portafolio."""
    if not results:
        return []

    insights: List[str] = []

    best_fill = max(results, key=lambda r: r.fill_rate)
    worst_fill = min(results, key=lambda r: r.fill_rate)
    most_expensive = max(results, key=lambda r: r.total_cost)

    insights.append(
        f"El mejor nivel de servicio lo tiene {best_fill.name} "
        f"({best_fill.sku_id}) con un fill rate de {best_fill.fill_rate:.1%}."
    )
    insights.append(
        f"El peor nivel de servicio lo tiene {worst_fill.name} "
        f"({worst_fill.sku_id}) con un fill rate de {worst_fill.fill_rate:.1%}."
    )
    insights.append(
        f"El SKU con mayor coste total es {most_expensive.name} "
        f"({most_expensive.sku_id}) con un coste de {most_expensive.total_cost:,.2f}."
    )

    return insights
