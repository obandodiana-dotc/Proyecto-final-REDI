from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, request
from flask_cors import CORS

from inventory_model import simulate_portfolio

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SCENARIOS_FILE = DATA_DIR / "scenarios.json"
REAL_SKUS_FILE = DATA_DIR / "real_skus.csv"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
)
logger = logging.getLogger("jit_simulator")

app = Flask(__name__)
CORS(app)


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SCENARIOS_FILE.exists():
        SCENARIOS_FILE.write_text("[]", encoding="utf-8")


def _load_scenarios() -> List[Dict[str, Any]]:
    _ensure_data_dir()
    try:
        return json.loads(SCENARIOS_FILE.read_text(encoding="utf-8") or "[]")
    except Exception as exc:
        logger.error("Error leyendo scenarios.json: %s", exc)
        return []


def _save_scenarios(scenarios: List[Dict[str, Any]]) -> None:
    _ensure_data_dir()
    SCENARIOS_FILE.write_text(json.dumps(scenarios, indent=2), encoding="utf-8")


def _parse_float(data: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def _parse_int(data: Dict[str, Any], key: str, default: int) -> int:
    try:
        return int(data.get(key, default))
    except (TypeError, ValueError):
        return default


def _parse_service_level(value: Any, default: float = 0.95) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if val > 1.5:  # probablemente está en porcentaje
        val = val / 100.0
    return max(0.80, min(0.999, val))


def _normalize_skus(raw_skus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    skus: List[Dict[str, Any]] = []
    for i, sku in enumerate(raw_skus):
        skus.append(
            {
                "skuId": sku.get("skuId") or sku.get("sku_id") or f"SKU_{i+1}",
                "name": sku.get("name") or sku.get("sku_name") or f"SKU {i+1}",
                "annualDemand": _parse_float(sku, "annualDemand", sku.get("annual_demand", 10000)),
                "demandStdPct": _parse_float(sku, "demandStdPct", 0.25),
                "leadTimeDays": _parse_float(sku, "leadTimeDays", sku.get("lead_time_days", 7)),
                "leadTimeStdPct": _parse_float(sku, "leadTimeStdPct", 0.25),
                "unitCost": _parse_float(sku, "unitCost", sku.get("unit_cost", 10.0)),
                "holdingCostPct": _parse_float(sku, "holdingCostPct", 0.25),
                "orderCostFixed": _parse_float(sku, "orderCostFixed", sku.get("order_cost_fixed", 50.0)),
                "stockoutCostPerUnit": _parse_float(
                    sku, "stockoutCostPerUnit", sku.get("stockout_cost_per_unit", 5.0)
                ),
                "policyType": (sku.get("policyType") or sku.get("policy_type") or "EOQ").upper(),
                "reviewPeriodDays": _parse_int(sku, "reviewPeriodDays", sku.get("review_period_days", 7)),
                "initialInventory": _parse_int(sku, "initialInventory", sku.get("initial_inventory", 0)),
            }
        )
    return skus


def _parse_simulation_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Valida / normaliza el payload de simulación desde el frontend o Postman."""
    scenario_name = payload.get("scenarioName") or "Ad-hoc Scenario"
    simulation_days = _parse_int(payload, "simulationDays", 180)
    if simulation_days <= 0:
        raise ValueError("simulationDays debe ser > 0")

    service_level = _parse_service_level(payload.get("serviceLevel", 0.95))

    raw_skus = payload.get("skus") or payload.get("SKUs") or []
    if not isinstance(raw_skus, list) or len(raw_skus) == 0:
        raise ValueError("Debe proporcionar al menos un SKU en 'skus'.")

    skus = _normalize_skus(raw_skus)

    return {
        "scenarioName": scenario_name,
        "simulationDays": simulation_days,
        "serviceLevel": service_level,
        "skus": skus,
    }


def _build_response(
    scenario_name: str,
    simulation_days: int,
    service_level: float,
    simulation_result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "scenarioName": scenario_name,
        "settings": {
            "simulationDays": simulation_days,
            "serviceLevelRequested": service_level,
        },
        "portfolio": simulation_result.get("portfolioKpis", {}),
        "insights": simulation_result.get("insights", []),
        "skus": simulation_result.get("skus", []),
    }


@app.route("/api/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "ok"})


@app.route("/api/simulate", methods=["POST"])
def simulate() -> Any:
    try:
        payload = request.get_json(force=True, silent=False) or {}
        parsed = _parse_simulation_payload(payload)
        sim_result = simulate_portfolio(
            parsed["skus"],
            days=parsed["simulationDays"],
            service_level=parsed["serviceLevel"],
        )
        response = _build_response(
            scenario_name=parsed["scenarioName"],
            simulation_days=parsed["simulationDays"],
            service_level=parsed["ServiceLevel"],
            simulation_result=sim_result,
        )
        return jsonify(response)
    except ValueError as ve:
        logger.warning("Error de validación: %s", ve)
        return jsonify({"error": str(ve)}), 400
    except Exception:
        logger.exception("Error en /api/simulate")
        return jsonify({"error": "Error interno en la simulación"}), 500


@app.route("/api/simulate_real", methods=["POST"])
def simulate_real() -> Any:
    """
    Usa el archivo data/real_skus.csv como fuente de SKUs.
    El body puede incluir:
    {
      "scenarioName": "...",
      "simulationDays": 180,
      "serviceLevel": 0.95
    }
    """
    import csv

    if not REAL_SKUS_FILE.exists():
        return jsonify({"error": "No existe data/real_skus.csv"}), 400

    try:
        payload = request.get_json(force=True, silent=True) or {}
        scenario_name = payload.get("scenarioName") or "Real SKUs Scenario"
        simulation_days = _parse_int(payload, "simulationDays", 180)
        service_level = _parse_service_level(payload.get("serviceLevel", 0.95))

        with REAL_SKUS_FILE.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw_skus: List[Dict[str, Any]] = []
            for row in reader:
                raw_skus.append(
                    {
                        "skuId": row.get("sku_id") or row.get("skuId") or row.get("SKU") or row.get("id"),
                        "name": row.get("name") or row.get("sku_name") or row.get("SKU Name"),
                        "annualDemand": row.get("annual_demand") or row.get("annualDemand") or 10000,
                        "unitCost": row.get("unit_cost") or row.get("unitCost") or 10.0,
                        "leadTimeDays": row.get("lead_time_days") or row.get("leadTimeDays") or 7,
                        "policyType": row.get("policy_type") or row.get("policyType") or "EOQ",
                    }
                )

        if not raw_skus:
            raise ValueError("El archivo real_skus.csv no tiene filas.")

        skus = _normalize_skus(raw_skus)

        sim_result = simulate_portfolio(
            skus,
            days=simulation_days,
            service_level=service_level,
        )
        response = _build_response(
            scenario_name=scenario_name,
            simulation_days=simulation_days,
            serviceLevel=service_level,
            simulation_result=sim_result,
        )
        return jsonify(response)
    except ValueError as ve:
        logger.warning("Error de validación en /api/simulate_real: %s", ve)
        return jsonify({"error": str(ve)}), 400
    except Exception:
        logger.exception("Error en /api/simulate_real")
        return jsonify({"error": "Error interno en la simulación real"}), 500


@app.route("/api/scenarios", methods=["GET"])
def list_scenarios() -> Any:
    scenarios = _load_scenarios()
    return jsonify(scenarios)


@app.route("/api/scenarios", methods=["POST"])
def save_scenario() -> Any:
    try:
        payload = request.get_json(force=True, silent=False) or {}
        parsed = _parse_simulation_payload(payload)

        scenarios = _load_scenarios()
        next_id = (max((s.get("id", 0) for s in scenarios), default=0) or 0) + 1

        scenario_obj = {
            "id": next_id,
            "name": parsed["scenarioName"],
            "simulationDays": parsed["simulationDays"],
            "serviceLevel": parsed["serviceLevel"],
            "createdAt": payload.get("createdAt"),
            "updatedAt": payload.get("updatedAt"),
            "data": parsed,
        }
        scenarios.append(scenario_obj)
        _save_scenarios(scenarios)

        return jsonify({"message": "Scenario guardado", "id": next_id})
    except ValueError as ve:
        logger.warning("Error de validación en /api/scenarios: %s", ve)
        return jsonify({"error": str(ve)}), 400
    except Exception:
        logger.exception("Error guardando escenario")
        return jsonify({"error": "Error interno guardando escenario"}), 500


@app.route("/api/scenarios/<int:scenario_id>", methods=["GET"])
def get_scenario(scenario_id: int) -> Any:
    scenarios = _load_scenarios()
    for s in scenarios:
        if int(s.get("id", 0)) == scenario_id:
            return jsonify(s)
    return jsonify({"error": "Scenario no encontrado"}), 404


if __name__ == "__main__":
    _ensure_data_dir()
    app.run(host="0.0.0.0", port=5000, debug=True)
