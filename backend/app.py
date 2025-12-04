from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

from inventory_model import simulate_portfolio

# ---------------------------------------------------------------------------
# PROJECT BASIC ROUTES
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_FOLDER = PROJECT_ROOT / "frontend"
DATA_DIR = PROJECT_ROOT / "data"
SCENARIOS_FILE = DATA_DIR / "scenarios.json"
REAL_SKUS_CSV = DATA_DIR / "real_skus.csv"

DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# FLASK CONFIGURATION + LOGGING
# ---------------------------------------------------------------------------

app = Flask(
    __name__,
    static_folder=str(FRONTEND_FOLDER),
    static_url_path="",  
)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("jit_backend")


# ---------------------------------------------------------------------------
# HELPERS GENERALES
# ---------------------------------------------------------------------------

def parse_float(value: Any, default: float) -> float:
    """Convierte a float o devuelve un valor por defecto si falla."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int) -> int:
    """Convierte a int o devuelve un valor por defecto si falla."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_service_level(value: Any, default: float = 0.95) -> float:
    """
    Convierte el nivel de servicio aceptando:

    - Valores en 0–1 (ej: 0.95)
    - Valores en 0–100 (ej: 95)

    y normaliza al rango [0.80, 0.999] para evitar extremos poco realistas.
    """
    raw = parse_float(value, default)
    # Si el usuario mete 95 en vez de 0.95
    if raw > 1.0:
        raw = raw / 100.0
    # Clamp del valor en un rango razonable
    raw = max(0.80, min(0.999, raw))
    return raw


def build_simulation_response(results: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
    """
    Estandariza la respuesta de simulate_portfolio para que el frontend
    siempre reciba el mismo formato, sin importar el origen de los SKUs.
    """
    return {
        "Scenario_Name": scenario_name,
        "Simulation_Days": results.get("Simulation_Days"),
        "Average_Fill_Rate": results.get("Average_Fill_Rate"),
        "Total_Cost": results.get("Total_Cost"),
        "Cost_per_Unit_Demand": results.get("Cost_per_Unit_Demand"),
        "SKUs": results.get("SKUs", []),
        "Insights": results.get("Insights", {}),
    }


def skus_from_csv_reader(reader: csv.DictReader) -> List[Dict[str, Any]]:
    """
    Convierte un csv.DictReader en la lista de SKUs que espera simulate_portfolio.

    Columnas esperadas (cabecera del CSV):
      SkuId, Name, AnnualDemand, DemandStdPct, LeadTimeDays, LeadTimeStdPct,
      UnitCost, HoldingCostPct, OrderCostFixed, StockoutCostPerUnit, PolicyType
    """
    skus: List[Dict[str, Any]] = []

    for idx, row in enumerate(reader, start=1):
        try:
            skus.append(
                {
                    "skuId": row.get("SkuId", idx),
                    "name": row.get("Name", f"SKU {idx}"),
                    "annualDemand": parse_float(row.get("AnnualDemand", 10000), 10000.0),
                    "demandStdPct": parse_float(row.get("DemandStdPct", 0.20), 0.20),
                    "leadTimeDays": parse_int(row.get("LeadTimeDays", 7), 7),
                    "leadTimeStdPct": parse_float(row.get("LeadTimeStdPct", 0.10), 0.10),
                    "unitCost": parse_float(row.get("UnitCost", 10.0), 10.0),
                    "holdingCostPct": parse_float(row.get("HoldingCostPct", 0.25), 0.25),
                    "orderCostFixed": parse_float(row.get("OrderCostFixed", 50.0), 50.0),
                    "stockoutCostPerUnit": parse_float(
                        row.get("StockoutCostPerUnit", 20.0), 20.0
                    ),
                    "policyType": (row.get("PolicyType") or "EOQ").upper(),
                }
            )
        except Exception as exc:
            logger.warning("Skipping invalid row %s: %s", idx, exc)
            continue

    return skus


# ---------------------------------------------------------------------------
# ENDPOINTS DE FRONTEND
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index() -> Any:
    """
    Sirve la Single-Page Application del simulador.
    Equivale a abrir frontend/index.html.
    """
    return app.send_static_file("index.html")


@app.route("/frontend/<path:filename>", methods=["GET"])
def frontend_static(filename: str) -> Any:
    """
    Sirve cualquier archivo estático dentro de /frontend (JS, imágenes, etc.).
    """
    return send_from_directory(str(FRONTEND_FOLDER), filename)


# ---------------------------------------------------------------------------
# 1) 🟢 RUN SIMULATION  ->  POST /api/simulate
# ---------------------------------------------------------------------------

@app.route("/api/simulate", methods=["POST"])
def api_simulate() -> Any:
    """
    Endpoint conectado al botón 🟢 "Run simulation".

    Espera un JSON como:
    {
      "scenarioName": "Q4 – High variability vs Min-Max",
      "simulationDays": 180,
      "serviceLevel": 0.95,         # o 95
      "skus": [ {...}, {...}, ... ] # SKUs configurados en la UI
    }
    """
    payload = request.get_json(force=True) or {}

    scenario_name = payload.get("scenarioName", "Unnamed scenario")
    skus = payload.get("skus", [])
    simulation_days = parse_int(payload.get("simulationDays", 365), 365)
    service_level = parse_service_level(payload.get("serviceLevel", 0.95))

    if not skus or not isinstance(skus, list):
        return jsonify({"error": "Payload must include a non-empty 'skus' array."}), 400

    logger.info(
        "[RUN SIMULATION] Scenario='%s' SKUs=%d Days=%d ServiceLevel=%.3f",
        scenario_name,
        len(skus),
        simulation_days,
        service_level,
    )

    try:
        results = simulate_portfolio(
            skus=skus,
            simulation_days=simulation_days,
            service_level=service_level,
        )
    except Exception as exc:
        logger.exception("Error during simulation")
        return jsonify({"error": f"Simulation error: {exc}"}), 500

    return jsonify(build_simulation_response(results, scenario_name))


# ---------------------------------------------------------------------------
# 2) 🔵 SIMULATE REAL SKUS  ->  POST /api/simulate_real
# ---------------------------------------------------------------------------

@app.route("/api/simulate_real", methods=["POST"])
def api_simulate_real() -> Any:
    """
    Endpoint conectado al botón 🔵 "Simulate real SKUs".

    No recibe SKUs desde la UI. En su lugar:

    - Lee el archivo data/real_skus.csv
    - Interpreta cada fila como un SKU real
    - Ejecuta simulate_portfolio con ese portafolio completo
    """
    if not REAL_SKUS_CSV.exists():
        return jsonify({"error": "real_skus.csv not found in /data."}), 404

    payload = request.get_json(force=True) or {}

    scenario_name = payload.get("scenarioName", "Real SKUs Scenario")
    simulation_days = parse_int(payload.get("simulationDays", 365), 365)
    service_level = parse_service_level(payload.get("serviceLevel", 0.95))

    logger.info(
        "[SIMULATE REAL SKUS] Scenario='%s' File='%s' Days=%d ServiceLevel=%.3f",
        scenario_name,
        REAL_SKUS_CSV,
        simulation_days,
        service_level,
    )

    with REAL_SKUS_CSV.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        skus = skus_from_csv_reader(reader)

    if not skus:
        return jsonify({"error": "real_skus.csv did not contain any valid rows."}), 400

    try:
        results = simulate_portfolio(
            skus=skus,
            simulation_days=simulation_days,
            service_level=service_level,
        )
    except Exception as exc:
        logger.exception("Error during simulation with real_skus.csv")
        return jsonify({"error": f"Simulation error: {exc}"}), 500

    return jsonify(build_simulation_response(results, scenario_name))


# ---------------------------------------------------------------------------
# 3) 🟣 RUN SIMULATION FROM FILE  ->  POST /api/upload_skus
# ---------------------------------------------------------------------------

@app.route("/api/upload_skus", methods=["POST"])
def api_upload_skus() -> Any:
    """
    Endpoint conectado al botón 🟣 "Run simulation from file".

    Espera un FormData (desde el navegador) con:

      file           -> archivo CSV (obligatorio)
      scenarioName   -> texto (opcional)
      simulationDays -> int (opcional)
      serviceLevel   -> float/int (opcional, 0–1 o 0–100)

    Permite que el usuario suba un CSV (por ejemplo desde su ERP) sin
    necesidad de copiarlo manualmente a data/real_skus.csv.
    """
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No file uploaded. Use form field name 'file'."}), 400

    scenario_name = request.form.get("scenarioName", "Uploaded SKUs Scenario")
    simulation_days = parse_int(request.form.get("simulationDays", 365), 365)
    service_level = parse_service_level(request.form.get("serviceLevel", 0.95))

    logger.info(
        "[UPLOAD SKUS] Scenario='%s' Days=%d ServiceLevel=%.3f",
        scenario_name,
        simulation_days,
        service_level,
    )

    try:
        content = file.read().decode("utf-8-sig")
        reader = csv.DictReader(StringIO(content))
        skus = skus_from_csv_reader(reader)

        if not skus:
            return jsonify({"error": "Uploaded CSV did not contain any valid rows."}), 400

        results = simulate_portfolio(
            skus=skus,
            simulation_days=simulation_days,
            service_level=service_level,
        )
    except Exception as exc:
        logger.exception("Error processing uploaded CSV")
        return jsonify({"error": f"Error processing file: {exc}"}), 500

    return jsonify(build_simulation_response(results, scenario_name))


# ---------------------------------------------------------------------------
# 4) ⬇ EXPORT CSV  ->  POST /api/export_csv
# ---------------------------------------------------------------------------

@app.route("/api/export_csv", methods=["POST"])
def api_export_csv() -> Any:
    """
    Endpoint conectado al botón ⬇ "Export CSV".

    El frontend le envía los resultados por SKU (SKUs simulados) y este
    endpoint devuelve un archivo CSV descargable.

    Body esperado:
      {
        "SKUs": [ {...}, {...}, ... ]
      }
    """
    payload = request.get_json(force=True) or {}
    skus = payload.get("SKUs") or payload.get("skus")

    if not skus:
        return jsonify({"error": "Payload must contain 'SKUs' with simulation results."}), 400

    fieldnames = [
        "SkuId",
        "Name",
        "Policy",
        "Total_Demand",
        "Filled_Demand",
        "Fill_Rate",
        "Avg_Inventory",
        "Stockout_Days",
        "Num_Orders",
        "Total_Cost",
        "Holding_Cost",
        "Ordering_Cost",
        "Stockout_Cost",
    ]

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for s in skus:
        row = {k: s.get(k, "") for k in fieldnames}
        writer.writerow(row)

    csv_data = output.getvalue().encode("utf-8")
    filename = f"inventory_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    headers = {
        "Content-Disposition": f"attachment; filename={filename}",
        "Content-Type": "text/csv; charset=utf-8",
    }

    logger.info("[EXPORT CSV] SKUs=%d Filename=%s", len(skus), filename)
    return Response(csv_data, status=200, headers=headers)


# ---------------------------------------------------------------------------
# ESCENARIOS GUARDADOS (para extensiones futuras en la UI)
# ---------------------------------------------------------------------------

@app.route("/api/scenarios", methods=["GET"])
def api_list_scenarios() -> Any:
    """Devuelve una lista de escenarios guardados en data/scenarios.json."""
    if not SCENARIOS_FILE.exists():
        return jsonify([])

    try:
        with SCENARIOS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return jsonify([])
        return jsonify(data)
    except Exception as exc:
        logger.warning("Error reading scenarios.json: %s", exc)
        return jsonify([])


@app.route("/api/scenarios", methods=["POST"])
def api_save_scenario() -> Any:
    """
    Guarda o actualiza un escenario en data/scenarios.json.

    No está conectado a un botón específico todavía, pero permite
    que en el futuro la UI pueda ofrecer "Save scenario / Load scenario".
    """
    payload = request.get_json(force=True) or {}
    name = payload.get("name") or payload.get("scenarioName")
    if not name:
        return jsonify({"error": "Scenario must have a 'name'."}), 400

    if SCENARIOS_FILE.exists():
        try:
            with SCENARIOS_FILE.open("r", encoding="utf-8") as f:
                scenarios = json.load(f)
            if not isinstance(scenarios, list):
                scenarios = []
        except Exception:
            scenarios = []
    else:
        scenarios = []

    now = datetime.utcnow().isoformat() + "Z"
    scenario_id = payload.get("id")

    if scenario_id is None:
        # Nuevo escenario → se asigna un ID incremental
        scenario_id = max((int(s.get("id", 0)) for s in scenarios), default=0) + 1
        scenario = {
            "id": scenario_id,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "data": payload.get("data") or payload,
        }
        scenarios.append(scenario)
        logger.info("[SCENARIOS] Created id=%s name='%s'", scenario_id, name)
    else:
        # Actualizar escenario existente
        updated = False
        for s in scenarios:
            if str(s.get("id")) == str(scenario_id):
                s["name"] = name
                s["updated_at"] = now
                s["data"] = payload.get("data") or payload
                updated = True
                logger.info("[SCENARIOS] Updated id=%s name='%s'", scenario_id, name)
                break
        if not updated:
            scenario = {
                "id": scenario_id,
                "name": name,
                "created_at": now,
                "updated_at": now,
                "data": payload.get("data") or payload,
            }
            scenarios.append(scenario)
            logger.info("[SCENARIOS] Created (custom id) id=%s name='%s'", scenario_id, name)

    try:
        with SCENARIOS_FILE.open("w", encoding="utf-8") as f:
            json.dump(scenarios, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.exception("Error writing scenarios.json")
        return jsonify({"error": f"Could not save scenario: {exc}"}), 500

    return jsonify({"status": "ok", "id": scenario_id})


@app.route("/api/scenarios/<scenario_id>", methods=["GET"])
def api_get_scenario(scenario_id: str) -> Any:
    """Devuelve un escenario concreto por ID."""
    if not SCENARIOS_FILE.exists():
        return jsonify({"error": "Scenario not found."}), 404

    try:
        with SCENARIOS_FILE.open("r", encoding="utf-8") as f:
            scenarios = json.load(f)
    except Exception as exc:
        logger.warning("Error reading scenarios.json: %s", exc)
        return jsonify({"error": "Scenario not found."}), 404

    for s in scenarios:
        if str(s.get("id")) == str(scenario_id):
            return jsonify(s)

    return jsonify({"error": "Scenario not found."}), 404


# ---------------------------------------------------------------------------
# HEALTHCHECK
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def api_health() -> Any:
    """
    Endpoint usado por el botón "API ✓" en el frontend.

    Si devuelve {"status": "ok"} significa que el backend está levantado
    y listo para responder simulaciones.
    """
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting JIT Inventory Simulator backend...")
    logger.info("BACKEND_DIR:     %s", BACKEND_DIR)
    logger.info("PROJECT_ROOT:    %s", PROJECT_ROOT)
    logger.info("FRONTEND_FOLDER: %s", FRONTEND_FOLDER)
    logger.info("DATA_DIR:        %s", DATA_DIR)
    app.run(debug=True, port=5000)
