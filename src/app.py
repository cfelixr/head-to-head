import json
import logging
import os
import urllib.parse
from typing import Dict, Any, List, Optional
import ctypes, ctypes.util

import gc

import boto3
from botocore.exceptions import ClientError
import polars as pl

import utils.create_tables as create_tables
import utils.upsert_on_head_to_head as upsert_on_head_to_head

# ----------------------------
# Config & Globals
# ----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

# Bucket de datos (lake) parametrizable por env var
DATA_BUCKET = os.environ.get("DATA_BUCKET", "s3-bucket-dev-lake-analytics")
HEAD_TO_HEAD_KEY = "bd_bets/head_to_head/head_to_head.parquet"
HEAD_TO_HEAD_S3_URI = f"s3://{DATA_BUCKET}/{HEAD_TO_HEAD_KEY}"


# ----------------------------
# Utilidades S3
# ----------------------------

def trim_heap():
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        libc.malloc_trim(0)  # devuelve páginas al SO si es posible
    except Exception:
        logger.error("No se pudo limpiar el heap")
        pass

def s3_object_exists(bucket: str, key: str) -> bool:
    """Devuelve True si el objeto S3 existe, False si no existe."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        # Otros errores (permisos, etc.) se re-lanzan
        raise


# ----------------------------
# Esquema y cargas
# ----------------------------
def head_to_head_schema() -> Dict[str, pl.DataType]:
    # OJO: si necesitas mantener una columna llamada exactamente "FristComx",
    # cámbiala aquí y en el resto del pipeline.
    return {
        "MatchId": pl.Int64,
        "BetType": pl.Int64,
        "HomeId": pl.Int64,
        "AwayId": pl.Int64,
        "HtHomeScore": pl.Int64,
        "HtAwayScore": pl.Int64,
        "FinalHomeScore": pl.Int64,
        "FinalAwayScore": pl.Int64,
        "EventDate": pl.String,
        "KickOffTime": pl.String,
        "LeagueId": pl.Int64,
        "SportId": pl.Int64,
        "FirstOddsId": pl.Int64,
        "LastOddsId": pl.Int64,
        "FirstOdds1": pl.Float64,
        "LastOdds1": pl.Float64,
        "FirstOdds2": pl.Float64,
        "LastOdds2": pl.Float64,
        "FirstCom1": pl.Float64,
        "FirstCom2": pl.Float64,
        "FirstComx": pl.Float64,   # <- corregido de "FristComx"
        "LastCom1": pl.Float64,
        "LastCom2": pl.Float64,
        "LastComx": pl.Float64,        
        "Winlost_SGD": pl.Float64,
        "TurnOver_SGD": pl.Float64,
        "ModifiedOn": pl.String,
    }


def empty_head_to_head_lf() -> pl.LazyFrame:
    """
    Crea un LazyFrame vacío con el esquema esperado.
    """
    return pl.LazyFrame(schema=head_to_head_schema())


def load_base_lazyframe() -> pl.LazyFrame:
    """
    Carga el LazyFrame base desde S3 si existe; en caso contrario, devuelve
    un LazyFrame vacío con el esquema correspondiente.
    """
    print("DATA_BUCKET",DATA_BUCKET)
    print("HEAD_TO_HEAD_KEY", HEAD_TO_HEAD_KEY)
    if s3_object_exists(DATA_BUCKET, HEAD_TO_HEAD_KEY):
        logger.info("Cargando base existente: %s", HEAD_TO_HEAD_S3_URI)
        return pl.scan_parquet(HEAD_TO_HEAD_S3_URI)
    else:
        logger.info("No existe base previa. Creando LazyFrame vacío con esquema.")
        return empty_head_to_head_lf()


# ----------------------------
# Lógica por registro
# ----------------------------
def process_record(bucket: str, key: str) -> Dict[str, Any]:
    """
    Procesa un único registro del evento S3.
    Devuelve un dict con el resultado del procesamiento.
    """
    # Decodificar key por si trae espacios/utf-8
    decoded_key = urllib.parse.unquote_plus(key, encoding="utf-8")
    origin_s3_uri = f"s3://{bucket}/{decoded_key}"
    logger.info("Nuevo objeto: %s", origin_s3_uri)

    # Cargar base (lazy)
    base_lf = load_base_lazyframe()

    # Construir el LazyFrame con los datos del nuevo objeto (definido por tus utils)
    # create_head_to_head_bets_lazy ya devuelve LazyFrame
    table_name = decoded_key.split("/")[1]
    if table_name == "bets":
        logger.info("Tabla: %s", table_name)
        new_rows_lf = create_tables.create_head_to_head_bets_lazy(bucket, decoded_key)
    elif table_name == "odds":
        logger.info("Tabla: %s", table_name)
        new_rows_lf = create_tables.create_head_to_head_odds_lazy(bucket, decoded_key)
    elif table_name == "match_result":
        logger.info("Tabla: %s", table_name)
        new_rows_lf = create_tables.create_head_to_head_match_lazy(bucket, decoded_key)
    else:
        logger.info("Not found table to update")
        return {"statusCode": 207, "body": "Not found table to update"}


    # upsert_on_head_to_head.upsert espera DataFrames "materializados", según tu código original
    base_df = base_lf.collect(streaming=True)
    new_rows_df = new_rows_lf.collect(streaming=True)

    logger.info(
        "Filas base: %d | Filas nuevas: %d",
        base_df.height,
        new_rows_df.height,
    )

    # Realizar upsert (devuelve DataFrame con el consolidado)
    if table_name == "bets":
        logger.info("Upsert tabla: %s", table_name)
        consolidated_df = upsert_on_head_to_head.upsert_bets(
            base_df, 
            new_rows_df, 
            key="MatchId"
            )
    elif table_name == "odds":
        logger.info("Upsert tabla: %s", table_name)
        consolidated_df = upsert_on_head_to_head.upsert_odds(
            base_df, 
            new_rows_df, 
            pk="MatchId", 
            preferir_reciente= ["BetType", "LastOddsId", "LastOdds1", "LastOdds2", "LastCom1", "LastCom2", "LastComx"], 
            preferir_antiguo=["FirstOddsId", "FirstOdds1", "FirstOdds2", "FirstCom1", "FirstCom2", "FirstComx"]
            )
    elif table_name == "match_result":
        logger.info("Upsert tabla: %s", table_name)
        consolidated_df = upsert_on_head_to_head.upsert_match(
            base_df, 
            new_rows_df, 
            key="MatchId"
            )
    else:
        logger.info("Not found table to upsert")
        return {"statusCode": 207, "body": "Not found table to upsert"}
    
    # consolidated_df = upsert_on_head_to_head.upsert(base_df, new_rows_df)

    # Persistir en parquet en S3 (misma ruta)
    consolidated_df.write_parquet(HEAD_TO_HEAD_S3_URI)
    logger.info("Escritura completada en: %s", HEAD_TO_HEAD_S3_URI)

    try:
        del base_df, new_rows_df, consolidated_df
    except NameError:
        pass
    gc.collect()
    trim_heap()

    return {
        "source": origin_s3_uri,
        "target": HEAD_TO_HEAD_S3_URI
    }


# ----------------------------
# Handler Lambda
# ----------------------------
def _extract_s3_from_sqs(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrae información S3 de un evento SQS que contiene eventos S3.
    """
    s3_data = event.get("s3", {})
    bucket = s3_data.get("bucket", {}).get("name")
    obj = s3_data.get("object", {}) or {}
    key = obj.get("key")
    return {
        "bucket": bucket,
        "key": key,
        "eventTime": event.get("eventTime"),
        "eventName": event.get("eventName"),
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Handler de AWS Lambda para eventos SQS que contienen eventos S3.
    Procesa todos los Records del evento de forma robusta con partial batch response.
    """
    __version__ = "1.1.1"

    logger.info(f"Version code: {__version__}")
    logger.info(json.dumps({"msg": "incoming_batch", "records": len(event.get("Records", []))}))
    
    batch_item_failures: List[Dict[str, str]] = []
    results = []

    for record in event.get("Records", []):
        message_id = record.get("messageId")
        body = record.get("body")
        
        try:
            # Parse SQS message body (contains S3 event)
            s3_event = json.loads(body) if isinstance(body, str) else body
        except Exception:
            logger.exception("Cannot parse SQS body as JSON")
            batch_item_failures.append({"itemIdentifier": message_id})
            continue

        # Handle S3 event nested in SQS message
        s3_records = s3_event.get("Records", [])
        if not s3_records:
            logger.warning("No S3 records found in SQS message")
            continue
            
        # Process first S3 record (usually only one per SQS message)
        s3_record = s3_records[0]
        s3_info = _extract_s3_from_sqs(s3_record)
        bucket = s3_info.get("bucket")
        key = s3_info.get("key")

        if not bucket or not key:
            logger.error("Missing bucket or key in S3 record")
            batch_item_failures.append({"itemIdentifier": message_id})
            continue

        try:
            # Usar la lógica existente de process_record
            res = process_record(bucket, key)
            results.append(res)
            logger.info("Successfully processed: %s/%s", bucket, key)
            
        except Exception as exc:
            logger.exception("Processing failed for message_id=%s key=%s", message_id, key)
            # Signal to SQS partial batch response to retry only this item
            batch_item_failures.append({"itemIdentifier": message_id})
        finally:
            # limpieza también entre mensajes del mismo batch
            gc.collect()
            trim_heap()

    # Return partial batch response format for SQS
    response = {"batchItemFailures": batch_item_failures}
    logger.info(json.dumps({"msg": "batch_complete", "failures": len(batch_item_failures), "processed": len(results)}))
    
    return response
