import polars as pl
from collections.abc import Mapping
from typing import Iterable, Optional, Sequence
from typing import Dict

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
    
    return empty_head_to_head_lf()

def ordenar_y_validar(
    df: pl.DataFrame,
    esquema: Dict[str, pl.datatypes.DataType],
    *,
    permitir_extras: bool = True,
) -> pl.DataFrame:
    """
    `esquema` define orden y tipo: {"MatchId": pl.Int64, "modifiedOn": pl.Utf8, ...}
    - Valida que existan y que los tipos coincidan.
    - Reordena columnas: primero las del esquema (en ese orden), luego las extras (si se permiten).
    """
    columnas = list(esquema.keys())

    # 1) Validar que existan
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas requeridas: {faltan}")

    # 2) Validar tipos (solo para las del esquema)
    malos = {c: (df.schema[c], esquema[c]) for c in columnas if df.schema.get(c) != esquema[c]}
    if malos:
        detalle = ", ".join(f"{c}: {act} != {esp}" for c, (act, esp) in malos.items())
        raise TypeError(f"Tipos no coinciden -> {detalle}")

    # 3) Manejar columnas extra
    extras = [c for c in df.columns if c not in columnas]
    if extras and not permitir_extras:
        raise ValueError(f"Columnas no esperadas: {extras}")

    # 4) Reordenar
    return df.select(columnas + extras)

def _ensure_df_from_schema(x):
    """Convierte `x` en DataFrame si es un schema (dict/pl.Schema)."""
    if isinstance(x, pl.DataFrame):
        return x
    # intenta construir desde schema (soporta dict o pl.Schema)
    try:
        return pl.DataFrame(schema=x)
    except Exception:
        pass
    # intenta convertir a dict y reintentar
    if isinstance(x, Mapping):
        return pl.DataFrame(schema=dict(x))
    raise TypeError(
        "base debe ser un pl.DataFrame o un schema compatible (dict o pl.Schema)"
    )

def upsert(base, delta: pl.DataFrame, key: str = "MatchId") -> pl.DataFrame:
    base = _ensure_df_from_schema(base)

    if key not in delta.columns:
        raise ValueError(f"'delta' debe tener la clave '{key}'")

    # si el schema de base no tiene la clave, la creamos (nula) con dtype del delta
    if key not in base.columns:
        key_dtype = delta.schema.get(key, pl.Utf8)
        base = base.with_columns(pl.lit(None, dtype=key_dtype).alias(key))

    # columnas en común (excepto la clave)
    overlap = [c for c in base.columns if c in delta.columns and c != key]

    # alinear dtypes del delta a los dtypes de base (incluye la clave)
    casts = []
    for c in [key] + overlap:
        tgt = base.schema[c]
        if delta.schema.get(c) != tgt:
            casts.append(pl.col(c).cast(tgt))
    if casts:
        delta = delta.with_columns(casts)

    # outer join: aparecen también las filas nuevas del delta

    j = base.join(delta, on=key, how="outer", suffix="_upd", coalesce=True)
    print("Join base and delta")

    # preferimos delta cuando no es nulo; si delta es nulo, se conserva base
    resolve = [pl.coalesce(pl.col(f"{c}_upd"), pl.col(c)).alias(c) for c in overlap]

    # columnas solo de base (incluye la clave y las que no solapan)
    base_only = [c for c in base.columns if c not in overlap]

    # columnas que están solo en delta (y por tanto entran sin sufijo)
    delta_only = [c for c in delta.columns if c not in overlap and c != key]

    # armamos salida manteniendo orden: base_first + resueltas + nuevas de delta
    out = j.select(
        [pl.col(c) for c in base_only] +
        resolve +
        [pl.col(c) for c in delta_only]
    )
    return out

def upsert_match(base, delta: pl.DataFrame, key: str = "MatchId") -> pl.DataFrame:
    base = _ensure_df_from_schema(base)

    if key not in delta.columns:
        raise ValueError(f"'delta' debe tener la clave '{key}'")

    # si el schema de base no tiene la clave, la creamos (nula) con dtype del delta
    if key not in base.columns:
        key_dtype = delta.schema.get(key, pl.Utf8)
        base = base.with_columns(pl.lit(None, dtype=key_dtype).alias(key))

    # columnas en común (excepto la clave)
    overlap = [c for c in base.columns if c in delta.columns and c != key]

    # alinear dtypes del delta a los dtypes de base (incluye la clave)
    casts = []
    for c in [key] + overlap:
        tgt = base.schema[c]
        if delta.schema.get(c) != tgt:
            casts.append(pl.col(c).cast(tgt))
    if casts:
        delta = delta.with_columns(casts)

    # outer join: aparecen también las filas nuevas del delta

    j = base.join(delta, on=key, how="outer", suffix="_upd", coalesce=True)
    print("Join base and delta")

    # preferimos delta cuando no es nulo; si delta es nulo, se conserva base
    resolve = [pl.coalesce(pl.col(f"{c}_upd"), pl.col(c)).alias(c) for c in overlap]

    # columnas solo de base (incluye la clave y las que no solapan)
    base_only = [c for c in base.columns if c not in overlap]

    # columnas que están solo en delta (y por tanto entran sin sufijo)
    delta_only = [c for c in delta.columns if c not in overlap and c != key]

    # armamos salida manteniendo orden: base_first + resueltas + nuevas de delta
    out = j.select(
        [pl.col(c) for c in base_only] +
        resolve +
        [pl.col(c) for c in delta_only]
    )
    out_val = ordenar_y_validar(out, head_to_head_schema())
    return out_val

def upsert_bets(base, delta: pl.DataFrame, key: str = "MatchId") -> pl.DataFrame:
    base = _ensure_df_from_schema(base)

    if key not in delta.columns:
        raise ValueError(f"'delta' debe tener la clave '{key}'")

    # si el schema de base no tiene la clave, la creamos (nula) con dtype del delta
    if key not in base.columns:
        key_dtype = delta.schema.get(key, pl.Utf8)
        base = base.with_columns(pl.lit(None, dtype=key_dtype).alias(key))

    # columnas en común (excepto la clave)
    overlap = [c for c in base.columns if c in delta.columns and c != key]

    base_only = [c for c in base.columns if c not in overlap]

    # alinear dtypes del delta a los dtypes de base (incluye la clave)
    casts = []
    for c in [key] + overlap:
        tgt = base.schema[c]
        if delta.schema.get(c) != tgt:
            casts.append(pl.col(c).cast(tgt))
    if casts:
        delta = delta.with_columns(casts)

    # outer join: aparecen también las filas nuevas del delta

    out = (
    base.join(delta, on=key, how="outer", suffix="_r")
      .with_columns([
          # suma numéricas (si falta en alguno, toma 0)
          (pl.coalesce([pl.col(c), pl.lit(0)]) +
           pl.coalesce([pl.col(f"{c}_r"), pl.lit(0)])).alias(c)
          for c in overlap
      ] + [
          # para no numéricas: toma la de df1 si existe, si no la de df2
          pl.coalesce([pl.col(c), pl.col(f"{c}_r")]).alias(c)
          for c in base_only
          if f"{c}_r" in base.join(delta, on=key, how="outer", suffix="_r").columns
      ])
      .drop([f"{c}_r" for c in overlap + base_only
             if f"{c}_r" in base.join(delta, on=key, how="outer", suffix="_r").columns])
    )
    out_val = ordenar_y_validar(out, head_to_head_schema())
    return out_val

def upsert_odds(
    df_base: pl.DataFrame,
    df_delta: pl.DataFrame,
    *,
    pk: str = "MatchId",
    ts_col: str = "ModifiedOn",
    preferir_reciente: Optional[Iterable[str]] = None,
    preferir_antiguo: Optional[Iterable[str]] = None,
) -> pl.DataFrame:
    preferir_reciente = set(preferir_reciente or [])
    preferir_antiguo = set(preferir_antiguo or [])

    # 1) Unificar columnas de ambos DFs
    todas = list({*df_base.columns, *df_delta.columns})
    if pk not in todas:
        raise ValueError(f"Falta la PK '{pk}' en al menos un DataFrame")
    if ts_col not in todas:
        raise ValueError(f"Falta la columna de timestamp '{ts_col}' en al menos un DataFrame")

    def _alinear(df: pl.DataFrame) -> pl.DataFrame:
        faltantes = [c for c in todas if c not in df.columns]
        if faltantes:
            df = df.with_columns([pl.lit(None).alias(c) for c in faltantes])
        return df.select(todas)

    a = _alinear(df_base)
    b = _alinear(df_delta)

    # 2) Apilar filas
    combo = pl.concat([a, b], how="diagonal")

    # 3) Agregaciones por columna: ordenamos por [es_nulo, modifiedOn]
    #    - es_nulo ascendente -> primero los NO nulos
    #    - modifiedOn asc/desc según regla de la columna
    aggs: Sequence[pl.Expr] = []
    for c in todas:
        if c == pk:
            continue
        # por defecto: preferir más reciente (desc = True)
        desc = True
        if c in preferir_antiguo:
            desc = False
        elif c in preferir_reciente:
            desc = True
        # ordenar sin filtrar para evitar el error de tamaños distintos
        expr = (
            pl.col(c)
            .sort_by(
                [pl.col(c).is_null(), pl.col(ts_col)],
                descending=[False, desc],
            )
            .first()
            .alias(c)
        )
        aggs.append(expr)

    # 4) Una fila por MatchId
    resultado = combo.group_by(pk, maintain_order=False).agg(*aggs)
    resultado_val = ordenar_y_validar(resultado, head_to_head_schema())
    return resultado_val