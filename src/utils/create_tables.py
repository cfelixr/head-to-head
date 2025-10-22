import polars as pl

def create_head_to_head_bets_lazy(bucket, key):
    # bets = pl.scan_parquet(f's3://s3-bucket-prod-lake-zero/bd_bets/bets/day=20250807/')
    bets = pl.scan_parquet(f's3://{bucket}/{key}')

    print("Creating head_to_head_bets", f's3://{bucket}/{key}')
    return (
        bets.filter(\
        (pl.col("SportId").is_in([1, 2, 5, 8, 9, 10, 15])) &\
        (pl.col("Status").is_in(["WON", "LOSE", "DRAW"]))
        ).select("MatchId", "Actual_Stake", "ActualRate", "Winlost", "SportId")
        .group_by("MatchId")
        .agg(
            (pl.col("Actual_Stake").cast(pl.Float64).fill_null(0) *
            pl.col("ActualRate").cast(pl.Float64).fill_null(0))
            .sum()
            .alias("TurnOver_SGD"),
            (((pl.col("Actual_Stake").cast(pl.Float64).fill_null(0) -
            pl.col("Winlost").cast(pl.Float64).fill_null(0)))*
            pl.col("ActualRate").cast(pl.Float64).fill_null(0))
            .sum()
            .alias("Winlost_SGD"),
        )
        .rename({"MatchId": "MatchId"})
    )

def create_head_to_head_match_lazy(bucket, key):
    # match = pl.scan_parquet(f"s3://s3-bucket-prod-lake-zero/bd_bets/match_result/day=20070101000000-20250222000000/")
    match = pl.scan_parquet(f's3://{bucket}/{key}')

    print("Creating head_to_head_match", f's3://{bucket}/{key}')

    return (
        match.filter((pl.col("sportId").is_in([1, 2, 5, 8, 9, 10, 15])))
        .group_by("matchId")
        .agg(pl.all().sort_by("modifiedOn").last())
        .select(
            pl.col("homeId").alias("HomeId"),
            pl.col("awayId").alias("AwayId"),
            pl.col("matchId").alias("MatchId"),
            pl.col("eventDate").alias("EventDate"),
            pl.col("kickOffTime").alias("KickOffTime"),
            pl.col("finalHomeScore").alias("FinalHomeScore"),
            pl.col("finalAwayScore").alias("FinalAwayScore"),
            pl.col("htHomeScore").alias("HtHomeScore"),
            pl.col("htAwayScore").alias("HtAwayScore"),
            pl.col("leagueId").alias("LeagueId"),
            pl.col("sportId").alias("SportId")
        )
    )
    
def create_head_to_head_odds_lazy(bucket, key):
    #odds = pl.read_parquet("s3://s3-bucket-prod-lake-zero/bd_bets/odds/day=20250221/")
    odds = pl.scan_parquet(f's3://{bucket}/{key}')
    print("Creating head_to_head_odds", f's3://{bucket}/{key}')

    mask_pre  = pl.col("liveIndicator") == False
    mask_live = pl.col("liveIndicator") == True

    mask_bettype_5 = (
        (pl.col("betType") == 5)
        & (pl.col("com1") == 0.01)
        & (pl.col("comX") == 0.01)
        & (pl.col("com2") == 0.01)
    )

    mask_bettype_11 = (
        (pl.col("betType") == 11)
        & (pl.col("odds1") == 0.01)
        & (pl.col("odds2") == 0.01)
    )

    return (
    odds
    .filter(
        pl.col("betType").is_in([5, 11]) &
        ~mask_bettype_5 &
        ~mask_bettype_11 
    ) 
    # Ordenamos globalmente por match y tiempo para que first()/last() respeten el tiempo
    .sort(["matchId", "modifiedOn"])
    .group_by("matchId")
    # ... mismo .sort() y .group_by() de antes
.agg(
    pl.col("betType").first().alias("BetType"),

    # IDs: si quieres 0 en vez de null y son numéricos, añade fill_null(0).
    # Si son strings, no lo hagas (o usa ""/"0").
    pl.col("oddsId").filter(mask_pre).first().alias("FirstOddsId"),
    pl.col("oddsId").filter(mask_live).last().alias("LastOddsId"),

    # Odds 1
    pl.col("odds1").filter(mask_pre).first().fill_null(0).alias("FirstOdds1"),
    pl.col("odds1").filter(mask_live).last().fill_null(0).alias("LastOdds1"),

    # Odds 2
    pl.col("odds2").filter(mask_pre).first().fill_null(0).alias("FirstOdds2"),
    pl.col("odds2").filter(mask_live).last().fill_null(0).alias("LastOdds2"),

    # Comisiones
    pl.col("com1").filter(mask_pre).first().fill_null(0).alias("FirstCom1"),
    pl.col("com2").filter(mask_pre).first().fill_null(0).alias("FirstCom2"),
    pl.col("comX").filter(mask_pre).first().fill_null(0).alias("FirstComx"),

    pl.col("com1").filter(mask_live).last().fill_null(0).alias("LastCom1"),
    pl.col("com2").filter(mask_live).last().fill_null(0).alias("LastCom2"),
    pl.col("comX").filter(mask_live).last().fill_null(0).alias("LastComx"),

    pl.col("modifiedOn").max().alias("ModifiedOn"),
)
    .rename({"matchId": "MatchId"})
    .select([
        "MatchId", "BetType", "FirstOddsId", "LastOddsId",
        "FirstOdds1", "LastOdds1", "FirstOdds2", "LastOdds2",
        "FirstCom1", "FirstCom2", "FirstComx",
        "LastCom1", "LastCom2", "LastComx",
        "ModifiedOn"
    ])
)