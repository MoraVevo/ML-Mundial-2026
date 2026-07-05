from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

from kinela.etl import build_api_football_etl, normalize_football_data
from kinela.fifa_ranking import (
    collect_fifa_ranking,
    collect_fifa_ranking_history,
    normalize_fifa_ranking,
    normalize_fifa_ranking_history,
)
from kinela.fouls_model import (
    export_neutral_fouls_matrix,
    predict_next_team_fouls,
    train_lightgbm_neutral_fouls,
    train_poisson_neutral_fouls,
)
from kinela.lightgbm_model import export_neutral_training_matrix, train_lightgbm
from kinela.model import export_clean_training_matrix, export_training_frame
from kinela.normalize import normalize_fotmob_world_cups, normalize_statsbomb_world_cup
from kinela.post_lineup_goals import train_post_lineup_goals, write_post_lineup_goals_matrix
from kinela.providers.api_football import ApiFootball
from kinela.providers.football_data import FootballData
from kinela.providers.fotmob import FotMobWorldCup
from kinela.providers.statsbomb import StatsBombOpenData
from kinela.worldcup_2026 import run_worldcup_simulation


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _collect_fixture_details(
    provider: ApiFootball,
    fixture_ids: list[int],
    *,
    refresh: bool = False,
    use_batch: bool = False,
) -> tuple[int, int, list[dict[str, str | int]]]:
    downloaded = 0
    api_calls = 0
    errors: list[dict[str, str | int]] = []
    if not use_batch:
        for fixture_id in fixture_ids:
            try:
                if provider.fixture_details_cached(fixture_id) and not refresh:
                    downloaded += 1
                    continue
                payload = provider.fixture_details(fixture_id, refresh=refresh)
                api_calls += 1
                payload_errors = json.dumps(payload.get("errors") or {}).lower()
                if any(
                    marker in payload_errors
                    for marker in ("request limit", "rate", "quota")
                ):
                    errors.append({"fixture_id": fixture_id, "error": payload_errors[:300]})
                    break
                if payload_errors:
                    errors.append({"fixture_id": fixture_id, "error": payload_errors[:300]})
                    continue
                downloaded += len(payload.get("response") or [])
            except Exception as exc:  # noqa: BLE001
                errors.append({"fixture_id": fixture_id, "error": str(exc)[:300]})
                if any(marker in str(exc).lower() for marker in ("429", "rate", "limit", "quota")):
                    break
        return downloaded, api_calls, errors

    for index in range(0, len(fixture_ids), 20):
        chunk = fixture_ids[index : index + 20]
        try:
            payload = provider.fixture_details_batch(chunk, refresh=refresh)
            api_calls += 1
            payload_errors = json.dumps(payload.get("errors") or {}).lower()
            if any(marker in payload_errors for marker in ("request limit", "rate", "quota")):
                errors.append({"fixture_id": chunk[0], "error": payload_errors[:300]})
                break
            if "ids parameter" in payload_errors:
                errors.append({"fixture_id": chunk[0], "error": payload_errors[:300]})
                break
            returned = {
                int((item.get("fixture") or {}).get("id"))
                for item in payload.get("response") or []
                if (item.get("fixture") or {}).get("id") is not None
            }
            downloaded += len(returned)
        except Exception as exc:  # noqa: BLE001
            errors.append({"fixture_id": chunk[0], "error": str(exc)[:300]})
            if any(marker in str(exc).lower() for marker in ("429", "rate", "limit", "quota")):
                break
    return downloaded, api_calls, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kinela")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Download provider data into the local cache")
    collect.add_argument(
        "source",
        choices=[
            "statsbomb-world-cup-2022",
            "api-football-team",
            "api-football-world-cup-teams",
            "api-football-date-fixtures",
            "api-football-missing-details",
            "api-football-standings",
            "football-data-team",
            "football-data-world-cup-teams",
            "football-data-bulk",
            "football-data-historical-scorers",
            "football-data-squad-player-matches",
            "fotmob-world-cups",
            "fifa-ranking",
            "fifa-ranking-history",
        ],
    )
    collect.add_argument("--refresh", action="store_true")
    collect.add_argument("--without-360", action="store_true")
    collect.add_argument("--team-id", type=int)
    collect.add_argument("--last", type=int, default=15)
    collect.add_argument(
        "--details",
        action="store_true",
        help="API-Football only: fetch event, lineup, team and player details per fixture",
    )
    collect.add_argument(
        "--batch-details",
        action="store_true",
        help=(
            "Use API-Football fixtures?ids= batch detail lookup. "
            "Do not use on Free plans; they do not allow the ids parameter."
        ),
    )
    collect.add_argument(
        "--detail-limit",
        type=int,
        default=0,
        help="Maximum new fixture-detail calls for the World Cup team collector",
    )
    collect.add_argument("--date", help="YYYY-MM-DD date for API-Football fixture lookup")
    collect.add_argument(
        "--from-date",
        help="FIFA ranking history only: earliest ranking match-window date to cache",
    )
    collect.add_argument("--league-id", type=int, help="API-Football league id filter")
    collect.add_argument("--season", type=int, help="API-Football season filter")
    collect.add_argument(
        "--from-season",
        type=int,
        default=2020,
        help="Historical football-data scorers: first season start year.",
    )
    collect.add_argument(
        "--to-season",
        type=int,
        default=2025,
        help="Historical football-data scorers: last season start year.",
    )

    normalize = subparsers.add_parser("normalize", help="Build analysis-ready CSV tables")
    normalize.add_argument(
        "source",
        choices=[
            "statsbomb-world-cup-2022",
            "api-football",
            "football-data",
            "fotmob-world-cups",
            "fifa-ranking",
            "fifa-ranking-history",
        ],
    )

    train = subparsers.add_parser("train", help="Train models from processed data")
    train.add_argument(
        "model",
        choices=[
            "lightgbm-neutral",
            "lightgbm-neutral-national",
            "lightgbm-neutral-fouls",
            "poisson-neutral-fouls",
            "post-lineup-goals",
        ],
    )

    export = subparsers.add_parser("export", help="Export model-ready datasets")
    export.add_argument(
        "artifact",
        choices=[
            "training-frame",
            "clean-training-matrix",
            "neutral-training-matrix",
            "post-lineup-goals-matrix",
            "neutral-fouls-matrix",
            "training-frame-national",
            "clean-training-matrix-national",
            "neutral-training-matrix-national",
        ],
    )

    predict = subparsers.add_parser("predict", help="Run direct model predictions")
    predict.add_argument("artifact", choices=["fouls-next"])
    predict.add_argument("--teams", nargs="+", required=True)
    predict.add_argument("--from-date", default="2026-06-13")
    predict.add_argument("--limit-per-team", type=int, default=1)
    predict.add_argument("--model", choices=["lightgbm", "poisson"], default="lightgbm")

    simulate = subparsers.add_parser("simulate", help="Run tournament simulations")
    simulate.add_argument("tournament", choices=["worldcup-2026"])
    simulate.add_argument("--runs", type=int, default=5000)
    simulate.add_argument("--seed", type=int, default=42)
    simulate.add_argument("--engine", choices=["lightgbm", "poisson"], default="lightgbm")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "collect":
        if args.source == "statsbomb-world-cup-2022":
            result = StatsBombOpenData(args.data_root).collect_world_cup_2022(
                include_360=not args.without_360,
                refresh=args.refresh,
            )
        elif args.source == "fotmob-world-cups":
            result = FotMobWorldCup(args.data_root).collect_world_cups(refresh=args.refresh)
        elif args.source == "api-football-team":
            if not args.team_id:
                raise SystemExit("--team-id is required")
            provider = ApiFootball(args.data_root)
            fixtures = []
            for season in (2023, 2024):
                payload = provider.fixtures_by_season(
                    team_id=args.team_id,
                    season=season,
                    refresh=args.refresh,
                )
                fixtures.extend(payload.get("response", []))
            fixtures.sort(key=lambda item: item.get("fixture", {}).get("date", ""), reverse=True)
            fixture_ids = [
                int(item["fixture"]["id"]) for item in fixtures[: args.last]
            ]
            downloaded = 0
            detail_api_calls = 0
            detail_errors: list[dict[str, str | int]] = []
            if args.details:
                downloaded, detail_api_calls, detail_errors = _collect_fixture_details(
                    provider,
                    fixture_ids,
                    refresh=args.refresh,
                    use_batch=args.batch_details,
                )
            result = {
                "provider": "api-football",
                "team_id": args.team_id,
                "fixtures": len(fixture_ids),
                "details_downloaded": downloaded,
                "fixture_detail_api_calls": detail_api_calls,
                "errors": detail_errors,
            }
        elif args.source == "api-football-world-cup-teams":
            provider = ApiFootball(args.data_root)
            catalog = provider.teams_by_league_season(
                league_id=1,
                season=2022,
                refresh=args.refresh,
            )
            teams = [
                item["team"]
                for item in catalog.get("response", [])
                if item.get("team", {}).get("id")
            ]
            fixture_ids: set[int] = set()
            current_seasons = (2023, 2024, 2025, 2026)
            for team in teams:
                team_fixtures: list[dict] = []
                for season in current_seasons:
                    payload = provider.fixtures_by_season(
                        team_id=int(team["id"]),
                        season=season,
                        refresh=args.refresh,
                    )
                    team_fixtures.extend(payload.get("response", []))
                team_fixtures.sort(
                    key=lambda item: item.get("fixture", {}).get("date", ""),
                    reverse=True,
                )
                fixture_ids.update(
                    int(item["fixture"]["id"])
                    for item in team_fixtures[: args.last]
                )

            pending = sorted(
                fixture_id
                for fixture_id in fixture_ids
                if not provider.fixture_details_cached(fixture_id)
            )
            limit = args.detail_limit or len(pending)
            downloaded, detail_api_calls, errors = _collect_fixture_details(
                provider,
                pending[:limit],
                refresh=args.refresh,
                use_batch=args.batch_details,
            )
            result = {
                "provider": "api-football",
                "scope": "2022-world-cup-teams",
                "teams": len(teams),
                "seasons": list(current_seasons),
                "unique_recent_fixtures": len(fixture_ids),
                "details_already_cached": len(fixture_ids) - len(pending),
                "details_downloaded_now": downloaded,
                "details_remaining": len(pending) - downloaded,
                "fixture_detail_api_calls": detail_api_calls,
                "errors": errors,
            }
        elif args.source == "api-football-date-fixtures":
            if not args.date:
                raise SystemExit("--date is required")
            provider = ApiFootball(args.data_root)
            payload = provider.fixtures_by_date(
                date=args.date,
                league_id=args.league_id,
                season=args.season,
                refresh=args.refresh,
            )
            fixtures = payload.get("response", [])
            fixture_ids = [
                int(item["fixture"]["id"])
                for item in fixtures
                if item.get("fixture", {}).get("id")
            ]
            downloaded = 0
            detail_api_calls = 0
            detail_errors: list[dict[str, str | int]] = []
            if args.details:
                limit = args.detail_limit or len(fixture_ids)
                downloaded, detail_api_calls, detail_errors = _collect_fixture_details(
                    provider,
                    fixture_ids[:limit],
                    refresh=args.refresh,
                    use_batch=args.batch_details,
                )
            result = {
                "provider": "api-football",
                "scope": "date-fixtures",
                "date": args.date,
                "league_id": args.league_id,
                "season": args.season,
                "fixtures": len(fixtures),
                "details_downloaded_now": downloaded,
                "fixture_detail_api_calls": detail_api_calls,
                "errors": detail_errors,
            }
        elif args.source == "api-football-missing-details":
            provider = ApiFootball(args.data_root)
            matches_path = args.data_root / "processed" / "api_football" / "matches.csv"
            if not matches_path.exists():
                raise SystemExit("Run `kinela normalize api-football` before collecting missing details")
            pending_rows: list[tuple[str, int]] = []
            with matches_path.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("competition_type") == "friendly":
                        continue
                    fixture_id = int(row["fixture_id"])
                    if not provider.fixture_details_cached(fixture_id):
                        pending_rows.append((row.get("date") or "", fixture_id))
            pending = [
                fixture_id
                for _, fixture_id in sorted(
                    pending_rows,
                    key=lambda item: (item[0], item[1]),
                    reverse=True,
                )
            ]
            limit = args.detail_limit or len(pending)
            downloaded, detail_api_calls, errors = _collect_fixture_details(
                provider,
                pending[:limit],
                refresh=args.refresh,
                use_batch=args.batch_details,
            )
            result = {
                "provider": "api-football",
                "scope": "non-friendly-known-fixture-details",
                "pending_before": len(pending),
                "details_downloaded_now": downloaded,
                "details_remaining": max(0, len(pending) - downloaded),
                "fixture_detail_api_calls": detail_api_calls,
                "errors": errors,
            }
        elif args.source == "api-football-standings":
            provider = ApiFootball(args.data_root)
            fixture_dir = args.data_root / "raw" / "api_football" / "fixtures"
            league_seasons: set[tuple[int, int]] = set()
            for path in fixture_dir.glob("team-*-season-*.json"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                for item in payload.get("response", []):
                    league = item.get("league") or {}
                    if league.get("standings") is True and league.get("id") and league.get("season"):
                        league_seasons.add((int(league["id"]), int(league["season"])))
            downloaded = 0
            errors: list[dict[str, str | int]] = []
            limit = args.detail_limit or len(league_seasons)
            for league_id, season in sorted(league_seasons)[:limit]:
                try:
                    provider.standings_by_league_season(
                        league_id=league_id,
                        season=season,
                        refresh=args.refresh,
                    )
                    downloaded += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append({"league_id": league_id, "season": season, "error": str(exc)[:300]})
                    if any(marker in str(exc).lower() for marker in ("429", "rate", "limit", "quota")):
                        break
            result = {
                "provider": "api-football",
                "scope": "standings-league-seasons",
                "league_seasons": len(league_seasons),
                "downloaded_now": downloaded,
                "errors": errors,
            }
        elif args.source == "football-data-team":
            if not args.team_id:
                raise SystemExit("--team-id is required")
            provider = FootballData(args.data_root)
            payload = provider.team_matches(
                team_id=args.team_id,
                limit=args.last,
                refresh=args.refresh,
            )
            result = {
                "provider": "football-data.org",
                "team_id": args.team_id,
                "matches": len(payload.get("matches", [])),
            }
        elif args.source == "fifa-ranking":
            result = collect_fifa_ranking(args.data_root, refresh=args.refresh)
        elif args.source == "fifa-ranking-history":
            result = collect_fifa_ranking_history(
                args.data_root,
                refresh=args.refresh,
                detail_limit=args.detail_limit,
                from_date=date.fromisoformat(args.from_date) if args.from_date else date(2022, 1, 1),
            )
        elif args.source == "football-data-squad-player-matches":
            provider = FootballData(args.data_root)
            teams_path = args.data_root / "raw" / "football_data" / "competitions" / "WC" / "teams.json"
            if not teams_path.exists():
                raise SystemExit("Run `kinela collect football-data-world-cup-teams` first")
            payload = json.loads(teams_path.read_text(encoding="utf-8"))
            player_ids: list[int] = []
            for team in payload.get("teams", []):
                for player in team.get("squad") or []:
                    if player.get("id"):
                        player_ids.append(int(player["id"]))
            unique_player_ids = sorted(set(player_ids))
            pending = [
                player_id
                for player_id in unique_player_ids
                if not (
                    args.data_root
                    / "raw"
                    / "football_data"
                    / "persons"
                    / f"{player_id}"
                    / "matches-last-20.json"
                ).exists()
            ]
            downloaded = 0
            errors: list[dict[str, str | int]] = []
            limit = args.detail_limit or len(pending)
            for player_id in pending[:limit]:
                try:
                    provider.person_matches(person_id=player_id, limit=20, refresh=args.refresh)
                    downloaded += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append({"player_id": player_id, "error": str(exc)[:300]})
                    if any(marker in str(exc).lower() for marker in ("429", "rate", "limit", "quota")):
                        break
            result = {
                "provider": "football-data.org",
                "scope": "world-cup-squad-player-matches",
                "squad_players": len(unique_player_ids),
                "pending_before": len(pending),
                "downloaded_now": downloaded,
                "remaining": max(0, len(pending) - downloaded),
                "errors": errors,
            }
        elif args.source == "football-data-historical-scorers":
            provider = FootballData(args.data_root)
            catalog = provider.competitions(refresh=False)
            competition_codes = sorted(
                {
                    str(item["code"])
                    for item in catalog.get("competitions", [])
                    if item.get("code")
                }
            )
            downloaded = 0
            errors: list[dict[str, str | int]] = []
            quota_reached = False
            for season in range(args.from_season, args.to_season + 1):
                if quota_reached:
                    break
                for code in competition_codes:
                    try:
                        provider.competition_scorers(
                            code,
                            season=season,
                            refresh=args.refresh,
                        )
                        downloaded += 1
                    except Exception as exc:  # noqa: BLE001
                        errors.append(
                            {
                                "competition_code": code,
                                "season": season,
                                "error": str(exc)[:300],
                            }
                        )
                        if any(
                            marker in str(exc).lower()
                            for marker in ("429", "rate", "limit", "quota")
                        ):
                            quota_reached = True
                            break
            result = {
                "provider": "football-data.org",
                "scope": "historical-competition-scorers",
                "competitions": len(competition_codes),
                "from_season": args.from_season,
                "to_season": args.to_season,
                "downloaded_or_cached": downloaded,
                "errors": errors,
            }
        else:
            provider = FootballData(args.data_root)
            if args.source == "football-data-world-cup-teams":
                catalog = provider.competition_teams("WC", refresh=args.refresh)
                teams = catalog.get("teams", [])
                match_ids: set[int] = set()
                for team in teams:
                    payload = provider.team_matches(
                        team_id=int(team["id"]),
                        limit=args.last,
                        refresh=args.refresh,
                    )
                    match_ids.update(
                        int(match["id"]) for match in payload.get("matches", [])
                    )
                result = {
                    "provider": "football-data.org",
                    "scope": "current-world-cup-teams",
                    "teams": len(teams),
                    "unique_recent_matches": len(match_ids),
                }
            else:
                catalog = provider.competitions(refresh=args.refresh)
                competitions = [
                    item["code"]
                    for item in catalog.get("competitions", [])
                    if item.get("code")
                ]
                summary = []
                for code in competitions:
                    matches = provider.competition_matches(code, refresh=args.refresh)
                    teams = provider.competition_teams(code, refresh=args.refresh)
                    scorers = provider.competition_scorers(code, refresh=args.refresh)
                    summary.append(
                        {
                            "code": code,
                            "matches": len(matches.get("matches", [])),
                            "teams": len(teams.get("teams", [])),
                            "scorers": len(scorers.get("scorers", [])),
                        }
                    )
                result = {
                    "provider": "football-data.org",
                    "scope": "bulk-accessible-competitions",
                    "competitions": len(competitions),
                    "summary": summary,
                }
    elif args.command == "normalize":
        if args.source == "statsbomb-world-cup-2022":
            result = normalize_statsbomb_world_cup(args.data_root)
        elif args.source == "fotmob-world-cups":
            result = normalize_fotmob_world_cups(args.data_root)
        elif args.source == "api-football":
            result = build_api_football_etl(args.data_root)
        elif args.source == "fifa-ranking":
            result = normalize_fifa_ranking(args.data_root)
        elif args.source == "fifa-ranking-history":
            result = normalize_fifa_ranking_history(args.data_root)
        else:
            result = normalize_football_data(args.data_root)
    elif args.command == "train":
        if args.model == "post-lineup-goals":
            result = train_post_lineup_goals(args.data_root)
        elif args.model == "poisson-neutral-fouls":
            result = train_poisson_neutral_fouls(args.data_root)
        elif args.model == "lightgbm-neutral-fouls":
            result = train_lightgbm_neutral_fouls(args.data_root)
        elif args.model in {"lightgbm-neutral", "lightgbm-neutral-national"}:
            training_rows = export_training_frame(
                args.data_root,
                combined=True,
                use_competition=True,
                use_stage=True,
                national_only=True,
            )
            _write_csv(
                args.data_root / "processed" / "combined" / "training_frame.csv",
                training_rows,
            )
            clean_rows = export_clean_training_matrix(
                args.data_root,
                combined=True,
                use_competition=True,
                use_stage=True,
                national_only=True,
            )
            _write_csv(
                args.data_root / "processed" / "combined" / "clean_training_matrix.csv",
                clean_rows,
            )
            neutral_rows = export_neutral_training_matrix(args.data_root)
            _write_csv(
                args.data_root / "processed" / "combined" / "neutral_training_matrix.csv",
                neutral_rows,
            )
            result = train_lightgbm(args.data_root)
        else:
            result = train_lightgbm(args.data_root)
    elif args.command == "export":
        explicitly_national = args.artifact.endswith("-national")
        national_default_artifacts = {
            "training-frame",
            "clean-training-matrix",
            "neutral-training-matrix",
        }
        national_only = (
            explicitly_national
            or args.artifact in national_default_artifacts
        )
        if args.artifact == "post-lineup-goals-matrix":
            result = write_post_lineup_goals_matrix(args.data_root)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if args.artifact == "neutral-fouls-matrix":
            rows = export_neutral_fouls_matrix(args.data_root)
            output = args.data_root / "processed" / "combined" / "neutral_fouls_matrix.csv"
            _write_csv(output, rows)
            result = {"rows": len(rows), "path": str(output)}
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if args.artifact in {"training-frame", "training-frame-national"}:
            rows = export_training_frame(
                args.data_root,
                combined=True,
                use_competition=True,
                use_stage=True,
                national_only=national_only,
            )
            name = "training_frame_national.csv" if explicitly_national else "training_frame.csv"
            output = args.data_root / "processed" / "combined" / name
        elif args.artifact in {"clean-training-matrix", "clean-training-matrix-national"}:
            rows = export_clean_training_matrix(
                args.data_root,
                combined=True,
                use_competition=True,
                use_stage=True,
                national_only=national_only,
            )
            name = (
                "clean_training_matrix_national.csv"
                if explicitly_national
                else "clean_training_matrix.csv"
            )
            output = args.data_root / "processed" / "combined" / name
        else:
            if national_only:
                rows = export_clean_training_matrix(
                    args.data_root,
                    combined=True,
                    use_competition=True,
                    use_stage=True,
                    national_only=True,
                )
                clean_output = args.data_root / "processed" / "combined" / "clean_training_matrix.csv"
                _write_csv(clean_output, rows)
            rows = export_neutral_training_matrix(args.data_root)
            name = (
                "neutral_training_matrix_national.csv"
                if explicitly_national
                else "neutral_training_matrix.csv"
            )
            output = args.data_root / "processed" / "combined" / name
        _write_csv(output, rows)
        result = {"rows": len(rows), "path": str(output)}
    elif args.command == "predict":
        if args.artifact == "fouls-next":
            result = predict_next_team_fouls(
                args.data_root,
                teams=args.teams,
                from_date=date.fromisoformat(args.from_date),
                limit_per_team=args.limit_per_team,
                model_name=args.model,
            )
        else:
            raise SystemExit(f"Unknown predict artifact: {args.artifact}")
    else:
        result = run_worldcup_simulation(
            args.data_root,
            simulations=args.runs,
            seed=args.seed,
            engine=args.engine,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
