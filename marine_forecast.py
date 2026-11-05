"""경량 해양 기상 추정 엔진.

복잡한 ML 모델 대신, Open-Meteo Marine API에서 불러온 "최근 3년 치 동월(同月) 데이터"로
아래 두 가지 선형 성분을 합쳐 미래 값을 빠르게 추정한다.

    예측값 = (연도별 동월 평균에 대한 선형 회귀 추세선을 목표 연도로 외삽한 값)
           + (월 안에서의 일별 기울기 × (목표 일자 - 월 중간 일자))

* 연도 추세(interannual trend): 각 연도의 해당 월 평균을 구한 뒤 `numpy.polyfit(deg=1)`로
  추세선을 그리고 목표 연도로 연장한다. 데이터가 1개 연도뿐이면 평균만 사용한다.
* 월내 기울기(intra-month slope): 연도별 평균을 뺀 일별 편차를 모아 일(day) 단위 회귀를 한다.
  예) 11월 수온은 월초보다 월말이 2~3℃ 낮으므로, 11월 28일 예측은 월평균보다 낮게 보정된다.
* 불확실성: 모델 잔차의 표준편차를 ±범위로 제공한다.

UI(Gradio)와 분리되어 있어 단독으로 테스트·재사용할 수 있다.
"""

from __future__ import annotations

import calendar
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Callable

import numpy as np
import requests

# --------------------------------------------------------------------------------------
# 설정값
# --------------------------------------------------------------------------------------
MARINE_API_URL = "https://marine-api.open-meteo.com/v1/marine"
TIMEZONE = "Asia/Seoul"
HISTORY_YEARS = 3          # 최근 몇 년 치 동월 데이터를 쓸지
MAX_LEAD_DAYS = 365        # 오늘로부터 며칠 뒤까지 예측을 허용할지
MIN_VALID_RATIO = 0.5      # 한 해의 해당 월에서 유효한 일자가 이 비율 이상이어야 사용
REQUEST_TIMEOUT = 10       # 초
ARCHIVE_LAG_DAYS = 2       # API 아카이브 반영 지연(일). 이 기간 안의 월은 '완료되지 않은 월'로 본다.

# 대표 해상 지점(연안 격자는 결측이 많아 약간 먼바다 좌표 사용)
LOCATIONS: dict[str, tuple[float, float]] = {
    "부산 앞바다": (35.00, 129.25),
    "제주 남쪽 먼바다": (32.90, 126.50),
    "포항 동해": (36.05, 129.70),
    "강릉 동해": (37.85, 129.15),
    "인천 서해": (37.20, 126.10),
    "여수 남해": (34.40, 127.80),
}


@dataclass(frozen=True)
class Variable:
    key: str            # API 변수명
    label: str          # 화면 표시명
    unit: str
    lower: float        # 물리적 하한(예측값 클리핑용)
    upper: float        # 물리적 상한


SST = Variable("sea_surface_temperature", "수온", "℃", -2.0, 35.0)
WAVE = Variable("wave_height", "파고", "m", 0.0, 15.0)
VARIABLES = (SST, WAVE)


# --------------------------------------------------------------------------------------
# 사용자 친화적 예외
# --------------------------------------------------------------------------------------
class ForecastError(Exception):
    """화면에 그대로 보여줄 수 있는 친절한 메시지를 담는 기본 예외."""

    title = "예측을 진행할 수 없어요"

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint


class InvalidDateError(ForecastError):
    title = "날짜를 다시 확인해 주세요"


class ApiError(ForecastError):
    title = "해양 데이터 서버에 연결하지 못했어요"


class DataUnavailableError(ForecastError):
    title = "참고할 과거 데이터가 없어요"


# --------------------------------------------------------------------------------------
# 결과 구조
# --------------------------------------------------------------------------------------
@dataclass
class VariableEstimate:
    variable: Variable
    value: float                      # 최종 예측값
    mean: float                       # 최근 동월 평균(단순 평균)
    trend_per_year: float             # 연도 추세 기울기(단위/년), 1개 연도면 0
    intra_month_per_day: float        # 월내 일별 기울기(단위/일)
    spread: float                     # ±불확실성(잔차 표준편차)
    yearly_means: dict[int, float]    # {연도: 동월 평균}
    method: str                       # "trend" | "mean"


@dataclass
class Forecast:
    target: date
    location: str
    lat: float
    lon: float
    years_used: list[int]
    estimates: dict[str, VariableEstimate]
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# 날짜 처리
# --------------------------------------------------------------------------------------
def parse_date(value) -> date:
    """Gradio DateTime 값(문자열/datetime/타임스탬프)을 date로 변환."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise InvalidDateError("날짜가 선택되지 않았어요.", "달력 아이콘을 눌러 예측할 날짜를 골라 주세요.")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value).date()
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    raise InvalidDateError(
        f"'{text}'는 날짜로 읽을 수 없어요.", "YYYY-MM-DD 형식(예: 2026-12-25)으로 입력해 주세요."
    )


def validate_target(target: date, today: date) -> None:
    if target < today:
        raise InvalidDateError(
            f"{target:%Y-%m-%d}은(는) 이미 지난 날짜예요.",
            f"오늘({today:%Y-%m-%d}) 이후의 미래 날짜를 선택해 주세요.",
        )
    if target > today + timedelta(days=MAX_LEAD_DAYS):
        raise InvalidDateError(
            f"{target:%Y-%m-%d}은(는) 너무 먼 미래예요.",
            f"최근 {HISTORY_YEARS}년 추세만으로는 정확도가 떨어져 "
            f"{MAX_LEAD_DAYS}일 이내({today + timedelta(days=MAX_LEAD_DAYS):%Y-%m-%d}까지)만 예측해요.",
        )


def reference_years(target: date, today: date, n: int = HISTORY_YEARS) -> list[int]:
    """목표 월과 같은 월 중, 이미 '끝난' 최근 n개 연도를 반환(오래된 순)."""
    cutoff = today - timedelta(days=ARCHIVE_LAG_DAYS)
    years: list[int] = []
    year = target.year
    while len(years) < n and year > target.year - n - 2:
        last_day = date(year, target.month, calendar.monthrange(year, target.month)[1])
        if last_day < cutoff:
            years.append(year)
        year -= 1
    return sorted(years)


# --------------------------------------------------------------------------------------
# API 호출
# --------------------------------------------------------------------------------------
Fetcher = Callable[[float, float, int, int], dict]


@lru_cache(maxsize=256)
def fetch_month(lat: float, lon: float, year: int, month: int) -> dict:
    """Open-Meteo Marine API에서 한 달 치 시간별 수온·파고를 받아온다(결과 캐시)."""
    start = date(year, month, 1)
    end = date(year, month, calendar.monthrange(year, month)[1])
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(v.key for v in VARIABLES),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": TIMEZONE,
    }
    try:
        resp = requests.get(MARINE_API_URL, params=params, timeout=REQUEST_TIMEOUT)
    except requests.Timeout as exc:
        raise ApiError(
            "해양 데이터 서버의 응답이 너무 늦어요.", "잠시 후 다시 시도해 주세요."
        ) from exc
    except requests.RequestException as exc:
        raise ApiError(
            "인터넷 연결 또는 데이터 서버(Open-Meteo)에 문제가 있어요.",
            "네트워크 상태를 확인한 뒤 다시 시도해 주세요.",
        ) from exc

    if resp.status_code == 429:
        raise ApiError("요청이 너무 많아 서버가 잠시 거절했어요.", "1분 정도 후에 다시 시도해 주세요.")
    if resp.status_code >= 500:
        raise ApiError(f"데이터 서버에 일시적인 오류가 있어요 (HTTP {resp.status_code}).", "잠시 후 다시 시도해 주세요.")
    try:
        payload = resp.json()
    except ValueError as exc:
        raise ApiError("서버 응답을 해석할 수 없어요.", "잠시 후 다시 시도해 주세요.") from exc
    if resp.status_code >= 400 or payload.get("error"):
        reason = payload.get("reason", f"HTTP {resp.status_code}")
        raise ApiError(f"데이터 요청이 거절됐어요: {reason}", "위치(위도·경도) 값을 확인해 주세요.")
    return payload


def daily_means(payload: dict, var_key: str) -> dict[int, float]:
    """시간별 응답을 {일(day): 일평균} 으로 집계. 결측(None)은 제외."""
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    values = hourly.get(var_key) or []
    buckets: dict[int, list[float]] = {}
    for t, v in zip(times, values):
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isnan(fv):
            continue
        buckets.setdefault(int(t[8:10]), []).append(fv)
    return {d: float(np.mean(vs)) for d, vs in buckets.items()}


# --------------------------------------------------------------------------------------
# 핵심 추정 함수 (순수 함수 — API와 무관하게 테스트 가능)
# --------------------------------------------------------------------------------------
def estimate_variable(
    variable: Variable,
    history: dict[int, dict[int, float]],
    target: date,
) -> VariableEstimate | None:
    """history = {연도: {일: 일평균}} 로부터 목표일 값을 추정. 쓸 데이터가 없으면 None."""
    days_in_target_month = calendar.monthrange(target.year, target.month)[1]
    usable: dict[int, dict[int, float]] = {}
    for year, by_day in history.items():
        n_days = calendar.monthrange(year, target.month)[1]
        if by_day and len(by_day) / n_days >= MIN_VALID_RATIO:
            usable[year] = by_day
    if not usable:
        return None

    years = np.array(sorted(usable), dtype=float)
    yearly_means = {int(y): float(np.mean(list(usable[int(y)].values()))) for y in years}
    means = np.array([yearly_means[int(y)] for y in years])

    # 1) 연도 추세선: 동월 평균 ~ 연도
    if len(years) >= 2:
        slope, intercept = np.polyfit(years, means, 1)
        base = slope * target.year + intercept
        method = "trend"
    else:
        slope, base, method = 0.0, float(means[0]), "mean"

    # 2) 월내 일별 기울기: (일 - 월중앙) ~ (값 - 해당 연도 평균)
    xs, ys = [], []
    for year, by_day in usable.items():
        mid = (calendar.monthrange(year, target.month)[1] + 1) / 2
        for day, val in by_day.items():
            xs.append(day - mid)
            ys.append(val - yearly_means[year])
    xs_arr, ys_arr = np.array(xs), np.array(ys)
    intra = float(np.polyfit(xs_arr, ys_arr, 1)[0]) if len(xs_arr) >= 3 and np.ptp(xs_arr) > 0 else 0.0

    target_offset = target.day - (days_in_target_month + 1) / 2
    value = float(np.clip(base + intra * target_offset, variable.lower, variable.upper))

    # 3) 불확실성: 일별 잔차 표준편차 (+ 연도 평균의 추세 잔차)
    daily_resid = ys_arr - intra * xs_arr
    spread = float(np.std(daily_resid)) if len(daily_resid) > 1 else 0.0
    if len(years) >= 3:
        trend_resid = means - (slope * years + (base - slope * target.year))
        spread = float(np.hypot(spread, np.std(trend_resid)))

    return VariableEstimate(
        variable=variable,
        value=value,
        mean=float(np.mean(means)),
        trend_per_year=float(slope),
        intra_month_per_day=intra,
        spread=spread,
        yearly_means=yearly_means,
        method=method,
    )


def forecast(
    target_input,
    location: str,
    lat: float | None = None,
    lon: float | None = None,
    *,
    today: date | None = None,
    fetcher: Fetcher = fetch_month,
) -> Forecast:
    """날짜·위치를 받아 수온/파고를 추정. 실패 시 ForecastError 하위 예외를 던진다."""
    today = today or date.today()
    target = parse_date(target_input)
    validate_target(target, today)

    if lat is None or lon is None:
        if location not in LOCATIONS:
            raise InvalidDateError("위치를 선택해 주세요.", "목록에서 해역을 고르거나 위도·경도를 직접 입력해 주세요.")
        lat, lon = LOCATIONS[location]
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise InvalidDateError("위도·경도 범위가 올바르지 않아요.", "위도는 -90~90, 경도는 -180~180 사이로 입력해 주세요.")

    years = reference_years(target, today)
    if not years:
        raise DataUnavailableError(
            f"{target.month}월의 과거 데이터를 찾을 수 없어요.", "다른 날짜를 선택해 주세요."
        )

    # 연도별 호출을 병렬로 — 3번의 왕복을 1번 시간에 끝낸다.
    payloads: dict[int, dict] = {}
    errors: list[ApiError] = []
    with ThreadPoolExecutor(max_workers=len(years)) as pool:
        futures = {y: pool.submit(fetcher, lat, lon, y, target.month) for y in years}
        for y, fut in futures.items():
            try:
                payloads[y] = fut.result()
            except ApiError as exc:
                errors.append(exc)
    if not payloads:
        raise errors[0] if errors else ApiError("데이터를 불러오지 못했어요.", "잠시 후 다시 시도해 주세요.")

    notes: list[str] = []
    if errors:
        notes.append(f"일부 연도({len(errors)}개) 데이터를 불러오지 못해 나머지 연도로만 추정했어요.")

    estimates: dict[str, VariableEstimate] = {}
    for var in VARIABLES:
        history = {y: daily_means(p, var.key) for y, p in payloads.items()}
        est = estimate_variable(var, history, target)
        if est is not None:
            estimates[var.key] = est
        else:
            notes.append(f"{var.label} 데이터가 부족해 이 항목은 표시하지 않아요.")

    if not estimates:
        raise DataUnavailableError(
            f"선택한 위치({lat:.2f}, {lon:.2f})의 {target.month}월 과거 해양 데이터가 비어 있어요.",
            "육지나 해안선에 너무 가까운 좌표일 수 있어요. 조금 더 먼바다 좌표를 선택해 주세요.",
        )

    used = sorted({y for e in estimates.values() for y in e.yearly_means})
    if len(used) < HISTORY_YEARS:
        notes.append(
            f"최근 {HISTORY_YEARS}년 중 {len(used)}개 연도만 사용할 수 있어 정확도가 낮을 수 있어요."
        )
    if any(e.method == "mean" for e in estimates.values()):
        notes.append("1개 연도 데이터뿐이라 추세 없이 평균값으로만 추정했어요.")

    return Forecast(target, location, float(lat), float(lon), used, estimates, notes)
