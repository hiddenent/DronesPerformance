import calendar
from datetime import date

import pytest
import requests

import marine_forecast as mf

TODAY = date(2026, 11, 5)


def make_payload(year, month, sst_fn, wave_fn, missing=False):
    """Open-Meteo 응답 형식의 가짜 시간별 데이터."""
    n = calendar.monthrange(year, month)[1]
    times, sst, wave = [], [], []
    for d in range(1, n + 1):
        for h in range(24):
            times.append(f"{year}-{month:02d}-{d:02d}T{h:02d}:00")
            sst.append(None if missing else sst_fn(year, d))
            wave.append(None if missing else wave_fn(year, d))
    return {"hourly": {"time": times, "sea_surface_temperature": sst, "wave_height": wave}}


def linear_fetcher(lat, lon, year, month):
    # 수온: 연 +0.5℃ 상승, 월내 하루 -0.1℃ / 파고: 연 +0.1m, 월내 변화 없음
    return make_payload(
        year, month,
        sst_fn=lambda y, d: 18.0 + 0.5 * (y - 2023) - 0.1 * (d - 15.5),
        wave_fn=lambda y, d: 1.0 + 0.1 * (y - 2023),
    )


def test_reference_years_skip_incomplete_current_month():
    assert mf.reference_years(date(2026, 11, 20), TODAY) == [2023, 2024, 2025]
    assert mf.reference_years(date(2027, 3, 1), TODAY) == [2024, 2025, 2026]


def test_linear_trend_is_recovered():
    fc = mf.forecast("2026-11-20", "부산 앞바다", today=TODAY, fetcher=linear_fetcher)
    sst = fc.estimates["sea_surface_temperature"]
    wave = fc.estimates["wave_height"]
    # 2026년: 18 + 0.5*3 = 19.5, 20일 → 19.5 - 0.1*(20-15.5) = 19.05
    assert sst.value == pytest.approx(19.05, abs=1e-6)
    assert sst.trend_per_year == pytest.approx(0.5)
    assert sst.intra_month_per_day == pytest.approx(-0.1)
    assert wave.value == pytest.approx(1.3, abs=1e-6)
    assert fc.years_used == [2023, 2024, 2025]
    assert fc.notes == []


def test_accepts_gradio_datetime_string_with_time():
    fc = mf.forecast("2026-11-20 00:00:00", "부산 앞바다", today=TODAY, fetcher=linear_fetcher)
    assert fc.target == date(2026, 11, 20)


@pytest.mark.parametrize("value, fragment", [
    ("2026-10-01", "이미 지난"),
    ("2028-01-01", "너무 먼 미래"),
    ("", "선택되지"),
    ("내일", "날짜로 읽을 수 없"),
])
def test_invalid_dates_give_friendly_message(value, fragment):
    with pytest.raises(mf.InvalidDateError) as exc:
        mf.forecast(value, "부산 앞바다", today=TODAY, fetcher=linear_fetcher)
    assert fragment in exc.value.message
    assert exc.value.hint


def test_all_api_calls_fail():
    def failing(*_):
        raise mf.ApiError("서버 오류", "다시 시도")

    with pytest.raises(mf.ApiError):
        mf.forecast("2026-11-20", "부산 앞바다", today=TODAY, fetcher=failing)


def test_partial_api_failure_still_predicts_with_note():
    def flaky(lat, lon, year, month):
        if year == 2023:
            raise mf.ApiError("timeout")
        return linear_fetcher(lat, lon, year, month)

    fc = mf.forecast("2026-11-20", "부산 앞바다", today=TODAY, fetcher=flaky)
    assert fc.years_used == [2024, 2025]
    assert any("일부 연도" in n for n in fc.notes)


def test_land_point_without_data_raises_data_unavailable():
    def empty(lat, lon, year, month):
        return make_payload(year, month, None, None, missing=True)

    with pytest.raises(mf.DataUnavailableError) as exc:
        mf.forecast("2026-11-20", "x", 36.5, 127.8, today=TODAY, fetcher=empty)
    assert "육지" in exc.value.hint


def test_single_year_falls_back_to_mean():
    def only_2025(lat, lon, year, month):
        if year != 2025:
            return make_payload(year, month, None, None, missing=True)
        return linear_fetcher(lat, lon, year, month)

    fc = mf.forecast("2026-11-20", "부산 앞바다", today=TODAY, fetcher=only_2025)
    assert fc.estimates["wave_height"].method == "mean"
    assert fc.estimates["wave_height"].value == pytest.approx(1.2)
    assert any("평균값으로만" in n for n in fc.notes)


def test_wave_height_never_negative():
    def falling(lat, lon, year, month):
        return make_payload(year, month, lambda y, d: 15.0, lambda y, d: max(0.0, 1.0 - 0.45 * (y - 2023)))

    fc = mf.forecast("2026-11-20", "부산 앞바다", today=TODAY, fetcher=falling)
    assert fc.estimates["wave_height"].value >= 0.0


def test_fetch_month_maps_network_error_to_api_error(monkeypatch):
    mf.fetch_month.cache_clear()

    def boom(*_, **__):
        raise requests.ConnectionError("no route")

    monkeypatch.setattr(mf.requests, "get", boom)
    with pytest.raises(mf.ApiError) as exc:
        mf.fetch_month(35.0, 129.25, 2025, 11)
    assert "연결" in exc.value.message
