"""해양 기상 추정 Gradio 웹앱.

실행:  python app.py   →  http://127.0.0.1:7860
"""

from __future__ import annotations

from datetime import date, timedelta

import gradio as gr
import plotly.graph_objects as go

from marine_forecast import (
    HISTORY_YEARS,
    LOCATIONS,
    MAX_LEAD_DAYS,
    SST,
    WAVE,
    ForecastError,
    Forecast,
    VariableEstimate,
    forecast,
)

CUSTOM = "직접 입력 (위도·경도)"

# 차트 색상: 고정 배경(라이트/다크 모드 모두에서 동일하게 읽히도록)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e5e4df"

# 파고 구간: 상태(status) 색 + 라벨 — 색만으로 의미를 전달하지 않도록 텍스트 라벨을 함께 표시
WAVE_BANDS = [
    (0.0, 1.0, "#0ca30c", "🟢 잔잔"),
    (1.0, 2.0, "#fab219", "🟡 약간 높음"),
    (2.0, 3.0, "#ec835a", "🟠 높음"),
    (3.0, 99.0, "#d03b3b", "🔴 풍랑주의보 수준"),
]
# 수온 구간: 차가움(파랑) ↔ 따뜻함(빨강) 발산형, 가운데는 중립 회색
SST_BANDS = [
    (-2.0, 8.0, "#2a78d6", "🧊 매우 차가움"),
    (8.0, 15.0, "#9ec5f4", "❄️ 차가움"),
    (15.0, 20.0, "#e5e4df", "🌊 적당함"),
    (20.0, 25.0, "#f2a9a8", "☀️ 따뜻함"),
    (25.0, 40.0, "#e34948", "🔥 매우 따뜻함"),
]


def band_label(value: float, bands) -> str:
    for lo, hi, _, label in bands:
        if lo <= value < hi:
            return label
    return bands[-1][3]


# --------------------------------------------------------------------------------------
# 게이지 차트
# --------------------------------------------------------------------------------------
def gauge(est: VariableEstimate | None, bands, axis_max: float, title: str) -> go.Figure | None:
    if est is None:
        return None
    v = est.variable
    axis_min = bands[0][0]
    axis_max = max(axis_max, est.value + est.spread + 0.5)
    steps = [
        {"range": [max(lo, axis_min), min(hi, axis_max)], "color": color}
        for lo, hi, color, _ in bands
        if lo < axis_max
    ]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            domain={"x": [0, 1], "y": [0.12, 1]},
            value=round(est.value, 1),
            number={"suffix": f" {v.unit}", "font": {"size": 44, "color": INK}, "valueformat": ".1f"},
            title={
                "text": f"{title}<br><span style='font-size:13px;color:{INK_2}'>"
                f"{band_label(est.value, bands)} · 예상 범위 ±{est.spread:.1f}{v.unit}</span>",
                "font": {"size": 18, "color": INK},
            },
            gauge={
                "shape": "angular",
                "axis": {"range": [axis_min, axis_max], "tickcolor": INK_2, "tickfont": {"color": INK_2, "size": 11}},
                "bar": {"color": INK, "thickness": 0.18},
                "bgcolor": SURFACE,
                "borderwidth": 0,
                "steps": steps,
                # 회색 눈금선 = 최근 3년 동월 평균 위치
                "threshold": {"line": {"color": INK_2, "width": 2}, "thickness": 0.9, "value": est.mean},
            },
        )
    )
    diff = est.value - est.mean
    arrow = "▲" if diff > 0.05 else "▼" if diff < -0.05 else "＝"
    fig.add_annotation(
        x=0.5, y=0.0, xref="paper", yref="paper", showarrow=False,
        text=f"{arrow} 3년 평균({est.mean:.1f}{v.unit}) 대비 {diff:+.1f}{v.unit}",
        font={"size": 13, "color": INK_2},
    )
    fig.update_layout(
        height=300,
        margin={"l": 30, "r": 30, "t": 80, "b": 30},
        paper_bgcolor=SURFACE,
        font={"family": "Pretendard, Noto Sans KR, sans-serif", "color": INK},
    )
    return fig


def trend_chart(fc: Forecast, est: VariableEstimate | None) -> go.Figure | None:
    """연도별 동월 평균 + 추세선 + 목표 연도 예측점. (변수별로 하나 — 단위가 달라 축을 섞지 않음)"""
    if est is None:
        return None
    v = est.variable
    years = sorted(est.yearly_means)
    means = [est.yearly_means[y] for y in years]
    target_year = fc.target.year
    # 월평균 기준 추세선(월내 보정 전)
    line_years = years + ([target_year] if target_year not in years else [])
    mean_year = sum(years) / len(years)
    line_vals = [est.mean + est.trend_per_year * (y - mean_year) for y in line_years]

    # 막대 대신 점: 수온처럼 0에서 먼 값도 연도 간 차이가 보이도록 y축을 데이터 범위에 맞춘다.
    fig = go.Figure()
    fig.add_scatter(
        x=[str(y) for y in years], y=means, mode="markers", name=f"{fc.target.month}월 평균",
        marker={"color": "#86b6ef", "size": 12, "line": {"color": SURFACE, "width": 2}},
        hovertemplate="%{x}년 " + f"{fc.target.month}월 평균" + ": %{y:.2f}" + v.unit + "<extra></extra>",
    )
    fig.add_scatter(
        x=[str(y) for y in line_years], y=line_vals, mode="lines", name="선형 추세선",
        line={"color": "#2a78d6", "width": 2, "dash": "dot"},
        hovertemplate="추세: %{y:.2f}" + v.unit + "<extra></extra>",
    )
    fig.add_scatter(
        x=[str(target_year)], y=[est.value], mode="markers+text", name=f"{fc.target:%m/%d} 예측",
        marker={"color": INK, "size": 11, "line": {"color": SURFACE, "width": 2}},
        error_y={"type": "data", "array": [est.spread], "color": INK_2, "thickness": 1.5, "width": 6},
        text=[f"{est.value:.1f}{v.unit}"], textposition="middle left", textfont={"color": INK},
        hovertemplate=f"{fc.target:%Y-%m-%d} 예측: %{{y:.2f}}{v.unit} ± {est.spread:.2f}<extra></extra>",
    )
    fig.update_layout(
        title={"text": f"{v.label} — 최근 동월 평균과 추세", "font": {"size": 15, "color": INK}},
        height=280,
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font={"color": INK_2},
        legend={"orientation": "h", "y": -0.18},
        yaxis={"title": v.unit, "gridcolor": GRID, "zeroline": False},
        xaxis={"type": "category"},
        hovermode="x unified",
    )
    return fig


# --------------------------------------------------------------------------------------
# 텍스트 요약
# --------------------------------------------------------------------------------------
def summary_markdown(fc: Forecast) -> str:
    sst = fc.estimates.get(SST.key)
    wave = fc.estimates.get(WAVE.key)
    lines = [f"### 📅 {fc.target:%Y년 %m월 %d일} · 📍 {fc.location}  ", f"<small>좌표 {fc.lat:.2f}, {fc.lon:.2f}</small>", ""]

    if sst:
        lines.append(f"- 🌡️ **수온 {sst.value:.1f}℃** (±{sst.spread:.1f}) — {band_label(sst.value, SST_BANDS)}")
    if wave:
        lines.append(f"- 🌊 **파고 {wave.value:.1f}m** (±{wave.spread:.1f}) — {band_label(wave.value, WAVE_BANDS)}")

    # 한 줄 해석
    tips = []
    if wave:
        if wave.value >= 3.0:
            tips.append("⚠️ 풍랑주의보 기준(3m)에 가까운 높은 파도가 예상돼요. 출항·드론 해상 운용을 재검토하세요.")
        elif wave.value >= 2.0:
            tips.append("소형 선박·해상 드론 운용 시 주의가 필요한 파고예요.")
        elif wave.value >= 1.0:
            tips.append("다소 물결이 있는 바다예요. 소형 보트·해상 드론은 당일 바람 예보를 함께 확인하세요.")
        else:
            tips.append("평년 기준으로는 잔잔한 바다가 예상돼요.")
    if sst and sst.value < 10:
        tips.append("수온이 낮아 입수 활동 시 저체온에 유의하세요.")
    if tips:
        lines += ["", "> " + " ".join(tips)]

    lines += ["", "<details><summary>🔎 계산 근거 보기</summary>", ""]
    lines.append(f"사용 연도: {', '.join(map(str, fc.years_used))} ({fc.target.month}월 데이터)  ")
    for est in (sst, wave):
        if not est:
            continue
        u = est.variable.unit
        per_year = " / ".join(f"{y}: {m:.2f}{u}" for y, m in sorted(est.yearly_means.items()))
        lines.append(
            f"- **{est.variable.label}** — 연도별 평균 {per_year} → "
            f"{HISTORY_YEARS}년 평균 {est.mean:.2f}{u}, 추세 {est.trend_per_year:+.2f}{u}/년, "
            f"월내 변화 {est.intra_month_per_day:+.3f}{u}/일"
        )
    lines += ["", "</details>"]

    if fc.notes:
        lines += ["", "ℹ️ " + "  \nℹ️ ".join(fc.notes)]
    lines += [
        "",
        "<small>※ 과거 동월 통계로 추정한 '기후학적 예상값'이며, 실제 기상 예보가 아닙니다. "
        "출항 등 안전 판단은 기상청 해상예보를 함께 확인하세요.</small>",
    ]
    return "\n".join(lines)


def error_markdown(err: ForecastError) -> str:
    hint = f"\n\n💡 {err.hint}" if err.hint else ""
    return f"### 😥 {err.title}\n\n{err.message}{hint}"


# --------------------------------------------------------------------------------------
# 이벤트 핸들러
# --------------------------------------------------------------------------------------
def run(target, location, lat, lon):
    try:
        if location == CUSTOM:
            if lat is None or lon is None:
                raise ForecastError("위도·경도를 입력해 주세요.", "예: 위도 35.0, 경도 129.25 (부산 앞바다)")
            fc = forecast(target, "사용자 지정 위치", float(lat), float(lon))
        else:
            fc = forecast(target, location)
    except ForecastError as err:
        gr.Warning(f"{err.title}: {err.message}")
        return error_markdown(err), None, None, None, None
    except Exception as err:  # 예상치 못한 오류도 사용자에게는 친절하게
        msg = ForecastError("알 수 없는 오류가 발생했어요.", f"잠시 후 다시 시도해 주세요. (상세: {type(err).__name__})")
        gr.Warning(msg.message)
        return error_markdown(msg), None, None, None, None

    sst, wave = fc.estimates.get(SST.key), fc.estimates.get(WAVE.key)
    return (
        summary_markdown(fc),
        gauge(sst, SST_BANDS, 30.0, "🌡️ 예상 수온"),
        gauge(wave, WAVE_BANDS, 4.0, "🌊 예상 파고"),
        trend_chart(fc, sst),
        trend_chart(fc, wave),
    )


def toggle_custom(location):
    show = location == CUSTOM
    return gr.update(visible=show), gr.update(visible=show)


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
CSS = """
.gradio-container {max-width: 1100px !important; margin: auto;}
#title h1 {margin-bottom: 0;}
#run-btn {min-height: 48px;}
"""


def build_app() -> gr.Blocks:
    today = date.today()
    default_day = today + timedelta(days=7)

    with gr.Blocks(title="해양 날씨 예측") as demo:
        gr.Markdown(
            "# 🌊 해양 날씨 예측기\n"
            f"미래 날짜를 고르면 **최근 {HISTORY_YEARS}년 동월 데이터**의 평균과 선형 추세로 "
            "**수온·파고**를 빠르게 추정해 드려요. (데이터: Open-Meteo Marine API)",
            elem_id="title",
        )
        with gr.Row(equal_height=True):
            date_in = gr.DateTime(
                value=default_day.isoformat(),
                include_time=False,
                type="string",
                label="📅 예측 날짜",
                info=f"오늘부터 {MAX_LEAD_DAYS}일 이내",
                scale=2,
            )
            loc_in = gr.Dropdown(
                choices=list(LOCATIONS) + [CUSTOM], value="부산 앞바다", label="📍 해역", scale=2
            )
            lat_in = gr.Number(label="위도", value=35.0, visible=False, scale=1)
            lon_in = gr.Number(label="경도", value=129.25, visible=False, scale=1)
            btn = gr.Button("예측하기", variant="primary", elem_id="run-btn", scale=1)

        with gr.Row():
            sst_gauge = gr.Plot(label="수온", show_label=False)
            wave_gauge = gr.Plot(label="파고", show_label=False)
        summary = gr.Markdown("👆 날짜와 해역을 고른 뒤 **예측하기**를 눌러 주세요.")

        with gr.Accordion("📈 연도별 추세 자세히 보기", open=False):
            with gr.Row():
                sst_trend = gr.Plot(show_label=False)
                wave_trend = gr.Plot(show_label=False)

        outputs = [summary, sst_gauge, wave_gauge, sst_trend, wave_trend]
        inputs = [date_in, loc_in, lat_in, lon_in]
        loc_in.change(toggle_custom, loc_in, [lat_in, lon_in], show_progress="hidden")
        btn.click(run, inputs, outputs)
        date_in.submit(run, inputs, outputs)

    return demo


if __name__ == "__main__":
    build_app().launch(theme=gr.themes.Soft(primary_hue="blue"), css=CSS)
