# streamlit_app.py
# -*- coding: utf-8 -*-
"""
Streamlit + GitHub Codespaces 데이터 대시보드 (기후위기 정신건강/학업/미래 확장)

구성:
1) 공식 공개 데이터 대시보 (NASA POWER 일일 기온 API, 서울 좌표)
2) 사용자 입력 대시보드 (프롬프트의 "폭염일수" 표 고정 내장)
3) 기후위기 & 청소년 정신건강 (연구 참고) 탭
4) 기후위기 & 청소년 학업 (연구 참고) 탭
5) 기후위기, 우리의 미래 (대안 탐색) 탭 (★새로운 메뉴)

폰트:
- /fonts/Pretendard-Bold.ttf 존재 시 Streamlit/Plotly에 적용 시도(없으면 자동 생략)
"""

import io
import json
import math
import textwrap
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dateutil.relativedelta import relativedelta
import plotly.express as px

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(page_title="기후위기 & 청소년 대응 대시보드", layout="wide")

# Pretendard 적용 시도 (없으면 자동 생략)
def inject_font_css():
    font_path = Path("/fonts/Pretendard-Bold.ttf")
    if font_path.exists():
        st.markdown(
            f"""
            <style>
            @font-face {{
                font-family: 'Pretendard';
                src: url('file://{font_path.as_posix()}') format('truetype');
                font-weight: 700;
                font-style: normal;
            }}
            html, body, [class*="css"], .stMarkdown, .stButton, .stSelectbox, .stSlider, .stText, .stMetric, .stDataFrame {{
                font-family: 'Pretendard', 'Noto Sans KR', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

inject_font_css()

PLOTLY_FONT = "Pretendard, Noto Sans KR, Arial, sans-serif"

# 유틸
KST_TODAY = datetime.now()
TODAY_DATE = KST_TODAY.date()

def to_date(s):
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").date()
    except Exception:
        try:
            return datetime.strptime(str(s), "%Y%m%d").date()
        except Exception:
            return pd.NaT

def clamp_to_today(df, date_col="date"):
    if df.empty:
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    return df[df[date_col] <= TODAY_DATE]

def clean_standardize(df, date_col="date", value_col="value", group_col=None):
    df = df.copy()
    # 결측/중복 처리
    df = df.dropna(subset=[date_col])
    if group_col:
        df = df.drop_duplicates(subset=[date_col, group_col])
    else:
        df = df.drop_duplicates(subset=[date_col])
    # 타입 통일
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    # value를 숫자형으로
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    # 미래 데이터 제거
    df = clamp_to_today(df, date_col)
    return df

def download_button_for_df(df, filename, label="CSV 다운로드"):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

def plot_line(df, title, yaxis_title):
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return
    fig = px.line(
        df,
        x="date",
        y="value",
        color="group",
        markers=True,
        title=title,
    )
    fig.update_layout(
        xaxis_title="날짜",
        yaxis_title=yaxis_title,
        legend_title="지표",
        font=dict(family=PLOTLY_FONT),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_bar(df, title, yaxis_title, barmode="group"):
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return
    fig = px.bar(
        df,
        x="date",
        y="value",
        color="group",
        title=title,
        barmode=barmode,
    )
    fig.update_layout(
        xaxis_title="월",
        yaxis_title=yaxis_title,
        legend_title="지표",
        font=dict(family=PLOTLY_FONT),
        hovermode="x",
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 1) 공개 데이터 대시보드 함수
# -----------------------------
@st.cache_data(show_spinner=True, ttl=60 * 60)
def fetch_nasa_power_daily(lat=37.5665, lon=126.9780, start="2015-01-01", end=None):
    if end is None:
        end = TODAY_DATE.strftime("%Y-%m-%d")
    start_str = pd.to_datetime(start).strftime("%Y%m%d")
    end_str = pd.to_datetime(end).strftime("%Y%m%d")
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {"parameters": "T2M,T2M_MAX", "community": "RE", "latitude": lat, "longitude": lon, "start": start_str, "end": end_str, "format": "JSON"}
    try:
        r = requests.get(base_url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        t2m = js["properties"]["parameter"]["T2M"]
        t2m_max = js["properties"]["parameter"]["T2M_MAX"]
        records = []
        for k, v in t2m.items():
            d = to_date(k)
            if pd.isna(d): continue
            records.append({"date": d, "t2m": v, "t2m_max": t2m_max.get(k, np.nan)})
        df = pd.DataFrame(records)
        out = df.rename(columns={"t2m": "value"}).assign(group="일 평균기온(℃)")
        out2 = df.rename(columns={"t2m_max": "value"}).assign(group="일 최고기온(℃)")
        all_df = pd.concat([out[["date", "value", "group"]], out2[["date", "value", "group"]]], ignore_index=True)
        all_df = clean_standardize(all_df, "date", "value", "group")
        all_df["fallback"] = False
        return all_df
    except Exception:
        dates = pd.date_range(end=TODAY_DATE, periods=60, freq="D")
        np.random.seed(42)
        base = 27 + np.sin(np.linspace(0, 3 * np.pi, len(dates))) * 5
        avg = base + np.random.normal(0, 1.2, len(dates))
        tmax = avg + np.random.uniform(3, 8, len(dates))
        df = pd.DataFrame({"date": dates.date, "value": np.r_[avg, tmax], "group": ["일 평균기온(℃)"] * len(dates) + ["일 최고기온(℃)"] * len(dates)})
        df = clean_standardize(df, "date", "value", "group")
        df["fallback"] = True
        return df

def make_heatwave_flags(df, threshold_max=33.0):
    if df.empty: return df
    w = df.copy().pivot_table(index="date", columns="group", values="value")
    w["폭염일"] = (w.get("일 최고기온(℃)", pd.Series(index=w.index)) >= threshold_max).astype(int)
    out = (w.reset_index()[["date", "폭염일"]].rename(columns={"폭염일": "value"}).assign(group=f"폭염일(최고기온≥{threshold_max}℃)"))
    return clean_standardize(out, "date", "value", "group")

def monthly_summary(df):
    if df.empty: return df
    x = df.copy()
    x["year"] = pd.to_datetime(x["date"]).dt.year
    x["month"] = pd.to_datetime(x["date"]).dt.month
    def agg_fn(g):
        return pd.Series({"value": g["value"].sum()}) if g.name[2].startswith("폭염일") else pd.Series({"value": g["value"].mean()})
    m = (x.groupby(["year", "month", "group"], as_index=False).apply(agg_fn).reset_index(drop=True))
    m["date"] = pd.to_datetime(dict(year=m["year"], month=m["month"], day=1)).dt.date
    return m[["date", "value", "group", "year", "month"]]

def add_risk_annotation():
    st.markdown("""
        > 참고: **연구에 따르면, 하루 평균기온이 1°C 높아질 때마다 청소년(12~24세) 자살 충동/행동으로 인한 응급실 방문이 약 1.3% 증가**하는 경향이 관찰되었습니다.  
        > (호주 뉴사우스웨일스州, 2012–2019 시계열 분석. 인과 단정 불가, 참고 지표로만 활용)
        """)
    with st.expander("연구 출처(주석) 보기", expanded=False):
        st.code(textwrap.dedent("""
            PubMed (청소년 자살충동 1°C당 1.3% 증가):
            https://pubmed.ncbi.nlm.nih.gov/39441101/
            """), language="text")

# -----------------------------
# 2) 사용자 입력 대시보드 함수
# -----------------------------
@st.cache_data(show_spinner=False)
def load_user_table():
    raw = """연도,1월,2월,3월,4월,5월,6월,7월,8월,9월,10월,11월,12월,연합계,순위
2015,0,0,0,0,0,1,4,3,0,0,0,0,8,10
2016,0,0,0,0,0,0,4,20,0,0,0,0,24,4
2017,0,0,0,0,0,1,5,7,0,0,0,0,13,8
2018,0,0,0,0,0,0,16,19,0,0,0,0,35,1
2019,0,0,0,0,1,0,4,10,0,0,0,0,15,7
2020,0,0,0,0,0,2,0,2,0,0,0,0,4,11
2021,0,0,0,0,0,0,15,3,0,0,0,0,18,6
2022,0,0,0,0,0,0,10,0,0,0,0,0,10,9
2023,0,0,0,0,0,2,6,11,0,0,0,0,19,5
2024,0,0,0,0,0,4,2,21,6,0,0,0,33,2
2025,0,0,0,0,0,3,15,9,1,,,,28,3
평균,0.0,0.0,0.0,0.0,0.1,1.2,7.4,9.6,0.6,0.0,0.0,0.0,,  
"""
    df = pd.read_csv(io.StringIO(raw))
    df = df[df["연도"].apply(lambda x: str(x).isdigit())].copy()
    df["연도"] = df["연도"].astype(int)
    month_cols = ["1월","2월","3월","4월","5월","6월","7월","8월","9월","10월","11월","12월"]
    keep_cols = ["연도","연합계","순위"]
    for c in month_cols:
        if c not in df.columns: df[c] = np.nan
    m = df.melt(id_vars=keep_cols + ["연도"], value_vars=month_cols, var_name="월", value_name="폭염일수")
    m["월_int"] = m["월"].str.replace("월", "", regex=False).astype(int)
    m["date"] = pd.to_datetime(dict(year=m["연도"], month=m["월_int"], day=1)).dt.date
    m["value"] = pd.to_numeric(m["폭염일수"], errors="coerce")
    out = m[["date", "value", "연도"]].rename(columns={"연도": "group"})
    out = clean_standardize(out, "date", "value", "group")
    out = clamp_to_today(out, "date")
    yr = df[["연도", "연합계", "순위"]].rename(columns={"연도":"year","연합계":"total","순위":"rank"})
    yr["total"] = pd.to_numeric(yr["total"], errors="coerce")
    yr["rank"] = pd.to_numeric(yr["rank"], errors="coerce")
    return out, yr

def plot_user_monthly(df_long):
    if df_long.empty: st.info("표시할 데이터가 없습니다."); return
    fig = px.line(df_long, x="date", y="value", color="group", markers=True, title="연도별 월간 폭염일수 추이")
    fig.update_layout(xaxis_title="월", yaxis_title="폭염일수(일)", legend_title="연도", font=dict(family=PLOTLY_FONT), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

def plot_user_rank(yr):
    y2 = yr.dropna(subset=["year","total","rank"]).copy()
    if y2.empty: st.info("순위 데이터가 없습니다."); return
    y2["date"] = pd.to_datetime(dict(year=y2["year"], month=1, day=1)).dt.date
    fig = px.scatter(y2, x="year", y="rank", size="total", text="total", title="연도별 폭염일수 연합계 & 순위")
    fig.update_traces(textposition="top center")
    fig.update_layout(xaxis_title="연도", yaxis_title="순위(낮을수록 상위)", yaxis=dict(autorange="reversed"), font=dict(family=PLOTLY_FONT))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 3) 기후위기 & 청소년 정신건강 대시보드 함수
# -----------------------------
@st.cache_data(show_spinner=False)
def get_mental_health_indicators():
    research_indicators = pd.DataFrame([
        {"지표": "폭염 vs 우울증 위험 증가", "단위": "%", "값": 13, "출처": "연구(중국 청소년)", "설명": "폭염 강도 1단위 증가당"},
        {"지표": "폭염 vs 불안 위험 증가", "단위": "%", "값": 12, "출처": "연구(중국 청소년)", "설명": "폭염 강도 1단위 증가당"},
        {"지표": "기온 1°C↑ vs 우울 증상 위험 증가", "단위": "%", "값": 14, "출처": "연구(한국 성인 19-40세)", "설명": "1961-1990 대비 연평균 기온 1°C 증가당"},
    ])
    kyrbs_data = pd.DataFrame({"연도": [2021, 2022, 2023, 2024, 2025], "우울감 경험률(%)": [25.0, 26.5, 27.2, 28.5, 29.1], "자살 생각률(%)": [10.5, 11.0, 11.3, 11.5, 11.8]})
    kyrbs_data["date"] = pd.to_datetime(dict(year=kyrbs_data["연도"], month=1, day=1)).dt.date
    kyrbs_data = clamp_to_today(kyrbs_data, "date")
    melted_kyrbs = kyrbs_data.melt(id_vars=["연도", "date"], value_vars=["우울감 경험률(%)", "자살 생각률(%)"], var_name="group", value_name="value_perc").rename(columns={"value": "value_perc"})
    return research_indicators, melted_kyrbs

def plot_kyrbs_trend(df):
    if df.empty: st.info("청소년 정신건강 현황 데이터가 없습니다."); return
    fig = px.line(df, x="연도", y="value_perc", color="group", markers=True, title="청소년 정신건강 주요 지표 추이 (가상 데이터, KYRBS 등 참고)")
    fig.update_layout(xaxis_title="연도", yaxis_title="비율(%)", legend_title="지표", font=dict(family=PLOTLY_FONT), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 4) 기후위기 & 청소년 학업 대시보드 함수
# -----------------------------
@st.cache_data(show_spinner=False)
def get_academic_indicators():
    academic_indicators = pd.DataFrame([
        {"지표": "기온 1°C↑ vs 학업 성취도 하락", "단위": "%", "값": 1, "출처": "연구(미국, 에어컨X 교실)", "설명": "외부 온도가 $1^\circ \text{C}$ 상승 시"},
    ])
    start_year = 2018
    end_year = TODAY_DATE.year
    academic_data = pd.DataFrame({"연도": range(start_year, end_year + 1)})
    np.random.seed(45)
    loss_increase = np.linspace(0, 15, len(academic_data))
    noise = np.random.normal(0, 3, len(academic_data))
    academic_data["고온 학습 손실 지수(가상)"] = (10 + loss_increase + noise).clip(0, 30).round(1)
    np.random.seed(46)
    base_change = np.linspace(1.0, -1.0, len(academic_data))
    change_noise = np.random.normal(0, 0.5, len(academic_data))
    academic_data["학업 성취도 변화율(%p, 가상)"] = (base_change + change_noise).round(2)
    academic_data["date"] = pd.to_datetime(dict(year=academic_data["연도"], month=1, day=1)).dt.date
    academic_data = clamp_to_today(academic_data, "date")
    melted_academic = academic_data.melt(id_vars=["연도", "date"], value_vars=["고온 학습 손실 지수(가상)", "학업 성취도 변화율(%p, 가상)"], var_name="group", value_name="value")
    return academic_indicators, melted_academic

def plot_academic_trend(df):
    if df.empty: st.info("학업 관련 지표 데이터가 없습니다."); return
    loss_df = df[df["group"] == "고온 학습 손실 지수(가상)"]
    change_df = df[df["group"] == "학업 성취도 변화율(%p, 가상)"]
    fig1 = px.bar(loss_df, x="연도", y="value", title="고온 학습 손실 지수 추이 (가상 지표)", color_discrete_sequence=['#ff7f0e'])
    fig1.update_layout(xaxis_title="연도", yaxis_title="학습 손실 지수 (0-100)", font=dict(family=PLOTLY_FONT))
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = px.line(change_df, x="연도", y="value", markers=True, title="학업 성취도 변화율 추이 (전년 대비 %p, 가상 지표)", color_discrete_sequence=['#1f77b4'])
    fig2.update_traces(name="학업 성취도 변화율", showlegend=True)
    fig2.update_layout(xaxis_title="연도", yaxis_title="변화율 (%p)", font=dict(family=PLOTLY_FONT), shapes=[dict(type='line', xref='paper', yref='y', x0=0, x1=1, y0=0, y1=0, line=dict(color='Red', width=1, dash='dash'))])
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# 5) 기후위기, 우리의 미래 대시보드 함수 (신규 추가)
# -----------------------------
def display_future_solutions():
    """ 기후위기 해결 방안 (학생/학교 차원)을 텍스트로 구성 및 표시 """
    st.markdown("### 🧑‍🤝‍🧑 학생 차원의 실천 방안")
    st.info("청소년들은 작은 습관 변화로도 큰 영향을 줄 수 있습니다. 아래 행동들은 당장 실천할 수 있는 '기후 행동'의 시작입니다.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### 🔋 에너지 절약 및 친환경 소비")
        st.markdown(
            """
            * **플러그 뽑기 (대기전력 줄이기):** 사용하지 않는 가전제품의 플러그를 뽑아 불필요한 전력 소모를 막습니다.
            * **LED 조명 사용 및 소등:** 교실이나 집에서 나갈 때 반드시 불을 끕니다.
            * **개인 컵/텀블러 사용:** 일회용 컵 사용을 최소화하여 쓰레기와 제조 과정의 탄소를 줄입니다.
            * **로컬 푸드 및 채소 중심 식단:** 먼 거리에서 운송된 식품(푸드 마일리지) 대신 지역 농산물을 선호하고, 육류 소비를 줄여 축산업에서 발생하는 메탄가스 배출을 감소시킵니다.
            """
        )
    with colB:
        st.markdown("#### 🌳 자원 재활용 및 환경 운동")
        st.markdown(
            """
            * **올바른 분리수거 습관:** 비우고, 헹구고, 분리하고, 섞지 않는 '4대 원칙'을 철저히 지킵니다.
            * **물품 재사용 및 공유:** 헌 옷, 책, 학용품 등을 버리지 않고 재사용하거나 나눕니다.
            * **대중교통 및 자전거 이용:** 가까운 거리는 걷거나 자전거를 타고, 장거리는 대중교통을 이용해 자동차 배기가스 배출을 줄입니다.
            * **기후 관련 학습 및 참여:** 기후변화 관련 정보를 꾸준히 학습하고, 학교 환경 동아리나 지역 환경 운동에 적극적으로 참여합니다.
            """
        )

    st.markdown("---")
    st.markdown("### 🏫 학교/제도 차원의 대안")
    st.warning("학교와 교육 당국의 제도 개선은 학생들의 기후 위기 적응력과 대응 능력을 높이는 핵심적인 방안입니다.")
    
    colC, colD = st.columns(2)
    with colC:
        st.markdown("#### 💡 교육 및 인식 개선")
        st.markdown(
            """
            * **기후 위기 적응 교육 강화:** 폭염, 폭우 등 기후 재난 상황에 대한 **안전 교육 및 심리적 회복탄력성 교육**을 정규 교과 과정에 포함해야 합니다.
            * **친환경 교과목 확대:** 기후 변화, 에너지 전환, 지속 가능한 개발 목표(SDGs)를 다루는 심화 과목을 개설하고 동아리 활동을 지원해야 합니다.
            * **환경 교육 의무화:** 초·중·고 교육 전반에 걸쳐 기후·환경 교육 시간을 의무화하고 전문 교사를 배치해야 합니다.
            """
        )
    with colD:
        st.markdown("#### 🌿 학교 환경 개선 및 제도 마련")
        st.markdown(
            """
            * **학교 건물의 그린 리모델링:** 고효율 단열재, 고성능 창호 등을 적용하여 냉난방 에너지 효율을 높이고 탄소 배출을 줄여야 합니다.
            * **쿨링 스페이스(Cooling Space) 확보:** 폭염 시 학생들이 안전하게 휴식하고 학습할 수 있도록 냉방 시설이 잘 갖춰진 공간을 확충해야 합니다.
            * **친환경 급식 시스템 도입:** 식자재 운송 거리를 최소화하고, 채식 선택지를 확대하는 등 탄소 발자국을 줄이는 급식 체계를 구축해야 합니다.
            * **탄소 중립 학교 선언:** 학교 운영 전반에서 탄소 배출량을 제로화하기 위한 목표를 설정하고, 태양광 발전 시설 등을 도입해야 합니다.
            """
        )

# -----------------------------
# 6) 보고서 출력 함수 (링크 업데이트)
# -----------------------------
def display_report():
    """ '환경오염이 바꾸는 미래의 학교생활' 보고서를 마크다운으로 출력 """
    report_text = """
## 📄 환경오염이 바꾸는 미래의 학교생활 보고서

### 서론, 청소년의 학교 생활을 위협하는 환경오염
오늘날 인류는 기후위기라는 심각한 도전에 직면해 있습니다. 지구 평균기온 상승은 단순한 날씨 변화에 그치지 않고, 홍수·가뭄뿐 아니라 **폭염을 일상화**시키고 있습니다. 특히 폭염은 청소년의 건강과 학교생활에 직접적인 부담을 주는 중요한 문제로 지적됩니다.

*(원문 인용: '1997년 8월' 대비 '2025년 8월' 폭염/대기질 심화 지도 자료)*
이 보고서는 환경오염으로 인한 해수면 상승과 폭염의 심화, 그리고 이로 인해 청소년의 학습 환경과 학교생활이 어떻게 달라질지, 그리고 그 것에 대해 어떻게 대응할지를 살펴보고자 합니다.

---

### 본론 1, 데이터로 확인한 환경오염의 주요 원인
환경오염은 다양한 원인에서 비롯되며, 이는 청소년의 건강과 학교생활에도 직접적인 영향을 미칩니다.
우리나라가 직면한 주요 환경문제는 **쓰레기·폐기물 처리 문제(65.6%)**, **대기오염·미세먼지 문제(51%)**, **과도한 포장과 폐플라스틱 쓰레기 발생(40.4%)** 등이 높은 비율을 차지합니다.

* **대기오염/미세먼지:** 청소년의 **호흡기 질환** 및 **알레르기**를 유발하여 학습 집중도를 떨어뜨리고 **결석률**을 높입니다. 야외 체육 활동 수행을 어렵게 만듭니다.
* **쓰레기/플라스틱:** 학교 급식에 사용되는 식재료의 안전성과 품질에도 영향을 미쳐 청소년들의 건강한 식습관 형성에 부정적인 요인이 될 수 있습니다.

결국 환경오염은 청소년의 학교생활을 **신체적, 정신적, 생활습관적 측면에서 모두 제약**하고 있으며, 기후위기와 맞물려 그 영향은 더욱 심화될 가능성이 큽니다.

---

### 본론 2, 폭염이 청소년의 학업과 건강에 미치는 영향
폭염은 청소년의 학습 활동 전반에 심각한 위협이 됩니다.

#### 🎓 학업 성취도 저하
* **연구 결과:** 미국 뉴욕시 공립 고등학생 약 100만 명을 분석한 결과, 기온이 **21°C에서 32°C로 상승**할 때 시험 성적은 평균 **15% 하락**했고, 과목 통과율은 **10.9% 감소**, 졸업 가능성도 **2.5%p** 낮아졌습니다.

#### 🧠 정신 건강 위험 증가
* **자살 충동/행동:** 국내외 연구에 따르면 하루 평균 기온이 **1°C 높아질 때마다** 청소년의 자살 충동이나 행동으로 인한 응급실 방문 건수가 **1.3% 증가**하는 것으로 확인되었습니다.
* **우울/불안:** 폭염 강도가 1단위 증가할 때마다 **우울증 발생률은 13%**, **불안 발생률은 12%** 증가했습니다. 이러한 정신적 부담은 청소년의 학업 스트레스를 심화시킵니다.
* **기후 불안 (Climate Anxiety):** 이미 학업 스트레스가 높은 청소년들에게 기후 변화에 대한 불안감이 더해져 심리적 압박이 커지며, 이는 미래 학습 및 진로에도 영향을 미칩니다.

#### 🏃 신체적 건강 문제
* **학교생활 질 저하:** 교실 온도 상승으로 인한 두통, 졸음, 탈수 증세는 학업 성취도 저하로 이어집니다. 폭염으로 인한 야외 체육 수업 취소는 청소년의 신체 발달 기회 부족을 초래합니다.

---

### 결론, 지속 가능한 미래를 위한 청소년과 학교의 역할
환경오염과 기후위기는 청소년의 학습 환경을 위협하는 중요한 요소입니다. 해수면 상승과 폭염 심화는 단순한 자연현상이 아니라, 미래의 학교생활을 변화시키는 직접적 원인이 될 수 있습니다. 쾌적하고 안전한 학습 환경을 지키기 위한 **청소년 스스로의 실천**과 **학교/사회 전체의 제도적 대응**이 동시에 필요합니다.

#### 🧑‍🤝‍🧑 학생 차원의 실천
* **에너지 절약:** 사용하지 않는 전자제품 전원 끄기, 샤워 시간 줄이기.
* **친환경 소비:** 텀블러, 장바구니 이용, 남은 음식 재활용하기.
* **친환경 이동:** 대중교통 및 자전거 이용하기.

#### 🏫 학교/제도 차원의 대안
* **교육 모델:** 지역별·학교별 특성에 맞는 **맞춤형 환경교육** 강화, 기후위기 대응 역량을 키울 수 있는 **체험 중심 학습 확대**.
* **시설 개선:** 학교 건물의 **그린 리모델링**으로 에너지 효율 개선, 폭염 대비 **쿨링 스페이스(Cooling Space) 확보**.
* **급식/운영:** 친환경 급식 시스템 도입, **탄소 중립 학교 선언** 및 에너지 관리 시스템 도입.

미래의 학교생활이 지속 가능하고 안전하려면, 청소년 한 명 한 명의 실천과 더불어 사회 전체가 **주 에너지를 친환경적으로 바꿀 수 있는 리더십**을 발휘해야 합니다. 이는 다음 세대를 위한 책임 있는 선택이 될 것입니다.

---

### 참고 자료
- 기상청 기상자료개방포털, 폭염일수 분포도
- 연합뉴스, 가장 시급한 환경문제는…2년 연속 '쓰레기·폐기물 처리' 꼽혀(2022-04-10)
- [링크 1](https://pmc.ncbi.nlm.nih.gov/a)
- [한겨레 기사](https://www.hani.co.kr/arti/hanihealth/healthlife/1212006.html)
- [UNSW 기사](https://www.unsw.edu.au/newsroom/news/2024/10/rise-in-suicidal-behaviours-among-young-people-linked-to-hotter-temperatures)
- [링크 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC7983931/)
"""
    st.markdown(report_text, unsafe_allow_html=True)

# -----------------------------
# 사이드바
# -----------------------------
with st.sidebar:
    st.header("옵션")
    st.caption("※ 모든 라벨은 한국어, 오늘 이후 데이터는 자동 제거됩니다.")
    # (탭 1, 탭 2의 사이드바 옵션은 기존대로 유지)

# -----------------------------
# 탭 구성
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 공개 데이터 대시보드", 
    "📘 사용자 입력 대시보드", 
    "🧠 기후위기 & 청소년 정신건강(연구참고)", 
    "📚 기후위기 & 청소년 학업(연구참고)",
    "🌱 기후위기, 우리의 미래" # ★새로운 탭 추가
])

# (탭 1, 탭 2, 탭 3, 탭 4의 내용은 위 함수 호출과 동일하게 유지)
with tab1:
    st.subheader("서울 일별 기온 & 폭염일 (NASA POWER)")
    st.caption("출처: NASA POWER API (T2M/T2M_MAX). API 실패 시 예시 데이터로 대체 표시됩니다.")

    colA, colB, colC = st.columns(3)
    with colA:
        start_date = st.date_input("조회 시작일", value=date(2015,1,1), min_value=date(1981,1,1), max_value=TODAY_DATE)
    with colB:
        end_date = st.date_input("조회 종료일", value=TODAY_DATE, min_value=start_date, max_value=TODAY_DATE)
    with colC:
        hw_threshold = st.number_input("폭염 기준(일최고기온, ℃)", min_value=30.0, max_value=40.0, value=33.0, step=0.5)

    data = fetch_nasa_power_daily(start=start_date.isoformat(), end=end_date.isoformat())
    if data["fallback"].any():
        st.warning("API 호출 실패로 예시 데이터가 표시됩니다. (네트워크/서비스 상태 확인 필요)")

    hw = make_heatwave_flags(data, threshold_max=hw_threshold)
    std = pd.concat([data[["date","value","group"]], hw[["date","value","group"]]], ignore_index=True)
    std = clean_standardize(std, "date", "value", "group")

    if not std.empty:
        min_d = pd.to_datetime(std["date"]).min().date()
        max_d = pd.to_datetime(std["date"]).max().date()
        
        with st.sidebar:
            st.markdown("#### 공개 데이터 기간 필터")
            rng = st.slider("표시 기간 선택", min_value=min_d, max_value=max_d, value=(min_d, max_d), key="tab1_rng")
            smooth_win = st.select_slider("이동평균 윈도우(일, 기온에만 적용)", options=[1,3,5,7,14], value=3, key="tab1_smooth")
        
        std = std[(std["date"] >= rng[0]) & (std["date"] <= rng[1])]
        
        if smooth_win > 1 and not std.empty:
            gtemp = std["group"].isin(["일 평균기온(℃)","일 최고기온(℃)"])
            std.loc[gtemp, "value"] = (
                std[gtemp]
                .sort_values("date")
                .groupby("group")["value"]
                .transform(lambda s: s.rolling(smooth_win, min_periods=1).mean())
            )

    plot_line(std[std["group"].isin(["일 평균기온(℃)", "일 최고기온(℃)"])], "일별 기온 추이", "기온(℃)")
    msum = monthly_summary(pd.concat([data[["date","value","group"]], hw], ignore_index=True))
    monthly_heat = msum[msum["group"].str.startswith("폭염일")]
    monthly_temp = msum[~msum["group"].str.startswith("폭염일")]
    plot_bar(monthly_heat, "월별 폭염일수(합계)", "폭염일수(일)")
    plot_line(monthly_temp, "월별 평균 기온/최고기온(평균)", "기온(℃)")

    add_risk_annotation()
    st.info("※ 본 대시보드는 **기온·폭염과 정신건강 지표 간 상관성**에 대한 참고 탐색용입니다. 인과관계를 단정하지 않으며, 지역·연령·제도 차이에 따라 결과 해석에 주의가 필요합니다.")

    st.markdown("#### 전처리된 표 다운로드")
    download_button_for_df(std[["date","value","group"]].sort_values(["date","group"]), "nasa_power_standardized.csv", "CSV 다운로드 (공개 데이터)")
    st.caption("주석: NASA POWER API 문서 URL은 코드 주석에 기재되어 있습니다. (앱 상단 주석 참조)")

with tab2:
    st.subheader("사용자 입력 데이터 대시보드 — 폭염일수(연도·월)")
    st.caption("프롬프트로 제공된 표만 사용합니다. 업로드나 추가 입력을 요구하지 않습니다.")

    user_long, user_year = load_user_table()

    if not user_long.empty:
        y_min = int(pd.to_datetime(user_long["date"]).dt.year.min())
        y_max = int(pd.to_datetime(user_long["date"]).dt.year.max())
        with st.sidebar:
            st.markdown("#### 사용자 데이터 기간/스무딩")
            y_start, y_end = st.slider("표시 연도 범위", min_value=y_min, max_value=y_max, value=(y_min, y_max), key="tab2_yr_rng")
            smooth_months = st.select_slider("월 이동평균(연도별 적용)", options=[1,3], value=1, key="tab2_smooth")

        view_df = user_long[(pd.to_datetime(user_long["date"]).dt.year >= y_start) & (pd.to_datetime(user_long["date"]).dt.year <= y_end)]
    else:
        view_df = user_long
        smooth_months = 1 

    if smooth_months > 1 and not view_df.empty:
        view_df = view_df.sort_values(["group","date"]).copy()
        view_df["value"] = view_df.groupby("group")["value"].transform(lambda s: s.rolling(smooth_months, min_periods=1).mean())

    plot_user_monthly(view_df)
    st.markdown("---")
    plot_user_rank(user_year)

    st.markdown("#### 전처리된 표 (표준화: date, value, group)")
    st.dataframe(view_df.sort_values(["date","group"]), use_container_width=True)
    download_button_for_df(view_df.sort_values(["date","group"]), "user_heatdays_standardized.csv", "CSV 다운로드 (사용자 데이터)")

with tab3:
    st.subheader("기후위기 & 청소년 정신건강(연구 참고)")
    st.caption("기온 변화와 정신건강 지표의 상관관계 연구 결과 및 한국 청소년 현황 통계를 간접 지표로 활용합니다.")

    research_df, kyrbs_df = get_mental_health_indicators()

    st.markdown("#### 🌡️ 기온 변화와 정신건강 위험도 증가 (주요 연구 인용)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label=f"{research_df.iloc[0]['지표']} ({research_df.iloc[0]['설명']})",
            value=f"+{research_df.iloc[0]['값']}{research_df.iloc[0]['단위']}",
            help=f"출처: {research_df.iloc[0]['출처']}"
        )
    with col2:
        st.metric(
            label=f"{research_df.iloc[1]['지표']} ({research_df.iloc[1]['설명']})",
            value=f"+{research_df.iloc[1]['값']}{research_df.iloc[1]['단위']}",
            help=f"출처: {research_df.iloc[1]['출처']}"
        )
    with col3:
        st.metric(
            label=f"{research_df.iloc[2]['지표']} ({research_df.iloc[2]['설명']})",
            value=f"+{research_df.iloc[2]['값']}{research_df.iloc[2]['단위']}",
            help=f"출처: {research_df.iloc[2]['출처']}"
        )

    st.markdown("---")
    
    st.markdown("#### 🇰🇷 한국 청소년 정신건강 현황 추이 (KYRBS 기반 예시)")
    plot_kyrbs_trend(kyrbs_df)

    st.markdown("#### 💡 데이터 유의 사항 및 연구 출처")
    st.warning(
        "**주의:** 이 탭의 데이터는 실제 공개 시계열이 아닌 **연구 인용 및 임의로 생성된 예시 데이터**입니다. 인과관계 단정을 피하고 참고 지표로만 활용해야 합니다."
    )
    with st.expander("참고 문헌 (주석)", expanded=False):
        st.markdown(
            """
            * **기존 연구 (청소년 자살충동 vs 기온):** PubMed: https://pubmed.ncbi.nlm.nih.gov/39441101/
            * **폭염 vs 우울증/불안 (중국 청소년):** Journal of Affective Disorders (2024). 
            * **기온 1°C↑ vs 우울 증상 (한국 성인 19-40세):** PubMed (2024). 
            * **한국 청소년 정신건강 현황:** 청소년건강행태조사(KYRBS)의 공표 통계를 참고하여 임의의 시계열 예시 데이터를 생성했습니다.
            """
        )

with tab4:
    st.subheader("📚 기후위기 & 청소년 학업 성취도(연구 참고)")
    st.caption("고온 환경이 학생들의 학습 능력에 미치는 영향을 다루는 해외 연구 결과를 인용하고, 가상 지표를 통해 영향을 탐색합니다.")

    academic_df, academic_melted_df = get_academic_indicators()
    
    # 1. 주요 연구 인용 요약
    st.markdown("#### 🌡️ 고온 노출과 학업 성취도 하락 (주요 연구 인용)")
    
    col_학업1, col_학업2, col_학업3 = st.columns(3)
    
    with col_학업1:
        st.metric(
            label=f"{academic_df.iloc[0]['지표']} ({academic_df.iloc[0]['설명']})",
            value=f"-{academic_df.iloc[0]['값']}{academic_df.iloc[0]['단위']}",
            help=f"출처: {academic_df.iloc[0]['출처']}"
        )
    with col_학업2:
        st.metric(
            label="연구 인용",
            value="👇",
            help="여름철 더위가 학생들의 학습 집중력과 기억력에 부정적 영향을 미침"
        )
    with col_학업3:
        # 가상의 지표로 채우기
        st.metric(
            label="고온 학습 손실 지수",
            value=f"{academic_melted_df[academic_melted_df['연도'] == academic_melted_df['연도'].max()]['value'].iloc[0].round(1)}",
            delta=f"{(academic_melted_df[academic_melted_df['연도'] == academic_melted_df['연도'].max()]['value'].iloc[0] - academic_melted_df[academic_melted_df['연도'] == academic_melted_df['연도'].min()]['value'].iloc[0]).round(1)}",
            delta_color="inverse", # 값이 높을수록 위험하므로 반전
            help="가상 지표: 최근 연도 '고온 학습 손실 지수' (최소 연도 대비 변화)"
        )

    st.markdown("---")

    # 2. 가상 시계열 지표 시각화
    st.markdown("#### 📈 고온 노출 관련 가상 학업 지표 추이")
    plot_academic_trend(academic_melted_df)


    # 3. 상세 연구 출처 및 유의 사항
    st.markdown("#### 💡 데이터 유의 사항 및 연구 출처")
    st.warning(
        "**주의:** 이 탭의 데이터는 실제 한국의 학업 성적 통계가 아닌, **해외 연구 인용 및 고온 영향 시나리오를 가정한 임의의 시계열 예시 데이터**입니다. "
        "기온 상승과 학업 성적 하락의 **상관관계**를 참고하는 용도로만 활용해야 합니다."
    )
    with st.expander("참고 문헌 (주석)", expanded=False):
        st.markdown(
            """
            * **기온 1°C↑ vs 학업 성취도 하락:** AERJ (2020) 논문 등. 에어컨이 없는 미국 교실 대상 연구에서 고온 노출과 학업 성취도 하락 간의 유의미한 상관관계 발견.
            * **가상 지표:** 고온 학습 손실 지수 및 학업 성취도 변화율은 **해당 연구 결과를 바탕으로 영향을 시뮬레이션한 임의의 값**입니다.
            """
        )

with tab5: # ★새로 추가된 '기후위기, 우리의 미래' 탭
    st.subheader("🌱 기후위기, 우리의 미래: 청소년과 학교의 대응 방안")
    st.caption("기후위기의 영향을 넘어, 학생들이 직접 실천하고 학교가 변화해야 할 구체적인 행동 방안을 탐색합니다.")
    display_future_solutions()
    
    st.markdown("---")
    st.info("기후위기 대응은 모두의 책임이며, 청소년들의 작은 실천과 학교의 제도적 지원이 더 나은 미래를 만드는 핵심 동력입니다.")


# -----------------------------
# 보고서 및 푸터 (항상 유지)
# -----------------------------
st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)
display_report()
st.markdown("---")
st.caption("© Streamlit 대시보드 예시. 데이터는 공개 API/제공 표/연구 인용 기준으로 구성되며, 오늘(로컬 자정) 이후 데이터는 제거됩니다.")