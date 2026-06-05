from __future__ import annotations

import base64
import html as html_lib
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
ASSET_DIR = BASE_DIR / "assets"
STYLE_PATH = BASE_DIR / "style.css"
LOGO_PATH  = ASSET_DIR / "deepshield_logo.png"
DATASET_YEAR = "2020-2026"

DATA_FILES: Dict[str, str] = {
    "balance":           "balance_summary.csv",
    "class_distribution":"class_distribution.csv",
    "image_features":    "image_features.csv",
    "image_inventory":   "image_inventory.csv",
    "pixel_stats":       "pixel_brightness_contrast_statistics.csv",
    "pixel_test":        "pixel_statistical_test.csv",
    "rgb_diff":          "rgb_channel_difference_summary.csv",
    "rgb_stats":         "rgb_channel_statistics.csv",
    "split_class":       "split_class_distribution.csv",
}

CLASS_ORDER   = ["Fake", "Real"]
SPLIT_ORDER   = ["Train", "Validation", "Test"]
CHANNEL_ORDER = ["Red", "Green", "Blue"]

COLORS: Dict[str, str] = {
    "bg":          "#08001A",
    "bg2":         "#12002B",
    "panel":       "rgba(22, 9, 48, 0.82)",
    "panel2":      "rgba(29, 13, 63, 0.78)",
    "stroke":      "rgba(139, 92, 246, 0.45)",
    "stroke_cyan": "rgba(34, 184, 207, 0.48)",
    "text":        "#FBF8FF",
    "muted":       "#B7ADD6",
    "muted2":      "#8D82AA",
    "fake":        "#C84BCB",
    "real":        "#38AFCF",
    "magenta":     "#C84BCB",
    "pink":        "#D946C9",
    "cyan":        "#38AFCF",
    "blue":        "#22B8CF",
    "purple":      "#8B5CF6",
    "green":       "#34D399",
    "orange":      "#F0B35D",
    "red":         "#F87171",
    "yellow":      "#FACC15",
    "white":       "#FBF8FF",
}

CLASS_COLORS = {"Fake": COLORS["fake"], "Real": COLORS["real"]}

# ---------------------------------------------------------------------------
# Pertanyaan Bisnis
# ---------------------------------------------------------------------------
BQ_QUESTIONS = {
    "BQ1": (
        "Apakah jumlah gambar Fake dan Real dalam dataset tahun 2020-2026 sudah seimbang, "
        "sehingga model dapat dilatih tanpa perlu penyesuaian khusus seperti class weight "
        "atau undersampling?"
    ),
    "BQ2": (
        "Apakah gambar Fake dan Real berbeda secara berarti dalam hal kecerahan (brightness) "
        "dan kontras pada dataset tahun 2020-2026, sehingga model perlu dibantu dengan "
        "augmentasi color jittering saat pelatihan?"
    ),
    "BQ3": (
        "Apakah warna (channel Red, Green, Blue) pada gambar Fake dan Real menunjukkan "
        "pergeseran distribusi yang cukup dalam dataset tahun 2020-2026, sehingga bisa "
        "dimanfaatkan untuk merancang arsitektur model yang lebih cerdas melalui Channel Attention?"
    ),
}

# ---------------------------------------------------------------------------
# Kerangka SMART
# ---------------------------------------------------------------------------
SMART_ROWS = {
    "BQ1": [
        (
            "S - Spesifik",
            "Menghitung jumlah dan persentase gambar Fake dan Real pada setiap split "
            "(Train, Validation, Test) dalam dataset tahun 2020-2026.",
        ),
        (
            "M - Terukur",
            "Diukur lewat rasio kelas, selisih absolut jumlah sampel, dan persentase "
            "ketimpangan. Ambang batas: rasio lebih dari 1,5:1 dianggap perlu penanganan khusus.",
        ),
        (
            "A - Dapat Ditindaklanjuti",
            "Jika tidak seimbang: terapkan class weight atau random undersampling pada "
            "split Train. Jika seimbang: latih model tanpa penyesuaian khusus.",
        ),
        (
            "R - Relevan",
            "Ketimpangan yang tidak ditangani membuat model cenderung memprediksi kelas "
            "mayoritas sehingga akurasi tampak tinggi padahal recall kelas minoritas bisa buruk.",
        ),
        (
            "T - Terikat Waktu",
            "Analisis berbasis dataset yang dikumpulkan pada tahun 2020-2026 "
            "sebagai referensi utama proyek.",
        ),
    ],
    "BQ2": [
        (
            "S - Spesifik",
            "Membandingkan rata-rata kecerahan grayscale (brightness_mean) dan "
            "standar deviasi kontras (contrast_std) antara kelas Fake dan Real.",
        ),
        (
            "M - Terukur",
            "Dievaluasi dengan uji Mann-Whitney U (p-value) dan Cohen's d sebagai ukuran "
            "efek praktis. Cohen's d lebih dari 0,5 berarti perbedaan bermakna secara praktis.",
        ),
        (
            "A - Dapat Ditindaklanjuti",
            "Jika perbedaan besar secara praktis: tambahkan color jittering agar model "
            "tidak belajar dari bias pencahayaan. Jika kecil: normalisasi gambar adalah prioritas utama.",
        ),
        (
            "R - Relevan",
            "Perbedaan kecerahan yang sistematis bisa menjadi jalan pintas yang dipelajari "
            "model karena kondisi cahaya berbeda, bukan karena pola deepfake yang sesungguhnya.",
        ),
        (
            "T - Terikat Waktu",
            "Analisis berbasis data tahun 2020-2026. Distribusi bisa berbeda pada dataset "
            "tahun lain.",
        ),
    ],
    "BQ3": [
        (
            "S - Spesifik",
            "Membandingkan rata-rata intensitas channel Red, Green, dan Blue antara kelas "
            "Fake dan Real, lalu mengidentifikasi channel mana yang paling berbeda.",
        ),
        (
            "M - Terukur",
            "Dievaluasi dengan distribusi KDE overlay, boxplot, dan Cohen's d per channel. "
            "Channel dengan nilai mutlak d lebih dari 0,2 dianggap memiliki sinyal yang relevan.",
        ),
        (
            "A - Dapat Ditindaklanjuti",
            "Jika ada channel yang konsisten berbeda: pertimbangkan arsitektur Channel "
            "Attention seperti SE-Net atau CBAM untuk memberi bobot lebih pada channel informatif.",
        ),
        (
            "R - Relevan",
            "Proses pembuatan deepfake sering meninggalkan jejak warna pada channel tertentu, "
            "terutama di area kulit wajah. Mendeteksi ini bisa jadi keunggulan arsitektur model.",
        ),
        (
            "T - Terikat Waktu",
            "Analisis berbasis data tahun 2020-2026. Pola artefak GAN bisa berubah "
            "seiring evolusi teknologi deepfake.",
        ),
    ],
}

# Halaman navigasi dengan emoji agar lebih mudah dipindai secara visual
PAGES = [
    "🏠  Overview",
    "⚖️  BQ1 : Keseimbangan Kelas",
    "☀️  BQ2 : Brightness & Contrast",
    "🎨  BQ3 : Channel RGB",
    "🔍  Data Quality",
    "📋  Kesimpulan",
]

st.set_page_config(
    page_title="DeepShield Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helpers dasar
# ---------------------------------------------------------------------------

def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def esc(value: object) -> str:
    return html_lib.escape("" if value is None else str(value))


def norm_col(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [norm_col(c) for c in out.columns]
    return out


def load_css() -> None:
    if STYLE_PATH.exists():
        html_block(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>")


def image_uri(path: Path) -> str:
    if not path.exists():
        return ""
    suffix = path.suffix.lower().replace(".", "") or "png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{suffix};base64,{encoded}"


def fmt_num(value: object, digits: int = 0) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(number) or math.isinf(number):
        return "-"
    txt = f"{number:,.0f}" if digits == 0 else f"{number:,.{digits}f}"
    return txt.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(value: object, digits: int = 2) -> str:
    return f"{fmt_num(value, digits)}%"


def fmt_ratio(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(number) or math.isinf(number):
        return "-"
    return f"{fmt_num(number, digits)} : 1"


def fmt_pvalue(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(number) or math.isinf(number):
        return "-"
    if number == 0:
        return "0"
    if abs(number) < 0.001:
        return f"{number:.2e}"
    return fmt_num(number, 4)


def first_existing_col(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    columns = set(df.columns)
    for alias in aliases:
        key = norm_col(alias)
        if key in columns:
            return key
    return None


def to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_class_value(value: object) -> str:
    txt = str(value).strip().lower()
    if txt in {"real", "true", "authentic", "original"}:
        return "Real"
    if txt in {"fake", "deepfake", "false", "ai"}:
        return "Fake"
    return str(value).strip().title()


def normalize_split_value(value: object) -> str:
    txt = str(value).strip().lower()
    mapping = {
        "train": "Train", "training": "Train",
        "val": "Validation", "valid": "Validation", "validation": "Validation",
        "test": "Test", "testing": "Test",
    }
    return mapping.get(txt, str(value).strip().title())


def standardize_common(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_columns(df)
    alias_map = {
        "class":                   ["class", "label", "target", "kelas"],
        "split":                   ["split", "subset", "data_split"],
        "count":                   ["count", "jumlah", "n", "n_images", "total"],
        "percentage":              ["percentage", "percent", "pct", "persentase"],
        "percentage_within_split": ["percentage_within_split", "pct_within_split", "percentage_split"],
        "file_size_kb":            ["file_size_kb", "size_kb", "file_size"],
        "extension":               ["extension", "ext", "file_extension"],
        "width":                   ["width", "image_width"],
        "height":                  ["height", "image_height"],
        "brightness_mean":         ["brightness_mean", "brightness", "mean_brightness"],
        "contrast_std":            ["contrast_std", "contrast", "contrast_mean"],
        "r_mean":                  ["r_mean", "red_mean"],
        "g_mean":                  ["g_mean", "green_mean"],
        "b_mean":                  ["b_mean", "blue_mean"],
    }
    for canonical, aliases in alias_map.items():
        col = first_existing_col(out, aliases)
        if col and canonical not in out.columns:
            out[canonical] = out[col]
    if "class" in out.columns:
        out["class"] = out["class"].map(normalize_class_value)
    if "split" in out.columns:
        out["split"] = out["split"].map(normalize_split_value)
    numeric_cols = [
        "count", "percentage", "percentage_within_split", "file_size_kb", "width", "height",
        "aspect_ratio", "megapixels", "brightness_mean", "brightness_median", "brightness_std",
        "brightness_min", "brightness_max", "contrast_std", "contrast_mean", "contrast_median",
        "contrast_min", "contrast_max", "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std",
        "fake_mean", "real_mean", "fake_std", "real_std", "fake_median", "real_median",
        "mean_difference_fake_minus_real", "cohen_d", "p_value", "n_images",
    ]
    out = to_numeric(out, numeric_cols)
    if "is_corrupt" in out.columns:
        out["is_corrupt"] = (
            out["is_corrupt"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
        )
    return out


@st.cache_data(show_spinner="Membaca file CSV DeepShield...")
def load_data() -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    data: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []
    for key, filename in DATA_FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            data[key] = pd.DataFrame()
            missing.append(filename)
            continue
        try:
            data[key] = standardize_common(pd.read_csv(path))
        except Exception as exc:
            data[key] = pd.DataFrame()
            missing.append(f"{filename} (gagal dibaca: {exc})")
    return data, missing


def balance_metrics(balance_df: pd.DataFrame) -> Dict[str, float]:
    if balance_df.empty or "metric" not in balance_df.columns or "value" not in balance_df.columns:
        return {}
    metrics: Dict[str, float] = {}
    for _, row in balance_df.iterrows():
        key = str(row.get("metric", "")).strip().lower()
        try:
            metrics[key] = float(row.get("value"))
        except (TypeError, ValueError):
            continue
    return metrics


def global_class_distribution(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = data.get("class_distribution", pd.DataFrame()).copy()
    if not df.empty and {"class", "count"}.issubset(df.columns):
        df = df[[c for c in ["class", "count", "percentage"] if c in df.columns]].copy()
    else:
        features = data.get("image_features", pd.DataFrame())
        if features.empty or "class" not in features.columns:
            return pd.DataFrame({"class": CLASS_ORDER, "count": [0, 0], "percentage": [0.0, 0.0]})
        df = features.groupby("class", as_index=False).size().rename(columns={"size": "count"})
        total = df["count"].sum()
        df["percentage"] = np.where(total > 0, df["count"] / total * 100, 0.0)
    df["class"] = df["class"].map(normalize_class_value)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    if "percentage" not in df.columns:
        total = df["count"].sum()
        df["percentage"] = np.where(total > 0, df["count"] / total * 100, 0.0)
    df = df[df["class"].isin(CLASS_ORDER)]
    full = pd.DataFrame({"class": CLASS_ORDER})
    df = full.merge(df, on="class", how="left").fillna({"count": 0, "percentage": 0.0})
    df["count"] = df["count"].astype(int)
    return df


def split_distribution(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = data.get("split_class", pd.DataFrame()).copy()
    if df.empty or not {"split", "class", "count"}.issubset(df.columns):
        features = data.get("image_features", pd.DataFrame())
        if features.empty or not {"split", "class"}.issubset(features.columns):
            return pd.DataFrame(columns=["split", "class", "count", "percentage_within_split"])
        df = features.groupby(["split", "class"], as_index=False).size().rename(columns={"size": "count"})
    df["split"] = df["split"].map(normalize_split_value)
    df["class"] = df["class"].map(normalize_class_value)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    if "percentage_within_split" not in df.columns:
        totals = df.groupby("split")["count"].transform("sum")
        df["percentage_within_split"] = np.where(totals > 0, df["count"] / totals * 100, 0.0)
    df = df[df["split"].isin(SPLIT_ORDER) & df["class"].isin(CLASS_ORDER)]
    return df


def get_global_numbers(data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    class_df = global_class_distribution(data)
    counts = {row["class"]: float(row["count"]) for _, row in class_df.iterrows()}
    fake  = counts.get("Fake", 0.0)
    real  = counts.get("Real", 0.0)
    total = fake + real
    metrics = balance_metrics(data.get("balance", pd.DataFrame()))
    if metrics:
        fake  = metrics.get("fake_images",  fake)
        real  = metrics.get("real_images",  real)
        total = metrics.get("total_images", total)
    majority = max(fake, real)
    minority = min(fake, real) if min(fake, real) > 0 else 0
    ratio    = metrics.get("majority_to_minority_ratio", majority / minority if minority else np.nan)
    diff     = metrics.get("absolute_difference", abs(real - fake))
    diff_pct = metrics.get("absolute_difference_percentage", diff / total * 100 if total else 0.0)
    fake_pct = metrics.get("fake_percentage", fake / total * 100 if total else 0.0)
    real_pct = metrics.get("real_percentage", real / total * 100 if total else 0.0)
    return {
        "year":     metrics.get("dataset_year", DATASET_YEAR),
        "total":    total,
        "Fake":     fake,
        "Real":     real,
        "fake_pct": fake_pct,
        "real_pct": real_pct,
        "ratio":    ratio,
        "diff":     diff,
        "diff_pct": diff_pct,
    }


def apply_filters(df: pd.DataFrame, splits: Sequence[str], classes: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "split" in out.columns and splits:
        out = out[out["split"].isin(splits)]
    if "class" in out.columns and classes:
        out = out[out["class"].isin(classes)]
    return out


def sample_for_chart(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    n = int(max(1, n))
    if len(df) <= n:
        return df.copy()
    if "class" in df.columns and df["class"].nunique() > 1:
        pieces = []
        per_class = max(1, n // df["class"].nunique())
        for _, group in df.groupby("class"):
            pieces.append(group.sample(n=min(len(group), per_class), random_state=42))
        sampled = pd.concat(pieces, ignore_index=True)
        if len(sampled) < n:
            remaining = df.drop(sampled.index, errors="ignore")
            if len(remaining) > 0:
                extra = remaining.sample(n=min(len(remaining), n - len(sampled)), random_state=43)
                sampled = pd.concat([sampled, extra], ignore_index=True)
        return sampled.sample(frac=1, random_state=44).reset_index(drop=True)
    return df.sample(n=n, random_state=42).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Komponen UI
# ---------------------------------------------------------------------------

def hero(kicker: str, title: str, gradient: str, description: str) -> None:
    """Hero card tanpa badge."""
    html_block(
        f"""
        <div class="hero-card">
            <div class="hero-kicker">{esc(kicker)}</div>
            <h1>{esc(title)} <span class="gradient-text">{esc(gradient)}</span></h1>
            <p>{esc(description)}</p>
        </div>
        """
    )


def metric_card(label: str, value: str, help_text: str, icon: str = "•", tone: str = "purple") -> None:
    tone = tone if tone in {"cyan", "magenta", "green", "orange", "purple", "red"} else "purple"
    html_block(
        f"""
        <div class="metric-card {tone}">
            <div class="metric-icon">{esc(icon)}</div>
            <div class="metric-label">{esc(label)}</div>
            <div class="metric-value">{esc(value)}</div>
            <div class="metric-help">{esc(help_text)}</div>
        </div>
        """
    )


def section_title(title: str, subtitle: str = "", badge: str = "") -> None:
    badge_html = f"<span class='section-badge'>{esc(badge)}</span>" if badge else ""
    html_block(
        f"""
        <div class="section-title">
            <h2>{esc(title)}</h2>
            <p>{esc(subtitle)}{badge_html}</p>
        </div>
        """
    )


def question_card(question: str) -> None:
    html_block(
        f"""
        <div class="question-card">
            <div class="label">Pertanyaan Bisnis</div>
            <p>{esc(question)}</p>
        </div>
        """
    )


def note_card(title: str, text: str) -> None:
    html_block(
        f"""
        <div class="note-card">
            <div class="label">{esc(title)}</div>
            <p>{esc(text)}</p>
        </div>
        """
    )


def smart_table(rows: Sequence[Tuple[str, str]]) -> None:
    row_html = "".join(
        f"<tr><td>{esc(dim)}</td><td>{esc(desc)}</td></tr>"
        for dim, desc in rows
    )
    html_block(
        f"""
        <div class="smart-card">
            <table class="smart-table">
                <thead><tr><th>Dimensi SMART</th><th>Penjelasan</th></tr></thead>
                <tbody>{row_html}</tbody>
            </table>
        </div>
        """
    )


def decision_card(title: str, headline: str, bullets: Sequence[str], tone: str = "info") -> None:
    bullet_html = "".join(f"<li>{esc(item)}</li>" for item in bullets)
    tone = tone if tone in {"ok", "warn", "info", "magenta"} else "info"
    html_block(
        f"""
        <div class="decision-card {tone}">
            <div class="decision-title">{esc(title)}</div>
            <div class="decision-headline">{esc(headline)}</div>
            <ul>{bullet_html}</ul>
        </div>
        """
    )


def footer() -> None:
    html_block(
        """
        <div class="footer-card">
            <span>DeepShield &bull; CC26-PSU284 &bull; Streamlit Dashboard &bull; EDA, Visualization, and Explanatory Analysis Deepfake Images</span>
        </div>
        """
    )


def sidebar_ui(data: Dict[str, pd.DataFrame]) -> Tuple[str, List[str], List[str], int]:
    logo      = image_uri(LOGO_PATH)
    logo_html = (
        f"<img src='{logo}' class='brand-logo' alt='DeepShield logo'>"
        if logo else
        "<div class='brand-logo-fallback'>DS</div>"
    )
    with st.sidebar:
        html_block(
            f"""
            <div class="brand-card">
                <div class="brand-top">
                    {logo_html}
                    <div class="brand-title">DeepShield</div>
                </div>
                <div class="brand-subtitle">
                    Sistem Deteksi Deepfake untuk Menjaga Kepercayaan Masyarakat di Era Informasi Digital
                </div>
            </div>
            """
        )

        html_block("<div class='sidebar-label'>Navigasi</div>")
        page = st.radio("Navigasi", PAGES, label_visibility="collapsed")

        html_block("<div class='sidebar-divider'></div>")
        html_block("<div class='sidebar-label big'>Filter Grafik Distribusi</div>")

        features      = data.get("image_features", pd.DataFrame())
        split_options = [
            s for s in SPLIT_ORDER
            if not features.empty
            and s in set(features.get("split", pd.Series(dtype=str)).dropna().unique())
        ]
        class_options = [
            c for c in CLASS_ORDER
            if not features.empty
            and c in set(features.get("class", pd.Series(dtype=str)).dropna().unique())
        ]
        if not split_options:
            split_options = SPLIT_ORDER
        if not class_options:
            class_options = CLASS_ORDER

        selected_splits   = st.multiselect("Split",  split_options,  default=split_options)
        selected_classes  = st.multiselect("Class",  class_options,  default=class_options)
        if not selected_splits:
            selected_splits = split_options
        if not selected_classes:
            selected_classes = class_options

        max_rows      = int(len(features)) if not features.empty else 5000
        upper         = max(1000, min(50000, max_rows))
        default_sample = min(12000, upper)
        sample_rows   = st.slider(
            "Maksimum sampel untuk grafik distribusi piksel/RGB",
            min_value=1000,
            max_value=upper,
            value=default_sample,
            step=1000,
            help="Slider ini hanya memengaruhi histogram dan boxplot berbasis baris, bukan metrik atau kesimpulan global.",
        )
        html_block(
            """
            <div class="sidebar-note">
                <strong>Catatan:</strong> filter di atas hanya memengaruhi grafik distribusi dan
                preview data. Angka keputusan BQ menggunakan data penuh 2020-2026 agar konsisten.
            </div>
            """
        )
    return page, list(selected_splits), list(selected_classes), int(sample_rows)


# ---------------------------------------------------------------------------
# Helpers plot
# ---------------------------------------------------------------------------

def apply_plot_theme(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7, 1, 18, 0.35)",
        font=dict(color=COLORS["text"], family="Inter, sans-serif", size=12),
        margin=dict(l=40, r=24, t=48, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor="#17002E",
            bordercolor="rgba(139,92,246,0.55)",
            font_color=COLORS["text"],
        ),
    )
    ax_common = dict(
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.12)",
        linecolor="rgba(255,255,255,0.14)",
        tickfont=dict(color=COLORS["muted"]),
        title_font=dict(color=COLORS["text"]),
    )
    fig.update_xaxes(**ax_common)
    fig.update_yaxes(**ax_common)
    return fig


def empty_figure(title: str = "Data tidak tersedia") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=title, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=16, color=COLORS["muted"]),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_plot_theme(fig, height=320)


def chart_class_distribution(class_df: pd.DataFrame, title: str = "Distribusi Kelas Fake vs Real") -> go.Figure:
    if class_df.empty:
        return empty_figure()
    df = class_df.copy()
    df["label"] = df.apply(
        lambda r: f"{fmt_num(r['count'])}<br>{fmt_pct(r['percentage'], 3)}", axis=1
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["class"], y=df["count"],
        text=df["label"], textposition="outside",
        marker=dict(
            color=[CLASS_COLORS.get(c, COLORS["purple"]) for c in df["class"]],
            line=dict(color="rgba(255,255,255,0.18)", width=1),
        ),
        opacity=0.88,
        hovertemplate="Kelas=%{x}<br>Jumlah=%{y:,}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title="Kelas", yaxis_title="Jumlah gambar", showlegend=False)
    fig.update_yaxes(range=[0, max(df["count"].max() * 1.18, 1)])
    return apply_plot_theme(fig, height=390)


def chart_donut_class(class_df: pd.DataFrame, title: str = "Proporsi Dataset") -> go.Figure:
    if class_df.empty:
        return empty_figure()
    fig = go.Figure(data=[go.Pie(
        labels=class_df["class"],
        values=class_df["count"],
        hole=0.62,
        marker=dict(
            colors=[CLASS_COLORS.get(c, COLORS["purple"]) for c in class_df["class"]],
            line=dict(color="rgba(255,255,255,0.18)", width=1),
        ),
        texttemplate="%{percent:.3%}<br>%{label}",
        hovertemplate="%{label}<br>%{value:,} gambar<br>%{percent:.3%}<extra></extra>",
        sort=False,
    )])
    fig.update_layout(
        title=title,
        annotations=[dict(
            text="20-26", x=0.5, y=0.5,
            font_size=18, showarrow=False, font_color=COLORS["muted"],
        )],
    )
    return apply_plot_theme(fig, height=390)


def chart_split_distribution(split_df: pd.DataFrame, title: str = "Distribusi Kelas per Split") -> go.Figure:
    if split_df.empty:
        return empty_figure()
    df  = split_df.copy()
    fig = go.Figure()
    for cls in CLASS_ORDER:
        part = df[df["class"] == cls].set_index("split").reindex(SPLIT_ORDER).reset_index()
        fig.add_trace(go.Bar(
            x=part["split"], y=part["count"].fillna(0),
            name=cls, marker_color=CLASS_COLORS[cls], opacity=0.86,
            hovertemplate="Split=%{x}<br>Jumlah=%{y:,}<extra></extra>",
        ))
    fig.update_layout(title=title, barmode="group", xaxis_title="Split", yaxis_title="Jumlah gambar")
    return apply_plot_theme(fig, height=390)


def chart_histogram(df: pd.DataFrame, column: str, title: str, x_title: str, nbins: int = 60) -> go.Figure:
    if df.empty or column not in df.columns:
        return empty_figure(f"Kolom {column} tidak tersedia")
    fig = go.Figure()
    for cls in CLASS_ORDER:
        part = df[df.get("class") == cls]
        if part.empty:
            continue
        fig.add_trace(go.Histogram(
            x=part[column].dropna(), name=cls,
            marker_color=CLASS_COLORS[cls], opacity=0.58,
            nbinsx=nbins, histnorm="probability density",
            hovertemplate=f"{cls}<br>{x_title}=%{{x:.2f}}<br>Density=%{{y:.4f}}<extra></extra>",
        ))
    if not fig.data:
        return empty_figure("Tidak ada data sesuai filter")
    fig.update_layout(title=title, barmode="overlay", xaxis_title=x_title, yaxis_title="Density")
    return apply_plot_theme(fig, height=410)


def chart_boxplot(df: pd.DataFrame, column: str, title: str, y_title: str) -> go.Figure:
    if df.empty or column not in df.columns:
        return empty_figure(f"Kolom {column} tidak tersedia")
    fig = go.Figure()
    for cls in CLASS_ORDER:
        part = df[df.get("class") == cls]
        if part.empty:
            continue
        fig.add_trace(go.Box(
            y=part[column].dropna(), name=cls,
            marker_color=CLASS_COLORS[cls], line_color=CLASS_COLORS[cls],
            boxmean=True, opacity=0.72,
            hovertemplate=f"{cls}<br>{y_title}=%{{y:.2f}}<extra></extra>",
        ))
    if not fig.data:
        return empty_figure("Tidak ada data sesuai filter")
    fig.update_layout(title=title, yaxis_title=y_title, xaxis_title="Kelas")
    return apply_plot_theme(fig, height=370)


def chart_rgb_bar(rgb_stats: pd.DataFrame, title: str = "Rata-rata Channel RGB") -> go.Figure:
    if rgb_stats.empty:
        return empty_figure()
    fig = go.Figure()
    for cls in CLASS_ORDER:
        row = rgb_stats[rgb_stats.get("class") == cls]
        if row.empty:
            continue
        values = [
            float(row.iloc[0].get("r_mean", np.nan)),
            float(row.iloc[0].get("g_mean", np.nan)),
            float(row.iloc[0].get("b_mean", np.nan)),
        ]
        fig.add_trace(go.Bar(
            x=CHANNEL_ORDER, y=values, name=cls,
            marker_color=CLASS_COLORS[cls],
            text=[fmt_num(v, 2) for v in values], textposition="outside",
            opacity=0.86,
            hovertemplate=f"{cls}<br>Channel=%{{x}}<br>Mean=%{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(title=title, barmode="group", xaxis_title="Channel", yaxis_title="Mean pixel value")
    fig.update_yaxes(range=[0, 140])
    return apply_plot_theme(fig, height=405)


def chart_channel_distribution(df: pd.DataFrame, channel_col: str, title: str) -> go.Figure:
    return chart_histogram(df, channel_col, title, "Nilai intensitas channel", nbins=60)


def chart_resolution(df: pd.DataFrame) -> go.Figure:
    if df.empty or not {"width", "height"}.issubset(df.columns):
        return empty_figure("Resolusi tidak tersedia")
    temp = df.dropna(subset=["width", "height"]).copy()
    if temp.empty:
        return empty_figure("Resolusi tidak tersedia")
    temp["resolution"] = (
        temp["width"].round(0).astype(int).astype(str) + "x"
        + temp["height"].round(0).astype(int).astype(str)
    )
    counts = temp["resolution"].value_counts().head(8).reset_index()
    counts.columns = ["resolution", "count"]
    fig = go.Figure(go.Bar(
        x=counts["resolution"], y=counts["count"],
        marker_color=COLORS["purple"], opacity=0.78,
    ))
    fig.update_layout(title="Top Resolusi Gambar", xaxis_title="Resolusi", yaxis_title="Jumlah gambar", showlegend=False)
    return apply_plot_theme(fig, height=370)


def chart_extension(df: pd.DataFrame) -> go.Figure:
    if df.empty or "extension" not in df.columns:
        return empty_figure("Ekstensi file tidak tersedia")
    counts = (
        df["extension"].fillna("unknown").astype(str).str.lower()
        .value_counts().head(8).reset_index()
    )
    counts.columns = ["extension", "count"]
    fig = go.Figure(go.Bar(
        x=counts["extension"], y=counts["count"],
        marker_color=COLORS["real"], opacity=0.78,
    ))
    fig.update_layout(title="Distribusi Ekstensi File", xaxis_title="Ekstensi", yaxis_title="Jumlah file", showlegend=False)
    return apply_plot_theme(fig, height=350)


# ---------------------------------------------------------------------------
# Helpers tabel
# ---------------------------------------------------------------------------

def display_table(df: pd.DataFrame, height: Optional[int] = None) -> None:
    if df.empty:
        st.info("Tabel belum tersedia karena file CSV terkait kosong atau tidak ditemukan.")
        return
    st.dataframe(df, use_container_width=True, hide_index=True, height=height)


def bq2_stats_table(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    stats = data.get("pixel_stats", pd.DataFrame()).copy()
    if stats.empty:
        return pd.DataFrame()
    cols = [c for c in [
        "class", "n_images", "brightness_mean", "brightness_median", "brightness_std",
        "contrast_mean", "contrast_median", "contrast_std",
    ] if c in stats.columns]
    out = stats[cols].copy().rename(columns={
        "class": "Kelas", "n_images": "Jumlah Gambar",
        "brightness_mean": "Brightness Mean", "brightness_median": "Brightness Median",
        "brightness_std": "Brightness Std", "contrast_mean": "Contrast Mean",
        "contrast_median": "Contrast Median", "contrast_std": "Contrast Std",
    })
    for col in out.columns:
        if col != "Kelas":
            out[col] = out[col].map(lambda x: fmt_num(x, 2) if pd.notna(x) else "-")
    return out


def bq2_test_table(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    test = data.get("pixel_test", pd.DataFrame()).copy()
    if test.empty:
        return pd.DataFrame()
    cols = [c for c in [
        "metric", "fake_mean", "real_mean", "mean_difference_fake_minus_real",
        "cohen_d", "effect_size_label", "p_value", "significance",
    ] if c in test.columns]
    out = test[cols].copy().rename(columns={
        "metric": "Metrik", "fake_mean": "Fake Mean", "real_mean": "Real Mean",
        "mean_difference_fake_minus_real": "Selisih Fake - Real",
        "cohen_d": "Cohen's d", "effect_size_label": "Effect Size",
        "p_value": "p-value", "significance": "Signifikansi",
    })
    for col in ["Fake Mean", "Real Mean", "Selisih Fake - Real", "Cohen's d"]:
        if col in out.columns:
            out[col] = out[col].map(
                lambda x: fmt_num(x, 4 if col == "Cohen's d" else 2) if pd.notna(x) else "-"
            )
    if "p-value" in out.columns:
        out["p-value"] = out["p-value"].map(fmt_pvalue)
    return out


def rgb_diff_table(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    diff = data.get("rgb_diff", pd.DataFrame()).copy()
    if diff.empty:
        return pd.DataFrame()
    cols = [c for c in [
        "channel", "fake_mean", "real_mean", "mean_difference_fake_minus_real",
        "cohen_d", "effect_size_label", "p_value", "significance",
    ] if c in diff.columns]
    out = diff[cols].copy().rename(columns={
        "channel": "Channel", "fake_mean": "Fake Mean", "real_mean": "Real Mean",
        "mean_difference_fake_minus_real": "Selisih Fake - Real",
        "cohen_d": "Cohen's d", "effect_size_label": "Effect Size",
        "p_value": "p-value", "significance": "Signifikansi",
    })
    for col in ["Fake Mean", "Real Mean", "Selisih Fake - Real", "Cohen's d"]:
        if col in out.columns:
            out[col] = out[col].map(
                lambda x: fmt_num(x, 4 if col == "Cohen's d" else 2) if pd.notna(x) else "-"
            )
    if "p-value" in out.columns:
        out["p-value"] = out["p-value"].map(fmt_pvalue)
    return out


def split_table_for_display(split_df: pd.DataFrame) -> pd.DataFrame:
    if split_df.empty:
        return pd.DataFrame()
    out = split_df.copy()
    out = out[[c for c in ["split", "class", "count", "percentage_within_split"] if c in out.columns]]
    out = out.rename(columns={
        "split": "Split", "class": "Kelas",
        "count": "Jumlah", "percentage_within_split": "Persentase dalam Split",
    })
    if "Jumlah" in out.columns:
        out["Jumlah"] = out["Jumlah"].map(lambda x: fmt_num(x, 0))
    if "Persentase dalam Split" in out.columns:
        out["Persentase dalam Split"] = out["Persentase dalam Split"].map(lambda x: fmt_pct(x, 3))
    return out


# ---------------------------------------------------------------------------
# Keputusan analitis (dari Notebook EDA terbaru, 2020-2026)
# ---------------------------------------------------------------------------

def bq1_decision(numbers: Dict[str, float]) -> Tuple[str, List[str], str]:
    ratio    = numbers.get("ratio",    np.nan)
    diff     = numbers.get("diff",     np.nan)
    diff_pct = numbers.get("diff_pct", np.nan)
    if pd.notna(ratio) and ratio <= 1.5:
        headline = "Rasio di bawah 1,5:1, namun split Train perlu ditangani dengan class weight."
        tone = "ok"
    else:
        headline = "Ketimpangan cukup besar. Terapkan class weight atau undersampling pada split Train."
        tone = "warn"
    bullets = [
        f"Rasio kelas Fake : Real = {fmt_ratio(ratio, 4)} secara keseluruhan (batas kritis lebih dari 1,5:1).",
        f"Selisih = {fmt_num(diff)} gambar ({fmt_pct(diff_pct, 3)} dari total), terkonsentrasi di split Train.",
        "Langkah pertama: berikan class weight pada kelas Real saat melatih model.",
        "Pantau F1-score atau AUC-ROC, jangan hanya mengandalkan akurasi.",
    ]
    return headline, bullets, tone


def bq2_decision(data: Dict[str, pd.DataFrame]) -> Tuple[str, List[str], str]:
    test = data.get("pixel_test", pd.DataFrame())
    if test.empty:
        return (
            "Statistik brightness dan contrast belum tersedia.",
            ["Tambahkan pixel_statistical_test.csv untuk keputusan yang lebih kuat."],
            "warn",
        )
    d_vals    = pd.to_numeric(test.get("cohen_d", pd.Series(dtype=float)), errors="coerce").abs()
    max_d     = float(d_vals.max()) if len(d_vals) else np.nan
    bright_row   = test[test.get("metric", pd.Series(dtype=str)).astype(str).str.contains("brightness", case=False, na=False)]
    contrast_row = test[test.get("metric", pd.Series(dtype=str)).astype(str).str.contains("contrast",  case=False, na=False)]
    bright_d   = float(bright_row["cohen_d"].iloc[0])   if not bright_row.empty   and "cohen_d" in bright_row.columns   else np.nan
    contrast_d = float(contrast_row["cohen_d"].iloc[0]) if not contrast_row.empty and "cohen_d" in contrast_row.columns else np.nan
    if pd.notna(max_d) and max_d > 0.5:
        headline = "Perbedaan praktis cukup besar. Normalisasi dan ColorJitter sangat disarankan."
        tone = "warn"
    else:
        headline = "Perbedaan brightness dan contrast sangat kecil. Normalisasi gambar lebih penting dari ColorJitter."
        tone = "info"
    bullets = [
        f"Brightness Cohen's d = {fmt_num(bright_d, 4)}; contrast Cohen's d = {fmt_num(contrast_d, 4)}, keduanya sangat kecil.",
        "Nilai p yang kecil muncul karena jumlah sampel sangat besar, bukan karena polanya kuat.",
        "Terapkan normalisasi gambar terlebih dahulu. ColorJitter boleh ditambahkan sebagai variasi data.",
    ]
    return headline, bullets, tone


def bq3_decision(data: Dict[str, pd.DataFrame]) -> Tuple[str, List[str], str]:
    diff = data.get("rgb_diff", pd.DataFrame())
    if diff.empty or "cohen_d" not in diff.columns:
        return (
            "Statistik RGB belum tersedia.",
            ["Tambahkan rgb_channel_difference_summary.csv untuk keputusan arsitektur."],
            "warn",
        )
    temp     = diff.copy()
    temp["abs_d"] = pd.to_numeric(temp["cohen_d"], errors="coerce").abs()
    strongest = temp.sort_values("abs_d", ascending=False).head(1)
    if strongest.empty:
        return "Sinyal channel RGB belum dapat dihitung.", ["Periksa kolom cohen_d pada file RGB."], "warn"
    channel = str(strongest.iloc[0].get("channel", "-"))
    d       = float(strongest.iloc[0].get("cohen_d", np.nan))
    if abs(d) > 0.2:
        headline = f"Channel {channel} memiliki sinyal relevan. Channel Attention layak diprioritaskan."
        tone = "warn"
    else:
        headline = f"Channel {channel} paling berbeda antar kelas, namun sinyalnya masih sangat kecil."
        tone = "info"
    bullets = [
        f"Channel terkuat = {channel}; Cohen's d = {fmt_num(d, 4)}, masuk kategori efek sangat kecil.",
        "Gambar Real selalu lebih terang dari Fake di semua channel, pola ini konsisten tapi belum cukup kuat.",
        "Uji Channel Attention (SE-Net/CBAM) sebagai eksperimen lanjutan setelah baseline stabil.",
    ]
    return headline, bullets, tone


# ---------------------------------------------------------------------------
# Halaman
# ---------------------------------------------------------------------------

def overview_page(data: Dict[str, pd.DataFrame], filtered: pd.DataFrame, sample_rows: int) -> None:
    numbers  = get_global_numbers(data)
    class_df = global_class_distribution(data)
    split_df = split_distribution(data)
    rgb_stats = data.get("rgb_stats", pd.DataFrame())
    sample   = sample_for_chart(filtered, sample_rows)

    hero(
        "Overview Dataset 2020-2026",
        "DeepShield",
        "Dashboard",
        "Eksplorasi dataset gambar Fake dan Real tahun 2020-2026 yang mencakup keseimbangan kelas, "
        "karakteristik piksel, dan distribusi warna sebagai dasar keputusan sebelum tahap pemodelan.",
    )

    # KPI cards
    cols = st.columns(5)
    with cols[0]:
        metric_card("Total gambar", fmt_num(numbers["total"]), "Dataset penuh 2020-2026", "🧮", "cyan")
    with cols[1]:
        metric_card("Fake", fmt_num(numbers["Fake"]), fmt_pct(numbers["fake_pct"], 3), "🎭", "magenta")
    with cols[2]:
        metric_card("Real", fmt_num(numbers["Real"]), fmt_pct(numbers["real_pct"], 3), "✅", "green")
    with cols[3]:
        metric_card("Rasio kelas", fmt_ratio(numbers["ratio"], 4), "Majority vs minority", "⚖️", "purple")
    corrupt = 0
    features = data.get("image_features", pd.DataFrame())
    if not features.empty and "is_corrupt" in features.columns:
        corrupt = int(features["is_corrupt"].sum())
    with cols[4]:
        metric_card("Corrupt image", fmt_num(corrupt), "Hasil parsing file", "⚠️", "orange" if corrupt else "green")

    # Grafik distribusi global
    section_title(
        "Fakta Pertama: Seberapa banyak Fake vs Real?",
        "Grafik ini menggunakan data penuh 2020-2026, tidak berubah meski filter sidebar aktif.",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(chart_class_distribution(class_df), use_container_width=True)
    with col_b:
        st.plotly_chart(chart_split_distribution(split_df), use_container_width=True)

    # Eksplorasi visual (filtered)
    section_title(
        "Jelajahi sebaran brightness dan warna",
        "Grafik ini mengikuti pilihan Split dan Class yang dipilih di sidebar.",
    )
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(
            chart_histogram(sample, "brightness_mean", "Distribusi Brightness", "Brightness mean"),
            use_container_width=True,
        )
    with col_d:
        st.plotly_chart(chart_rgb_bar(rgb_stats, "Rata-rata Channel RGB"), use_container_width=True)

    # Kesimpulan di bagian bawah
    section_title(
        "Apa yang bisa kita simpulkan?",
        "Tiga poin utama dari analisis ini, satu untuk setiap pertanyaan bisnis.",
    )
    bq1_head, bq1_bullets, bq1_tone = bq1_decision(numbers)
    bq2_head, bq2_bullets, bq2_tone = bq2_decision(data)
    bq3_head, bq3_bullets, bq3_tone = bq3_decision(data)
    c1, c2, c3 = st.columns(3)
    with c1:
        decision_card("Keseimbangan Kelas", bq1_head, bq1_bullets, bq1_tone)
    with c2:
        decision_card("Brightness & Contrast", bq2_head, bq2_bullets, bq2_tone)
    with c3:
        decision_card("Channel RGB", bq3_head, bq3_bullets, bq3_tone)

    footer()


def bq1_page(data: Dict[str, pd.DataFrame]) -> None:
    numbers  = get_global_numbers(data)
    class_df = global_class_distribution(data)
    split_df = split_distribution(data)
    headline, bullets, tone = bq1_decision(numbers)

    hero(
        "Business Question 1",
        "BQ1 •",
        "Keseimbangan Kelas",
        "Sebelum melatih model deteksi, penting untuk tahu apakah data latihannya sudah seimbang. "
        "Halaman ini menghitung proporsi kelas dan memberi rekomendasi berdasarkan hasilnya.",
    )
    question_card(BQ_QUESTIONS["BQ1"])
    smart_table(SMART_ROWS["BQ1"])

    section_title(
        "Seberapa seimbang datanya?",
        "Dihitung dari file distribusi kelas dan ringkasan balance. Data global, tidak terpengaruh filter.",
    )
    cols = st.columns(5)
    with cols[0]:
        metric_card("Fake",    fmt_num(numbers["Fake"]), fmt_pct(numbers["fake_pct"], 3), "🎭", "magenta")
    with cols[1]:
        metric_card("Real",    fmt_num(numbers["Real"]), fmt_pct(numbers["real_pct"], 3), "✅", "cyan")
    with cols[2]:
        metric_card("Rasio",   fmt_ratio(numbers["ratio"], 4), "Batas kritis lebih dari 1,5:1", "⚖️", "purple")
    with cols[3]:
        metric_card("Selisih", fmt_num(numbers["diff"]), fmt_pct(numbers["diff_pct"], 3), "↔️", "orange")
    with cols[4]:
        metric_card(
            "Keputusan",
            "Hampir Seimbang" if numbers["ratio"] <= 1.5 else "Perlu Handling",
            "Perhatikan split Train" if numbers["ratio"] <= 1.5 else "Terapkan class weight",
            "🧠",
            "green" if numbers["ratio"] <= 1.5 else "orange",
        )

    col_a, col_b = st.columns([1.15, 0.85])
    with col_a:
        st.plotly_chart(chart_class_distribution(class_df), use_container_width=True)
    with col_b:
        st.plotly_chart(chart_donut_class(class_df), use_container_width=True)

    col_c, col_d = st.columns([1.1, 0.9])
    with col_c:
        st.plotly_chart(
            chart_split_distribution(split_df, "Distribusi Kelas pada Train, Validation, Test"),
            use_container_width=True,
        )
    with col_d:
        section_title("Perbandingan per subset data", "Persentase dihitung di dalam masing-masing split.")
        display_table(split_table_for_display(split_df), height=260)

    section_title("Apa yang perlu dilakukan selanjutnya?", "Langkah konkret berdasarkan kondisi dataset saat ini.")
    c1, c2 = st.columns([1.15, 0.85])
    with c1:
        decision_card("Keputusan BQ1", headline, bullets, tone)
    with c2:
        note_card(
            "Mengapa ini penting?",
            "Data latih yang lebih banyak Fake membuat model cenderung menebak Fake terlalu sering. "
            "Memberikan class weight pada kelas Real adalah cara paling mudah untuk mengoreksi ini "
            "tanpa mengubah jumlah data. Split Validation dan Test sudah mendekati 50:50, "
            "pertahankan apa adanya.",
        )
    footer()


def bq2_page(data: Dict[str, pd.DataFrame], filtered: pd.DataFrame, sample_rows: int) -> None:
    test   = data.get("pixel_test", pd.DataFrame())
    sample = sample_for_chart(filtered, sample_rows)
    headline, bullets, tone = bq2_decision(data)

    hero(
        "Business Question 2",
        "BQ2 •",
        "Brightness & Contrast",
        "Apakah gambar Fake cenderung lebih gelap atau lebih terang dari gambar Real? "
        "Halaman ini mengukur perbedaan kecerahan dan kontras serta dampaknya pada strategi augmentasi.",
    )
    question_card(BQ_QUESTIONS["BQ2"])
    smart_table(SMART_ROWS["BQ2"])

    bright_row   = test[test.get("metric", pd.Series(dtype=str)).astype(str).str.contains("brightness", case=False, na=False)] if not test.empty else pd.DataFrame()
    contrast_row = test[test.get("metric", pd.Series(dtype=str)).astype(str).str.contains("contrast",  case=False, na=False)] if not test.empty else pd.DataFrame()
    fake_brightness = float(bright_row["fake_mean"].iloc[0])   if not bright_row.empty   and "fake_mean" in bright_row.columns   else np.nan
    real_brightness = float(bright_row["real_mean"].iloc[0])   if not bright_row.empty   and "real_mean" in bright_row.columns   else np.nan
    bright_d   = float(bright_row["cohen_d"].iloc[0])   if not bright_row.empty   and "cohen_d" in bright_row.columns   else np.nan
    contrast_d = float(contrast_row["cohen_d"].iloc[0]) if not contrast_row.empty and "cohen_d" in contrast_row.columns else np.nan
    bright_p   = float(bright_row["p_value"].iloc[0])   if not bright_row.empty   and "p_value" in bright_row.columns   else np.nan
    contrast_p = float(contrast_row["p_value"].iloc[0]) if not contrast_row.empty and "p_value" in contrast_row.columns else np.nan

    section_title(
        "Seberapa berbeda kecerahannya?",
        "Dihitung dari data global 2020-2026. Tidak berubah meski filter sidebar aktif.",
    )
    cols = st.columns(4)
    with cols[0]:
        metric_card("Brightness Fake", fmt_num(fake_brightness, 2), "Rata-rata intensitas piksel", "☀️", "magenta")
    with cols[1]:
        metric_card("Brightness Real", fmt_num(real_brightness, 2), "Rata-rata intensitas piksel", "🌤️", "cyan")
    with cols[2]:
        metric_card("Cohen's d Brightness", fmt_num(bright_d, 4),   "Effect size brightness", "📏", "purple")
    with cols[3]:
        metric_card("Cohen's d Contrast",   fmt_num(contrast_d, 4), "Effect size contrast",   "🔎", "green")

    section_title(
        "Sebaran kecerahan dan kontras per kelas",
        "Grafik ini mengikuti pilihan Split dan Class yang dipilih di sidebar.",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            chart_histogram(sample, "brightness_mean", "Histogram Brightness per Kelas", "Brightness mean"),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            chart_histogram(sample, "contrast_std", "Histogram Contrast per Kelas", "Contrast std"),
            use_container_width=True,
        )
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(chart_boxplot(sample, "brightness_mean", "Boxplot Brightness", "Brightness mean"), use_container_width=True)
    with col_d:
        st.plotly_chart(chart_boxplot(sample, "contrast_std", "Boxplot Contrast", "Contrast std"), use_container_width=True)

    section_title(
        "Angka lengkap di balik grafik",
        "Sumber: pixel_brightness_contrast_statistics.csv dan pixel_statistical_test.csv.",
    )
    col_e, col_f = st.columns([1.1, 1])
    with col_e:
        display_table(bq2_stats_table(data), height=220)
    with col_f:
        display_table(bq2_test_table(data), height=220)

    col_g, col_h = st.columns([1.15, 0.85])
    with col_g:
        decision_card(
            "Keputusan BQ2", headline,
            bullets + [f"p-value brightness = {fmt_pvalue(bright_p)}; p-value contrast = {fmt_pvalue(contrast_p)}."],
            tone,
        )
    with col_h:
        note_card(
            "Apa artinya untuk pelatihan model?",
            "Perbedaan kecerahan terdeteksi secara statistik, tapi efeknya sangat kecil. "
            "Nilai p yang kecil ini bukan karena polanya kuat, melainkan karena jumlah sampelnya sangat besar. "
            "Lakukan normalisasi gambar sebagai prioritas. "
            "ColorJitter boleh ditambahkan sebagai variasi data, bukan sebagai respons darurat.",
        )
    footer()


def bq3_page(data: Dict[str, pd.DataFrame], filtered: pd.DataFrame, sample_rows: int) -> None:
    rgb_stats = data.get("rgb_stats", pd.DataFrame())
    rgb_diff  = data.get("rgb_diff",  pd.DataFrame())
    sample    = sample_for_chart(filtered, sample_rows)
    headline, bullets, tone = bq3_decision(data)

    hero(
        "Business Question 3",
        "BQ3 •",
        "Channel RGB",
        "Model deteksi deepfake bisa lebih akurat jika arsitekturnya tahu channel warna mana yang paling berbeda. "
        "Halaman ini mengukur perbedaan warna per channel dan implikasinya untuk desain model.",
    )
    question_card(BQ_QUESTIONS["BQ3"])
    smart_table(SMART_ROWS["BQ3"])

    strongest_channel = "-"
    strongest_d = np.nan
    if not rgb_diff.empty and "cohen_d" in rgb_diff.columns:
        temp = rgb_diff.copy()
        temp["abs_d"] = pd.to_numeric(temp["cohen_d"], errors="coerce").abs()
        row = temp.sort_values("abs_d", ascending=False).head(1)
        if not row.empty:
            strongest_channel = str(row.iloc[0].get("channel", "-"))
            strongest_d = float(row.iloc[0].get("cohen_d", np.nan))

    section_title(
        "Seberapa besar perbedaan warna antar kelas?",
        "Dihitung dari data global 2020-2026. Tidak berubah meski filter sidebar aktif.",
    )
    cols = st.columns(4)
    with cols[0]:
        metric_card("Channel terkuat", strongest_channel, "Berdasarkan nilai mutlak Cohen's d", "🎨", "cyan")
    with cols[1]:
        metric_card("Cohen's d",  fmt_num(strongest_d, 4), "Ambang relevan: lebih dari 0,2", "📏", "purple")
    with cols[2]:
        metric_card(
            "Status sinyal",
            "Lemah"   if pd.notna(strongest_d) and abs(strongest_d) < 0.2 else "Relevan",
            "Kekuatan sinyal channel",
            "🧪",
            "orange" if pd.notna(strongest_d) and abs(strongest_d) < 0.2 else "green",
        )
    with cols[3]:
        metric_card("Rekomendasi", "Opsional", "Uji Channel Attention", "🧠", "magenta")

    st.plotly_chart(chart_rgb_bar(rgb_stats, "Perbandingan Rata-rata Channel RGB"), use_container_width=True)

    section_title(
        "Intensitas warna per kelas",
        "Grafik ini mengikuti pilihan Split dan Class yang dipilih di sidebar.",
    )
    tab_r, tab_g, tab_b, tab_box, tab_table = st.tabs([
        "Channel R", "Channel G", "Channel B", "Boxplot RGB", "Tabel Statistik"
    ])
    with tab_r:
        st.plotly_chart(chart_channel_distribution(sample, "r_mean", "Distribusi Channel Red"),   use_container_width=True)
    with tab_g:
        st.plotly_chart(chart_channel_distribution(sample, "g_mean", "Distribusi Channel Green"), use_container_width=True)
    with tab_b:
        st.plotly_chart(chart_channel_distribution(sample, "b_mean", "Distribusi Channel Blue"),  use_container_width=True)
    with tab_box:
        col_r, col_g, col_b_ = st.columns(3)
        with col_r:
            st.plotly_chart(chart_boxplot(sample, "r_mean", "Boxplot Red",   "Red mean"),   use_container_width=True)
        with col_g:
            st.plotly_chart(chart_boxplot(sample, "g_mean", "Boxplot Green", "Green mean"), use_container_width=True)
        with col_b_:
            st.plotly_chart(chart_boxplot(sample, "b_mean", "Boxplot Blue",  "Blue mean"),  use_container_width=True)
    with tab_table:
        display_table(rgb_diff_table(data), height=260)

    col_a, col_b = st.columns([1.15, 0.85])
    with col_a:
        decision_card("Keputusan BQ3", headline, bullets, tone)
    with col_b:
        note_card(
            "Apa artinya untuk arsitektur model?",
            "Channel Green adalah yang paling informatif di dataset ini: gambar Real secara konsisten "
            "lebih terang dari Fake, dengan Green menunjukkan selisih terbesar. "
            "Meski efeknya masih sangat kecil, jadikan ini hipotesis eksperimen awal, bukan asumsi final.",
        )
    footer()


def data_quality_page(
    data: Dict[str, pd.DataFrame],
    filtered_features: pd.DataFrame,
    filtered_inventory: pd.DataFrame,
) -> None:
    hero(
        "Dataset Readiness",
        "Data Quality •",
        "Readiness Check",
        "Sebelum model dilatih, ada baiknya memeriksa kondisi fisik data: apakah ada file rusak, "
        "apakah resolusi sudah seragam, dan apakah formatnya konsisten.",
    )

    section_title(
        "Kondisi data berdasarkan filter aktif",
        "Bagian ini mengikuti pilihan Split dan Class di sidebar.",
    )

    df_feat = filtered_features.copy()
    df_inv  = filtered_inventory.copy()
    n_filtered = len(df_feat) if not df_feat.empty else len(df_inv)
    corrupt    = int(df_feat["is_corrupt"].sum()) if not df_feat.empty and "is_corrupt" in df_feat.columns else 0
    ext_count  = (
        df_inv["extension"].nunique() if not df_inv.empty and "extension" in df_inv.columns
        else (df_feat["extension"].nunique() if not df_feat.empty and "extension" in df_feat.columns else 0)
    )
    avg_size = (
        df_inv["file_size_kb"].mean() if not df_inv.empty and "file_size_kb" in df_inv.columns
        else (df_feat["file_size_kb"].mean() if not df_feat.empty and "file_size_kb" in df_feat.columns else np.nan)
    )
    dom_res = "-"
    if not df_feat.empty and {"width", "height"}.issubset(df_feat.columns):
        temp = df_feat.dropna(subset=["width", "height"]).copy()
        if not temp.empty:
            temp["resolution"] = (
                temp["width"].round(0).astype(int).astype(str)
                + "x"
                + temp["height"].round(0).astype(int).astype(str)
            )
            dom_res = str(temp["resolution"].value_counts().idxmax())

    cols = st.columns(5)
    with cols[0]:
        metric_card("Data terfilter",    fmt_num(n_filtered),       "Sesuai sidebar",               "🧮", "cyan")
    with cols[1]:
        metric_card("Corrupt image",     fmt_num(corrupt),          "Semakin kecil semakin baik",   "⚠️", "orange" if corrupt else "green")
    with cols[2]:
        metric_card("Format file",       fmt_num(ext_count),        "Jumlah ekstensi unik",         "🗂️", "purple")
    with cols[3]:
        metric_card("Resolusi dominan",  dom_res,                   "Berdasarkan lebar x tinggi",   "🖼️", "magenta")
    with cols[4]:
        metric_card("Avg size",          f"{fmt_num(avg_size, 2)} KB", "Rata-rata ukuran file",     "💾", "cyan")

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(chart_resolution(df_feat), use_container_width=True)
    with col_b:
        st.plotly_chart(chart_extension(df_inv if not df_inv.empty else df_feat), use_container_width=True)

    section_title("Lihat data secara langsung", "Preview dari image_features.csv setelah filter sidebar diterapkan.")
    preview_cols = [c for c in [
        "split", "class", "file_name", "file_size_kb", "extension",
        "width", "height", "brightness_mean", "contrast_std",
        "r_mean", "g_mean", "b_mean", "is_corrupt",
    ] if c in df_feat.columns]
    display_table(df_feat[preview_cols].head(50) if preview_cols else pd.DataFrame(), height=330)

    note_card(
        "Ringkasan kesiapan dataset",
        "Data sudah cukup lengkap untuk analisis awal: metadata, statistik piksel, dan nilai RGB tersedia. "
        "Sekitar 7,53% gambar memiliki lebar bukan 256 piksel sehingga perlu resize atau padding sebelum pelatihan. "
        "Pantau jumlah corrupt image sebelum memulai training.",
    )
    footer()


def conclusion_page(data: Dict[str, pd.DataFrame]) -> None:
    numbers = get_global_numbers(data)
    bq1_head, bq1_bullets, bq1_tone = bq1_decision(numbers)
    bq2_head, bq2_bullets, bq2_tone = bq2_decision(data)
    bq3_head, bq3_bullets, bq3_tone = bq3_decision(data)

    hero(
        "Final Insight",
        "Kesimpulan",
        "Dashboard",
        "Menerjemahkan hasil analisis dari pertanyaan bisnis menjadi saran dan langkah yang dilakukan oleh Tim AI Engineer dalam melatih model",
    )

    section_title(
        "Jawaban dari tiga pertanyaan bisnis",
        "Berbasis dataset penuh 2020-2026",
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        decision_card("Keseimbangan Kelas", bq1_head, bq1_bullets, bq1_tone)
    with col_b:
        decision_card("Brightness & Contrast", bq2_head, bq2_bullets, bq2_tone)
    with col_c:
        decision_card("Channel RGB", bq3_head, bq3_bullets, bq3_tone)

    section_title(
        "Saran bagi tim AI Engineer untuk pelatihan model",
    )
    col_d, col_e = st.columns(2)
    with col_d:
        decision_card(
            "Preprocessing dan pelatihan",
            "",
            [
                "Terapkan normalisasi dan augmentasi gambar.",
                "Pantau F1-score atau AUC-ROC, bukan hanya akurasi.",
            ],
            "ok",
        )
    with col_e:
        decision_card(
            "Eksperimen arsitektur model",
            "Channel Attention adalah eksperimen lanjutan yang menarik, bukan fitur wajib di baseline.",
            [
                "Channel Green memiliki perbedaan terbesar antar kelas, meski effect size-nya sangat kecil.",
                "Uji SE-Net atau CBAM setelah baseline model sudah stabil.",
                "Validasi ulang pada data baru karena pola artefak deepfake terus berkembang.",
            ],
            "info",
        )

    note_card(
        "Catatan penting",
        "Dashboard ini menyajikan hasil analisis data untuk mendukung keputusan teknis. "
        "Hasil deteksi deepfake tetap harus divalidasi dengan pengujian model nyata, bukan sekadar analisis data awal.",
    )
    footer()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    load_css()
    data, missing = load_data()

    page, selected_splits, selected_classes, sample_rows = sidebar_ui(data)

    if missing:
        st.warning("Beberapa file CSV tidak ditemukan atau gagal dibaca: " + ", ".join(missing))

    features  = data.get("image_features",  pd.DataFrame())
    inventory = data.get("image_inventory", pd.DataFrame())
    filtered_features  = apply_filters(features,  selected_splits, selected_classes)
    filtered_inventory = apply_filters(inventory, selected_splits, selected_classes)

    # Routing berbasis konten nama halaman (tahan terhadap prefix emoji)
    if "Overview" in page:
        overview_page(data, filtered_features, sample_rows)
    elif "Keseimbangan" in page:
        bq1_page(data)
    elif "Brightness" in page:
        bq2_page(data, filtered_features, sample_rows)
    elif "Channel RGB" in page:
        bq3_page(data, filtered_features, sample_rows)
    elif "Data Quality" in page:
        data_quality_page(data, filtered_features, filtered_inventory)
    elif "Kesimpulan" in page:
        conclusion_page(data)
    else:
        overview_page(data, filtered_features, sample_rows)


if __name__ == "__main__":
    main()
