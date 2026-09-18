from pathlib import Path
import hmac
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="ABCXYZ Sales Analysis",
    page_icon="📊",
    layout="wide",
)


def require_authentication():
    """Require the deployment-provided password before loading any data."""
    app_password = os.environ.get("ABCXYZ_APP_PASSWORD")
    if not app_password:
        try:
            app_password = st.secrets["ABCXYZ_APP_PASSWORD"]
        except (KeyError, FileNotFoundError):
            app_password = None
    if not app_password:
        st.error("Application password is not configured.")
        st.stop()

    if st.session_state.get("authenticated"):
        return

    st.title("ABCXYZ Sales Analysis")
    with st.form("login_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")

    if submitted:
        if hmac.compare_digest(password, app_password):
            st.session_state.authenticated = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


require_authentication()

BASE_DIR = Path(__file__).resolve().parent
ABCXYZ_ORDER = ["AX", "AY", "AZ", "AN", "BX", "BY", "BZ", "BN", "CX", "CY", "CZ", "CN"]


@st.cache_data(show_spinner="Loading sales data...")
def load_and_prepare_data():
    sales_files = sorted(BASE_DIR.glob("sales-regional-*.parquet"))
    master_path = BASE_DIR / "Master Artikel.XLSX"

    if not sales_files:
        raise FileNotFoundError("No sales-regional-*.parquet files were found.")
    if not master_path.exists():
        raise FileNotFoundError("Master Artikel.XLSX was not found.")

    sales_frames = [pd.read_parquet(path) for path in sales_files]
    df = pd.concat(sales_frames, ignore_index=True)
    df["ARTICLE"] = df["ARTICLE"].astype("string")
    df = df.drop(columns=[col for col in df.columns if col.startswith("GR_")], errors="ignore")

    master = pd.read_excel(master_path)
    master["Article"] = master["Article"].astype("string")
    df["ARTICLE"] = df["ARTICLE"].astype("string")
    master["Article"] = master["Article"].astype("string")
    master_indexed = master.set_index("Article")
    mappings = {
        "price": "Loc Amount",
        "ProdClass": "Product Classification",
        "lv3": "Lv3-Description",
        "lv4": "Lv4-Description",
        "generic": "Generic Code",
    }
    for output_col, source_col in mappings.items():
        df[output_col] = df["ARTICLE"].map(master_indexed[source_col])

    sales_cols = [
        col for col in df.columns
        if col.startswith("QTY_SALES_") and col.endswith("_2025")
    ]
    if not sales_cols:
        raise ValueError("No January-December 2025 QTY_SALES columns were found.")

    group_cols = [
        "ARTICLE", "generic", "SITE_ID", "SITE_NAME", "REGIONAL_AREA",
        "ProdClass", "lv3", "lv4",
    ]
    sales_by_article = (
        df.groupby(group_cols, sort=False, observed=True, dropna=False)[sales_cols]
        .sum()
        .reset_index()
    )

    prices = df[["ARTICLE", "price"]].drop_duplicates("ARTICLE")
    sales_by_article = sales_by_article.merge(prices, on="ARTICLE", how="left")

    sales_by_article[sales_cols] = sales_by_article[sales_cols].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)
    sales_by_article["price"] = pd.to_numeric(
        sales_by_article["price"], errors="coerce"
    ).fillna(0)

    monthly_qty = sales_by_article[sales_cols]
    sales_by_article["total_qty"] = monthly_qty.sum(axis=1)
    sales_by_article["avg_monthly_qty"] = monthly_qty.mean(axis=1)
    sales_by_article["total_value"] = sales_by_article["price"] * sales_by_article["total_qty"]
    sales_by_article["avg_monthly_value"] = (
        sales_by_article["price"] * sales_by_article["avg_monthly_qty"]
    )
    sales_by_article["cv"] = monthly_qty.std(axis=1, ddof=0).div(
        monthly_qty.mean(axis=1).replace(0, np.nan)
    ).fillna(0)

    return sales_by_article[
        [
            "ARTICLE", "generic", "SITE_ID", "SITE_NAME", "REGIONAL_AREA",
            "ProdClass", "lv3", "lv4",
            "price", "total_qty", "avg_monthly_qty", "total_value",
            "avg_monthly_value", "cv",
        ]
    ].copy()


def classify_articles(data, x_limit, y_limit):
    result = data.sort_values(
        ["SITE_ID", "ProdClass", "lv3", "lv4", "total_value"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)
    ranking_cols = ["REGIONAL_AREA", "SITE_ID", "ProdClass", "lv3", "lv4"]
    result["cum_value"] = result.groupby(
        ranking_cols, observed=True, dropna=False
    )["total_value"].cumsum()
    result["group_total_value"] = result.groupby(
        ranking_cols, observed=True, dropna=False
    )["total_value"].transform("sum")
    result["cum_pct"] = np.where(
        result["group_total_value"] != 0,
        result["cum_value"] / result["group_total_value"],
        0,
    )
    result["abc_class"] = np.select(
        [result["cum_pct"] <= 0.80, result["cum_pct"] <= 0.95],
        ["A", "B"],
        default="C",
    )
    result["xyz_class"] = np.select(
        [
            result["total_qty"] == 0,
            result["cv"] < x_limit,
            (result["cv"] > x_limit) & (result["cv"] < y_limit),
        ],
        ["N", "X", "Y"],
        default="Z",
    )
    result["abc_xyz"] = result["abc_class"] + result["xyz_class"]
    return result


def format_table(data):
    formatted = data.copy()
    for column in ["total_qty", "total_value", "sales_qty", "sales_value"]:
        if column in formatted:
            formatted[column] = formatted[column].map(lambda value: f"{value:,.0f}")
    if "cv" in formatted:
        formatted["cv"] = formatted["cv"].map(lambda value: f"{value:.3f}")
    return formatted

def format_numeric_columns(data, exclude_columns=None):
    formatted = data.copy()
    exclude_columns = set(exclude_columns or [])

    for column in formatted.columns:
        if (
            column not in exclude_columns
            and pd.api.types.is_numeric_dtype(formatted[column])
        ):
            formatted[column] = formatted[column].map(
                lambda value: f"{value:,.0f}"
            )

    return formatted


st.title("ABCXYZ Sales Analysis")
st.caption("2025 sales | ABC ranked by sales value | XYZ based on quantity CV")

try:
    base_data = load_and_prepare_data()
except Exception as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.header("Analysis Controls")
    with st.form("cv_controls"):
        x_limit = st.number_input(
            "X: CV below",
            min_value=0.0,
            max_value=10.0,
            value=0.5,
            step=0.05,
            format="%.2f",
        )
        y_limit = st.number_input(
            "Y: CV below",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.05,
            format="%.2f",
        )
        apply_filters = st.form_submit_button("Apply CV rules", type="primary")

    if y_limit <= x_limit:
        st.error("Y CV limit must be greater than X CV limit.")
        st.stop()

    st.divider()
    st.caption("Zero total quantity is always classified as N.")

classified = classify_articles(base_data, x_limit, y_limit)

with st.sidebar:
    region_options = sorted(
        classified["REGIONAL_AREA"].dropna().astype(str).unique().tolist()
    )
    selected_regions = st.multiselect(
        "Regional Area", region_options, default=region_options
    )
    site_options = sorted(classified["SITE_ID"].dropna().unique().tolist())
    selected_sites = st.multiselect("Site", site_options, default=site_options)
    prod_options = sorted(classified["ProdClass"].dropna().astype(str).unique().tolist())
    selected_prod = st.multiselect("Product Classification", prod_options, default=prod_options)
    abcxyz_options = [value for value in ABCXYZ_ORDER if value in classified["abc_xyz"].unique()]
    selected_classes = st.multiselect("ABCXYZ class", abcxyz_options, default=abcxyz_options)

filtered = classified[
    classified["REGIONAL_AREA"].astype(str).isin(selected_regions)
    & classified["SITE_ID"].isin(selected_sites)
    & classified["ProdClass"].astype(str).isin(selected_prod)
    & classified["abc_xyz"].isin(selected_classes)
].copy()

metric_columns = st.columns(4)
metric_columns[0].metric("Articles", f"{filtered['ARTICLE'].nunique():,}")
metric_columns[1].metric("Sales Qty", f"{filtered['total_qty'].sum():,.0f}")
metric_columns[2].metric("Sales Value", f"{filtered['total_value'].sum():,.0f}")
metric_columns[3].metric("Sites", f"{filtered['SITE_ID'].nunique():,}")

if filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

left_chart, right_chart = st.columns(2)
with left_chart:
    class_counts = filtered["abc_xyz"].value_counts().reindex(ABCXYZ_ORDER, fill_value=0).reset_index()
    class_counts.columns = ["abc_xyz", "count_article"]
    figure = px.bar(
        class_counts,
        x="abc_xyz",
        y="count_article",
        title="Article Count by ABCXYZ Class",
        labels={"abc_xyz": "Class", "count_article": "Articles"},
    )
    st.plotly_chart(figure, use_container_width=True)

with right_chart:
    value_by_abc = filtered.groupby("abc_class", as_index=False)["total_value"].sum()
    figure = px.bar(
        value_by_abc,
        x="abc_class",
        y="total_value",
        title="Sales Value by ABC Class",
        labels={"abc_class": "ABC Class", "total_value": "Sales Value"},
    )
    st.plotly_chart(figure, use_container_width=True)

st.subheader("ABCXYZ Summary")
summary = filtered.groupby("abc_xyz", as_index=False).agg(
    count_article=("ARTICLE", "nunique"),
    sales_qty=("total_qty", "sum"),
    sales_value=("total_value", "sum"),
)
summary["abc_xyz"] = pd.Categorical(summary["abc_xyz"], categories=ABCXYZ_ORDER, ordered=True)
summary = summary.sort_values("abc_xyz")
st.dataframe(format_table(summary), use_container_width=True, hide_index=True)

st.subheader("ABCXYZ Matrix")
matrix_groupby = filtered.groupby(["abc_class", "xyz_class"], observed=True).agg(
    sales_qty=("total_qty", "sum"),
    sales_value=("total_value", "sum"),
    count_generic=("generic", "nunique"),
)
qty_matrix = matrix_groupby["sales_qty"].unstack(fill_value=0).reindex(
    index=["A", "B", "C"], columns=["N", "X", "Y", "Z"], fill_value=0
)
value_matrix = matrix_groupby["sales_value"].unstack(fill_value=0).reindex(
    index=["A", "B", "C"], columns=["N", "X", "Y", "Z"], fill_value=0
)
generic_matrix = matrix_groupby["count_generic"].unstack(fill_value=0).reindex(
    index=["A", "B", "C"], columns=["N", "X", "Y", "Z"], fill_value=0
)

matrix_tabs = st.tabs(["Sales Qty", "Sales Value", "Count Generic"])
with matrix_tabs[0]:
    st.dataframe(format_table(qty_matrix.reset_index()), use_container_width=True, hide_index=True)
with matrix_tabs[1]:
    st.dataframe(format_table(value_matrix.reset_index()), use_container_width=True, hide_index=True)
with matrix_tabs[2]:
    st.dataframe(generic_matrix.reset_index(), use_container_width=True, hide_index=True)

st.subheader("Article Detail")
detail_columns = [
    "ARTICLE", "generic", "REGIONAL_AREA", "SITE_ID", "SITE_NAME",
    "ProdClass", "lv3", "lv4",
    "total_qty", "total_value", "cv", "abc_class", "xyz_class", "abc_xyz",
]
st.dataframe(format_table(filtered[detail_columns]), use_container_width=True, hide_index=True)

st.download_button(
    "Download filtered ABCXYZ detail",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="abcxyz_filtered_detail.csv",
    mime="text/csv",
)

# ABCXYZ tables by site
st.subheader("ABCXYZ by Site")

site_order = sorted(
    filtered["SITE_ID"].dropna().unique(),
    key=lambda value: str(value),
)

site_name_mapping = (
    classified[["SITE_ID", "SITE_NAME"]]
    .dropna(subset=["SITE_ID"])
    .drop_duplicates("SITE_ID")
    .set_index("SITE_ID")["SITE_NAME"]
    .to_dict()
)

site_labels = {
    site_id: f"{str(site_id).replace('.0', '')} - {site_name_mapping.get(site_id, '')}"
    for site_id in site_order
}

count_generic_by_site = filtered.pivot_table(
    index="abc_xyz",
    columns="SITE_ID",
    values="generic",
    aggfunc="nunique",
    fill_value=0,
).reindex(
    index=ABCXYZ_ORDER,
    columns=site_order,
    fill_value=0,
)

sales_value_by_site = filtered.pivot_table(
    index="abc_xyz",
    columns="SITE_ID",
    values="total_value",
    aggfunc="sum",
    fill_value=0,
).reindex(
    index=ABCXYZ_ORDER,
    columns=site_order,
    fill_value=0,
)

count_generic_by_site = count_generic_by_site.rename(columns=site_labels)
sales_value_by_site = sales_value_by_site.rename(columns=site_labels)

count_generic_by_site.index.name = "ABCXYZ RESULT"
sales_value_by_site.index.name = "ABCXYZ RESULT"

site_table_tabs = st.tabs(["Count Generic by Site", "Sales Value by Site"])

with site_table_tabs[0]:
    count_generic_display = format_numeric_columns(
        count_generic_by_site.reset_index(),
        exclude_columns=["ABCXYZ RESULT"],
    )

    st.dataframe(
        count_generic_display,
        use_container_width=True,
        hide_index=True,
    )

with site_table_tabs[1]:
    sales_value_display = format_numeric_columns(
        sales_value_by_site.reset_index(),
        exclude_columns=["ABCXYZ RESULT"],
    )

    st.dataframe(
        sales_value_display,
        use_container_width=True,
        hide_index=True,
    )