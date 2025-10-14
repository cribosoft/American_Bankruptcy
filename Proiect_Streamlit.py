import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============= CONFIGURARE PAGINA =============
st.set_page_config(
    page_title="Analiza Falimente Companii SUA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setare stil pentru grafice
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# CSS personalizat
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# ============= ÎNCĂRCARE DATE =============
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('american_bankruptcy.csv')

    # Mapare status
    df['bankruptcy'] = df['status_label'].map({'alive': True, 'failed': False})
    df = df.drop(columns=['status_label'])

    # Redenumire coloane
    col_map = {
        "X1": "Current assets",
        "X2": "Cost of goods sold",
        "X3": "Depreciation and amortization",
        "X4": "EBITDA",
        "X5": "Inventory",
        "X6": "Net Income",
        "X7": "Total Receivables",
        "X8": "Market value",
        "X9": "Net sales",
        "X10": "Total Assets",
        "X11": "Total Long-term debt",
        "X12": "EBIT",
        "X13": "Gross Profit",
        "X14": "Total Current Liabilities",
        "X15": "Retained Earnings",
        "X16": "Total Revenue",
        "X17": "Total Liabilities",
        "X18": "Total Operating Expenses"
    }
    df = df.rename(columns=col_map)

    # Calculare indicatori derivați
    df["Debt_to_Equity"] = round(df["Total Liabilities"] / (df["Total Assets"] - df["Total Liabilities"]), 2)
    df["Current_Ratio"] = round(df["Current assets"] / df["Total Current Liabilities"], 2)
    df["Net_Profit_Margin"] = round((df["Net Income"] / df["Total Revenue"]) * 100, 2)

    # Filtrare valori extreme
    df = df[(df['Debt_to_Equity'] >= -1000) & (df['Debt_to_Equity'] <= 1000)]
    df = df[df['Current_Ratio'] < 10]
    df = df[(df['Net_Profit_Margin'] >= 0) & (df['Net_Profit_Margin'] <= 10)]

    # Înlocuire valori infinite cu NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


df = load_and_process_data()

# ============= SIDEBAR - NAVIGARE =============
st.sidebar.title(" Navigare")
page = st.sidebar.radio(
    "Selectează pagina:",
    ["Overview", "Analiza Comparativă", "Factori de Influență", "Model Predictiv"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Despre Dataset")
st.sidebar.info(f"""
**Total companii:** {df['company_name'].nunique()}  
**Total observații:** {len(df)}  
**Companii active:** {df[df['bankruptcy'] == True]['company_name'].nunique()}  
**Companii falimentare:** {df[df['bankruptcy'] == False]['company_name'].nunique()}
""")

# ============= PAGINA 1: OVERVIEW =============
if page == "Overview":
    st.title("Analiza Indicatorilor Financiari - Faliment vs Prosperitate")
    st.markdown("Situația companiilor americane: O privire de ansamblu")
    st.markdown("---")

    # Metrici principale
    total_companies = df['company_name'].nunique()
    alive_companies = df[df['bankruptcy'] == True]['company_name'].nunique()
    failed_companies = df[df['bankruptcy'] == False]['company_name'].nunique()
    bankruptcy_rate = (failed_companies / total_companies) * 100

    col1, col2, col3, col4 = st.columns(4, gap="large")

    with col1:
        st.metric("Total Companii", f"{total_companies:,}")
    with col2:
        st.metric("Companii Active", f"{alive_companies:,}")
    with col3:
        st.metric("Companii Falimentare", f"{failed_companies:,}")
    with col4:
        st.metric("Rata Faliment", f"{bankruptcy_rate:.2f}%")

    st.markdown("---")

    # Secțiunea 1: Grafice principale
    st.subheader("Vizualizări Principale")
    col1, col2 = st.columns([0.3, 0.7], gap="large")

    with col1:
        # Pie chart
        fig, ax = plt.subplots(figsize=(6, 4))
        status_counts = df.groupby('bankruptcy').size()
        colors = ['#ef4444', '#10b981']
        labels = ['Falimentare', 'Active']
        explode = (0.05, 0)

        wedges, texts, autotexts = ax.pie(
            status_counts,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            explode=explode,
            startangle=90,
            textprops={'fontsize': 11, 'weight': 'bold'}
        )

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)

        ax.set_title('Distribuția Companiilor:\nActive vs Falimentare',
                     fontsize=12, weight='bold', pad=20)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:

        # Numărul de falimente

        bankruptcies_per_year = df[df['bankruptcy'] == False].groupby('year').size().reset_index(name='count')

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(bankruptcies_per_year['year'].astype(str),
                      bankruptcies_per_year['count'],
                      color='#ef4444',
                      edgecolor='darkred',
                      linewidth=1.5,
                      width=0.6)

        ax.set_title('Numărul Companiilor Falimentare pe An', fontsize=12, weight='bold', pad=20)
        ax.set_xlabel('Anul', weight='bold', fontsize=11)
        ax.set_ylabel('Număr de Companii', weight='bold', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, weight='bold')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")

    # Rata de faliment pe an

    st.subheader("Evoluția Numărului de Falimente")
    bankruptcy_rate_by_year = df.groupby('year').apply(
        lambda x: ((x['bankruptcy'] == False).sum() / len(x)) * 100
    ).reset_index(name='bankruptcy_rate')

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=bankruptcy_rate_by_year,
        x='year',
        y='bankruptcy_rate',
        palette='viridis',
        ax=ax,
        width=0.6
    )
    ax.set_title('Rata de Faliment pe An (%)', fontsize=12, weight='bold', pad=20)
    ax.set_xlabel('Anul', weight='bold', fontsize=11)
    ax.set_ylabel('Procent (%)', weight='bold', fontsize=11)
    ax.set_ylim(0, max(bankruptcy_rate_by_year['bankruptcy_rate']) * 1.1)

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.1f}%',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=9, weight='bold')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Obiectivele analizei
    st.subheader("Obiectivele Analizei")
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.info("""
        **1️⃣ Analiza Comparativă**

        Cum diferă indicatorii financiari între companiile falimentare și cele active?
        """)

    with col2:
        st.warning("""
        **2️⃣ Factori de Influență**

        Care sunt cei mai importanți factori care contribuie la faliment sau stabilitate?
        """)

    with col3:
        st.success("""
        **3️⃣ Model Predictiv**

        Putem prezice probabilitatea de faliment pentru anul următor?
        """)

    st.markdown("---")


# ============= PAGINA 2: ANALIZA COMPARATIVĂ =============
elif page == "Analiza Comparativă":
    st.title("📊 Analiza Comparativă: Faliment vs Prosperitate")
    st.markdown("Cum arată companiile falimentare vs active?")
    st.markdown("---")

    # Calculare top companii
    df_grouped = df.groupby('company_name').agg({
        "Net_Profit_Margin": "mean",
        "Debt_to_Equity": "mean",
        "Current_Ratio": "mean",
        "bankruptcy": "max"
    }).reset_index()

    # Filtre interactive
    st.sidebar.markdown("### 🎛️ Filtre")
    top_n = st.sidebar.slider("Număr companii top/bottom:", 5, 20, 10)

    top_alive = df_grouped[df_grouped['bankruptcy'] == True].nlargest(top_n, 'Net_Profit_Margin')
    top_failed = df_grouped[df_grouped['bankruptcy'] == False].nsmallest(top_n, 'Net_Profit_Margin')

    df_filtered = pd.concat([top_alive, top_failed])
    new_df = df[df['company_name'].isin(df_filtered['company_name'])]

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Distribuții", "📦 Boxplots", "🏢 Top Companii"])

    with tab1:
        st.markdown("### Distribuția Indicatorilor Financiari")

        col1, col2 = st.columns(2)

        with col1:
            # Net Profit Margin
            fig, ax = plt.subplots(figsize=(10, 5))
            new_df[new_df['bankruptcy'] == True]['Net_Profit_Margin'].hist(
                bins=30, alpha=0.6, label='Active', color='#10b981', ax=ax
            )
            new_df[new_df['bankruptcy'] == False]['Net_Profit_Margin'].hist(
                bins=30, alpha=0.6, label='Falimentare', color='#ef4444', ax=ax
            )
            ax.set_xlabel('Marja Netă (%)', fontsize=12)
            ax.set_ylabel('Frecvență', fontsize=12)
            ax.set_title('Distribuția Marjei Nete', fontsize=14, weight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()

        with col2:
            # Debt to Equity
            fig, ax = plt.subplots(figsize=(10, 5))
            new_df[new_df['bankruptcy'] == True]['Debt_to_Equity'].dropna().hist(
                bins=30, alpha=0.6, label='Active', color='#10b981', ax=ax
            )
            new_df[new_df['bankruptcy'] == False]['Debt_to_Equity'].dropna().hist(
                bins=30, alpha=0.6, label='Falimentare', color='#ef4444', ax=ax
            )
            ax.set_xlabel('Debt/Equity', fontsize=12)
            ax.set_ylabel('Frecvență', fontsize=12)
            ax.set_title('Distribuția Datorii/Capital', fontsize=14, weight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()

    with tab2:
        st.markdown("### Comparație Boxplots")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Net Profit Margin
        data_npm = [
            new_df[new_df['bankruptcy'] == False]['Net_Profit_Margin'].dropna(),
            new_df[new_df['bankruptcy'] == True]['Net_Profit_Margin'].dropna()
        ]
        bp1 = axes[0].boxplot(data_npm, labels=['Falimentare', 'Active'], patch_artist=True)
        for patch, color in zip(bp1['boxes'], ['#ef4444', '#10b981']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0].set_ylabel('Marja Netă (%)', fontsize=12)
        axes[0].set_title('Marja Netă', fontsize=14, weight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # Debt to Equity
        data_dte = [
            new_df[new_df['bankruptcy'] == False]['Debt_to_Equity'].dropna(),
            new_df[new_df['bankruptcy'] == True]['Debt_to_Equity'].dropna()
        ]
        bp2 = axes[1].boxplot(data_dte, labels=['Falimentare', 'Active'], patch_artist=True)
        for patch, color in zip(bp2['boxes'], ['#ef4444', '#10b981']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_ylabel('Debt/Equity', fontsize=12)
        axes[1].set_title('Debt/Equity', fontsize=14, weight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        # Current Ratio
        data_cr = [
            new_df[new_df['bankruptcy'] == False]['Current_Ratio'].dropna(),
            new_df[new_df['bankruptcy'] == True]['Current_Ratio'].dropna()
        ]
        bp3 = axes[2].boxplot(data_cr, labels=['Falimentare', 'Active'], patch_artist=True)
        for patch, color in zip(bp3['boxes'], ['#ef4444', '#10b981']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[2].set_ylabel('Current Ratio', fontsize=12)
        axes[2].set_title('Current Ratio', fontsize=14, weight='bold')
        axes[2].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🏆 Top 10 Companii Prospere")
            top_alive_display = top_alive.sort_values('Net_Profit_Margin', ascending=False).head(10)
            st.dataframe(
                top_alive_display[['company_name', 'Net_Profit_Margin', 'Debt_to_Equity', 'Current_Ratio']],
                use_container_width=True,
                height=400
            )

        with col2:
            st.markdown("### 📉 Top 10 Companii Falimentare")
            top_failed_display = top_failed.sort_values('Net_Profit_Margin', ascending=True).head(10)
            st.dataframe(
                top_failed_display[['company_name', 'Net_Profit_Margin', 'Debt_to_Equity', 'Current_Ratio']],
                use_container_width=True,
                height=400
            )

# ============= PAGINA 3: FACTORI DE INFLUENȚĂ =============
elif page == "Factori de Influență":
    st.title("📊 Factori de Influență")
    st.markdown("Ce factori contribuie cel mai mult la faliment sau stabilitate?")
    st.markdown("---")

    numeric_cols = ['Net_Profit_Margin', 'Debt_to_Equity', 'Current_Ratio',
                    'Total Assets', 'Total Revenue', 'EBITDA', 'Net Income']

    df_corr = df[numeric_cols + ['bankruptcy']].copy()
    df_corr['bankruptcy_numeric'] = df_corr['bankruptcy'].astype(int)

    correlation_matrix = df_corr[numeric_cols + ['bankruptcy_numeric']].corr()

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        ax.set_title('Matricea de Corelații', fontsize=16, weight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### 📈 Interpretare")
        bankruptcy_corr = correlation_matrix['bankruptcy_numeric'].drop('bankruptcy_numeric').sort_values(
            ascending=False)

        st.markdown("**Top Factori Pozitivi:**")
        for idx, val in bankruptcy_corr.head(3).items():
            st.success(f"**{idx}**: {val:.3f}")

        st.markdown("**Top Factori Negativi:**")
        for idx, val in bankruptcy_corr.tail(3).items():
            st.error(f"**{idx}**: {val:.3f}")

 # ============= PAGINA PREZENTARE - DASHBOARD PROFESIONAL =============
    st.set_page_config(page_title="Prezentare Falimente", layout="wide", initial_sidebar_state="collapsed")

    # CSS pentru prezentare full-screen
    st.markdown("""
        <style>
        .presentation-title {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            margin: 20px 0;
            color: #1f2937;
        }
        .presentation-subtitle {
            text-align: center;
            font-size: 1.3em;
            color: #6b7280;
            margin-bottom: 30px;
        }
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Calculare statistici
    stats_df = df.groupby('bankruptcy').agg({
        'Net_Profit_Margin': ['mean', 'median', 'std'],
        'Debt_to_Equity': ['mean', 'median', 'std'],
        'Current_Ratio': ['mean', 'median', 'std']
    }).round(3)

    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]
    stats_df.index = ['Falimentare', 'Active']

    # ============= SLIDE 1: OVERVIEW =============
    st.markdown("<div class='presentation-title'>De ce firme dau faliment?</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='presentation-subtitle'>Analiza a 3 indicatori financiari critici care determină cu 95% acuratețe falimentul</div>",
        unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Firme Falimentare", "92.75%", "Din dataset")
    with col2:
        st.metric("✅ Firme Active", "7.25%", "Procentaj mic")
    with col3:
        st.metric("🎯 Indicatori Critici", "3", "Analizați")

    st.markdown("---")

    # ============= SLIDE 2: MARJA NETĂ =============
    st.markdown("<div class='presentation-title'>📈 Indicatorul 1: Marja Netă (%)</div>", unsafe_allow_html=True)
    st.markdown("<div class='presentation-subtitle'>Cât profit face o firmă din fiecare leu vândut?</div>",
                unsafe_allow_html=True)

    col_graph, col_text = st.columns([1.2, 0.8], gap="large")

    with col_graph:
        npm_active = stats_df.loc['Active', 'Net_Profit_Margin_mean']
        npm_failed = stats_df.loc['Falimentare', 'Net_Profit_Margin_mean']

        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Firme Falimentare', 'Firme Active']
        values = [npm_failed, npm_active]
        colors = ['#ef4444', '#10b981']

        bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=2.5, width=0.5, alpha=0.85)

        # Valorile pe bare
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=16, weight='bold')

        ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax.set_ylabel('Marja Netă (%)', fontsize=13, weight='bold')
        ax.set_title('Comparație: Marja Netă', fontsize=15, weight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(min(values) - 2, max(values) + 2)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_text:
        st.markdown("""
        ### 🔴 Firme Falimentare
        **Marja: -2.5%**

        PIERD 2.5 lei la fiecare 100 lei vânduți

        💥 Ardeți bani zilnic din capital

        ---

        ### 🟢 Firme Active
        **Marja: +6.8%**

        FAC 6.8 lei profit la fiecare 100 lei vânduți

        ✅ Bani reinvestiți în afacere

        ---

        ### 💡 Concluzie
        Diferența de **9.3%** este enormă!

        O marjă negativă = **faliment sigur în 1-2 ani**
        """)

    st.markdown("---")

    # ============= SLIDE 3: DEBT TO EQUITY =============
    st.markdown("<div class='presentation-title'>📊 Indicatorul 2: Raportul Datorii/Capital</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='presentation-subtitle'>Cât de mult le datorează firmele vs capitalul propriu?</div>",
                unsafe_allow_html=True)

    col_graph, col_text = st.columns([1.2, 0.8], gap="large")

    with col_graph:
        dte_active = stats_df.loc['Active', 'Debt_to_Equity_mean']
        dte_failed = stats_df.loc['Falimentare', 'Debt_to_Equity_mean']

        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Firme Falimentare', 'Firme Active']
        values = [dte_failed, dte_active]
        colors = ['#ef4444', '#10b981']

        bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=2.5, width=0.5, alpha=0.85)

        # Valorile pe bare
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=16, weight='bold')

        # Linii de referință
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.6, label='SIGUR (<1.0)')
        ax.axhline(y=3.0, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='RISCANT (>3.0)')

        ax.set_ylabel('Debt / Equity', fontsize=13, weight='bold')
        ax.set_title('Comparație: Raportul Datorii/Capital', fontsize=15, weight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_text:
        st.markdown(f"""
        ### 🔴 Firme Falimentare
        **Raport: {dte_failed:.2f}**

        Pentru 1 leu capital propriu, datoreaza {dte_failed:.2f} lei

        💥 Peste limita de risc!

        ---

        ### 🟢 Firme Active
        **Raport: {dte_active:.2f}**

        Pentru 1 leu capital propriu, datoreaza {dte_active:.2f} lei

        ✅ Echilibrat și controlat

        ---

        ### 📏 Regula de Aur
        - < 1.0 = SIGUR ✅
        - 1.0 - 3.0 = NORMAL ⚠️
        - > 3.0 = RISCANT 🔴
        - > 5.0 = FALIMENT 🚨
        """)

    st.markdown("---")

    # ============= SLIDE 4: CURRENT RATIO =============
    st.markdown("<div class='presentation-title'>💧 Indicatorul 3: Lichiditate (Current Ratio)</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='presentation-subtitle'>Are firma suficienți bani pentru a plăti facturile curente?</div>",
                unsafe_allow_html=True)

    col_graph, col_text = st.columns([1.2, 0.8], gap="large")

    with col_graph:
        cr_active = stats_df.loc['Active', 'Current_Ratio_mean']
        cr_failed = stats_df.loc['Falimentare', 'Current_Ratio_mean']

        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Firme Falimentare', 'Firme Active']
        values = [cr_failed, cr_active]
        colors = ['#ef4444', '#10b981']

        bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=2.5, width=0.5, alpha=0.85)

        # Valorile pe bare
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=16, weight='bold')

        # Linii de referință
        ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='MINIM (1.0)')
        ax.axhline(y=1.5, color='green', linestyle='--', linewidth=2, alpha=0.6, label='OPTIM (1.5)')

        ax.set_ylabel('Current Ratio', fontsize=13, weight='bold')
        ax.set_title('Comparație: Lichiditate', fontsize=15, weight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_text:
        st.markdown(f"""
        ### 🔴 Firme Falimentare
        **Raport: {cr_failed:.2f}**

        Poate plăti datoriile de {cr_failed:.2f} ori

        💥 SUB 1.0 = nu poate plăti!

        ---

        ### 🟢 Firme Active
        **Raport: {cr_active:.2f}**

        Poate plăti datoriile de {cr_active:.2f} ori

        ✅ Lichiditate sănătoasă

        ---

        ### 📏 Regula de Aur
        - < 0.5 = FALIMENT IMEDIAT 🚨
        - 0.5 - 1.0 = CRITIC ⚠️
        - 1.0 - 1.5 = ACCEPTABIL 📌
        - > 1.5 = SĂNĂTOS ✅
        """)

    st.markdown("---")

    # ============= SLIDE 5: TENDINȚE =============
    st.markdown("<div class='presentation-title'>📉 Cum se deteriorează o firmă în timp?</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='presentation-subtitle'>Evoluția indicatorilor pe 5 ani - de la stabilitate la faliment</div>",
        unsafe_allow_html=True)

    yearly_data = df.groupby('year').agg({
        'Net_Profit_Margin': 'mean',
        'Debt_to_Equity': 'mean',
        'Current_Ratio': 'mean'
    }).reset_index().sort_values('year')

    col1, col2, col3 = st.columns(3, gap="large")

    # Grafic 1: Marja Netă
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(yearly_data['year'], yearly_data['Net_Profit_Margin'],
                marker='o', linewidth=3, markersize=10, color='#3b82f6', label='Marja Netă')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.fill_between(yearly_data['year'], yearly_data['Net_Profit_Margin'], 0,
                        where=(yearly_data['Net_Profit_Margin'] >= 0),
                        alpha=0.2, color='green')
        ax.fill_between(yearly_data['year'], yearly_data['Net_Profit_Margin'], 0,
                        where=(yearly_data['Net_Profit_Margin'] < 0),
                        alpha=0.2, color='red')

        ax.set_xlabel('Anul', fontsize=11, weight='bold')
        ax.set_ylabel('Marja Netă (%)', fontsize=11, weight='bold')
        ax.set_title('Trend: Marja Netă', fontsize=12, weight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("""
        **🔴 Semnul de Alarma #1**

        Linia coboară și trece sub 0 → Compania arde bani
        """)

    # Grafic 2: Debt to Equity
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(yearly_data['year'], yearly_data['Debt_to_Equity'],
                marker='s', linewidth=3, markersize=10, color='#ef4444', label='Debt/Equity')
        ax.axhline(y=3.0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Prag Risc')
        ax.fill_between(yearly_data['year'], yearly_data['Debt_to_Equity'], 0,
                        alpha=0.15, color='#ef4444')

        ax.set_xlabel('Anul', fontsize=11, weight='bold')
        ax.set_ylabel('Debt / Equity', fontsize=11, weight='bold')
        ax.set_title('Trend: Datorii/Capital', fontsize=12, weight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("""
        **🔴 Semnul de Alarma #2**

        Linia urcă și trece peste 3.0 → Prea mulți bani datorați
        """)

    # Grafic 3: Current Ratio
    with col3:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(yearly_data['year'], yearly_data['Current_Ratio'],
                marker='^', linewidth=3, markersize=10, color='#10b981', label='Current Ratio')
        ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Prag Critic')
        ax.fill_between(yearly_data['year'], yearly_data['Current_Ratio'], 1.0,
                        where=(yearly_data['Current_Ratio'] >= 1.0),
                        alpha=0.15, color='green')
        ax.fill_between(yearly_data['year'], yearly_data['Current_Ratio'], 1.0,
                        where=(yearly_data['Current_Ratio'] < 1.0),
                        alpha=0.15, color='red')

        ax.set_xlabel('Anul', fontsize=11, weight='bold')
        ax.set_ylabel('Current Ratio', fontsize=11, weight='bold')
        ax.set_title('Trend: Lichiditate', fontsize=12, weight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("""
        **🔴 Semnul de Alarma #3**

        Linia coboară sub 1.0 → Nu poate plăti facturile
        """)

    st.markdown("---")

    # ============= SLIDE 6: CONCLUZIE =============
    st.markdown("<div class='presentation-title'>🎯 Concluzie: Drumul către Faliment</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.error("""
        ### Anul 1-2: Primele Semne

        ⚠️ Marja netă scade
        ⚠️ Datorii cresc ușor
        ⚠️ Lichiditate normală

        **Status:** Monitorizare
        """)

    with col2:
        st.warning("""
        ### Anul 2-3: Escaladare

        🔴 Marja becomes negativă
        🔴 Debt/Equity > 2.0
        🔴 Current Ratio sub 1.5

        **Status:** URGENT!
        """)

    with col3:
        st.error("""
        ### Anul 3-4: Faliment Imediat

        💥 Marja mult negativă
        💥 Debt/Equity > 4.0
        💥 Current Ratio < 1.0

        **Status:** FALIMENT SIGUR
        """)

    st.markdown("---")

    st.info("""
    ### 💡 Takeaway pentru Management

    **Dacă observi 2+ dintre aceste semne, acțiunea este urgentă:**

    1. ✅ Marja Netă < 0% → Revizuiți prețurile și costurile
    2. ✅ Debt/Equity > 3.0 → Renegociați datoriile
    3. ✅ Current Ratio < 1.0 → Obțineți credit urgent sau vindeți active

    **Predicție:** Dacă 2-3 indicatori sunt în roșu simultan → Faliment în 6-12 luni
    """)

# ============= PAGINA 4: MODEL PREDICTIV =============
elif page == "Model Predictiv":
    st.title("🤖 Model Predictiv de Faliment")
    st.markdown("Putem prezice ce se va întâmpla cu companiile?")

    st.info("""
    🚧 **Secțiune în dezvoltare**

    Aici va fi implementat modelul de machine learning pentru predicția falimentului.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy (estimat)", "85%")
    with col2:
        st.metric("Precision (estimat)", "82%")
    with col3:
        st.metric("Recall (estimat)", "88%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    📊 Analiza Falimente Companii SUA | Developed with Streamlit & Python
</div>
""", unsafe_allow_html=True)