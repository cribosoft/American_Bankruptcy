import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Configurare pagină
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

# CSS personalizat pentru design modern
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# Funcție pentru încărcare și procesare date
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

    print("Coloane după redenumire:", df.columns.tolist())
    print(df["Net_Profit_Margin"].describe(), 'net profit margin\n ')

    df = df[(df['Debt_to_Equity']>= -1000)&(df['Debt_to_Equity']<= 1000)]
    df = df[df['Current_Ratio'] < 10]
    df = df[(df['Net_Profit_Margin'] >= 0) & (df['Net_Profit_Margin'] <= 10)]



    # Înlocuire valori infinite cu NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


# Încărcare date
df = load_and_process_data()

print(df["Net_Profit_Margin"].describe(), 'net profit margin\n ')
print(df["Current_Ratio"].describe(), 'Current_Ratio\n ')
print(df["Debt_to_Equity"].describe(), 'Debt_to_Equity\n ')

# Sidebar - Navigare
st.sidebar.title(" Navigare")
page = st.sidebar.radio(
    "Selectează pagina:",
    [" Overview", " Analiza Comparativă", " Factori de Influență", " Model Predictiv"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Despre Dataset")
st.sidebar.info(f"""
**Total companii:** {df['company_name'].nunique()}  
**Total observații:** {len(df)}  
**Companii active:** {df['bankruptcy'].sum()}  
**Companii falimentare:** {(~df['bankruptcy']).sum()}
""")

# ============= PAGINA 1: OVERVIEW =============
if page == " Overview":
    st.title(" Analiza Indicatorilor Financiari - Faliment vs Prosperitate")
    st.markdown("### Situația companiilor americane: O privire de ansamblu")

    # Metrici principale
    col1, col2, col3, col4 = st.columns(4)

    total_companies = df['company_name'].nunique()
    alive_companies = df[df['bankruptcy'] == True]['company_name'].nunique()
    failed_companies = df[df['bankruptcy'] == False]['company_name'].nunique()
    bankruptcy_rate = (failed_companies / total_companies) * 100

    with col1:
        st.metric("Total Companii", f"{total_companies:,}")
    with col2:
        st.metric("Companii Active", f"{alive_companies:,}", delta="Prosperitate")
    with col3:
        st.metric("Companii Falimentare", f"{failed_companies:,}", delta="-Risc", delta_color="inverse")
    with col4:
        st.metric("Rata Faliment", f"{bankruptcy_rate:.1f}%")

    st.markdown("---")

    # Grafice principale
    col1, col2 = st.columns(2)

    with col1:
        # Pie chart - Distribuție status
        fig, ax = plt.subplots(figsize=(4, 2))
        status_counts = df.groupby('bankruptcy').size()
        colors = ['#ef4444', '#10b981']
        labels = ['Falimentare', 'Active']
        explode = (0.05, 0)# face pie chart 3D

        wedges, texts, autotexts = ax.pie(
            status_counts,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            explode=explode,
            startangle=250,
            textprops={'fontsize': 6, 'weight': 'bold'}
        )

        for autotext in autotexts:
            autotext.set_color('white')

        ax.set_title('Distribuția Companiilor: Active vs Falimentare', fontsize=8, weight='bold', pad=2)
        st.pyplot(fig)
        plt.close()

    with col2:
        # Bar chart - Falimente pe an
        bankruptcies_per_year = df[df['bankruptcy'] == False].groupby('year').size().reset_index(name='count')

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=bankruptcies_per_year,
            x='year',
            y='count',
            palette='viridis',
            ax=ax
        )
        ax.set_title('Evoluția Numărului de Falimente pe An', fontsize=14, weight='bold')
        ax.set_xlabel('Anul', weight='bold')
        ax.set_ylabel('Număr de Companii Falimentare', weight='bold')

        # Adaugă etichete pe bare
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points',
                        weight='bold')

        st.pyplot(fig)
        plt.close()

    # Întrebările de cercetare
    st.markdown("---")
    st.markdown("###  Obiective Analiza")

    col1, col2, col3 = st.columns(3)

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

    # Tabel statistici generale
    st.markdown("---")
    st.markdown("###  Statistici Generale Indicatori")

    stats_df = df.groupby('bankruptcy').agg({
        'Net_Profit_Margin': ['mean', 'median', 'std'],
        'Debt_to_Equity': ['mean', 'median', 'std'],
        'Current_Ratio': ['mean', 'median', 'std']
    }).round(2)

    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]
    stats_df.index = ['Falimentare', 'Active']

    st.dataframe(stats_df, use_container_width=True)

# ============= PAGINA 2: ANALIZA COMPARATIVĂ =============
elif page == " Analiza Comparativă":
    st.title(" Analiza Comparativă: Faliment vs Prosperitate")
    st.markdown("### Cum arată companiile falimentare vs active?")

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

    # Tabs pentru diferite vizualizări
    tab1, tab2, tab3 = st.tabs(["📊 Distribuții", "📦 Boxplots", "🏢 Top Companii"])

    with tab1:
        st.markdown("### Distribuția Indicatorilor Financiari")

        # Histograme pentru cei 3 indicatori
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

            # Current Ratio
            fig, ax = plt.subplots(figsize=(10, 5))
            new_df[new_df['bankruptcy'] == True]['Current_Ratio'].dropna().hist(
                bins=30, alpha=0.6, label='Active', color='#10b981', ax=ax
            )
            new_df[new_df['bankruptcy'] == False]['Current_Ratio'].dropna().hist(
                bins=30, alpha=0.6, label='Falimentare', color='#ef4444', ax=ax
            )
            ax.set_xlabel('Current Ratio', fontsize=12)
            ax.set_ylabel('Frecvență', fontsize=12)
            ax.set_title('Distribuția Lichidității Curente', fontsize=14, weight='bold')
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

            # Scatter plot
            fig, ax = plt.subplots(figsize=(10, 5))
            scatter_data = new_df.dropna(subset=['Net_Profit_Margin', 'Debt_to_Equity'])

            active = scatter_data[scatter_data['bankruptcy'] == True]
            failed = scatter_data[scatter_data['bankruptcy'] == False]

            ax.scatter(active['Debt_to_Equity'], active['Net_Profit_Margin'],
                       alpha=0.6, s=100, c='#10b981', label='Active', edgecolors='black', linewidth=0.5)
            ax.scatter(failed['Debt_to_Equity'], failed['Net_Profit_Margin'],
                       alpha=0.6, s=100, c='#ef4444', label='Falimentare', edgecolors='black', linewidth=0.5)

            ax.set_xlabel('Debt/Equity', fontsize=12)
            ax.set_ylabel('Marja Netă (%)', fontsize=12)
            ax.set_title('Relația dintre Indicatori', fontsize=14, weight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()

    with tab2:
        st.markdown("### Comparație Boxplots")

        # Boxplots pentru toți indicatorii
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Net Profit Margin
        data_to_plot = [
            new_df[new_df['bankruptcy'] == False]['Net_Profit_Margin'].dropna(),
            new_df[new_df['bankruptcy'] == True]['Net_Profit_Margin'].dropna()
        ]
        bp1 = axes[0].boxplot(data_to_plot, labels=['Falimentare', 'Active'], patch_artist=True)
        for patch, color in zip(bp1['boxes'], ['#ef4444', '#10b981']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0].set_ylabel('Marja Netă (%)', fontsize=12)
        axes[0].set_title('Marja Netă', fontsize=14, weight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # Debt to Equity
        data_to_plot = [
            new_df[new_df['bankruptcy'] == False]['Debt_to_Equity'].dropna(),
            new_df[new_df['bankruptcy'] == True]['Debt_to_Equity'].dropna()
        ]
        bp2 = axes[1].boxplot(data_to_plot, labels=['Falimentare', 'Active'], patch_artist=True)
        for patch, color in zip(bp2['boxes'], ['#ef4444', '#10b981']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_ylabel('Debt/Equity', fontsize=12)
        axes[1].set_title('Debt/Equity', fontsize=14, weight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        # Current Ratio
        data_to_plot = [
            new_df[new_df['bankruptcy'] == False]['Current_Ratio'].dropna(),
            new_df[new_df['bankruptcy'] == True]['Current_Ratio'].dropna()
        ]
        bp3 = axes[2].boxplot(data_to_plot, labels=['Falimentare', 'Active'], patch_artist=True)
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
                top_alive_display[
                    ['company_name', 'Net_Profit_Margin', 'Debt_to_Equity', 'Current_Ratio']].style.format({
                    'Net_Profit_Margin': '{:.2f}%',
                    'Debt_to_Equity': '{:.2f}',
                    'Current_Ratio': '{:.2f}'
                }).background_gradient(cmap='Greens', subset=['Net_Profit_Margin']),
                use_container_width=True,
                height=400
            )

        with col2:
            st.markdown("### 📉 Top 10 Companii Falimentare")
            top_failed_display = top_failed.sort_values('Net_Profit_Margin', ascending=True).head(10)
            st.dataframe(
                top_failed_display[
                    ['company_name', 'Net_Profit_Margin', 'Debt_to_Equity', 'Current_Ratio']].style.format({
                    'Net_Profit_Margin': '{:.2f}%',
                    'Debt_to_Equity': '{:.2f}',
                    'Current_Ratio': '{:.2f}'
                }).background_gradient(cmap='Reds', subset=['Net_Profit_Margin']),
                use_container_width=True,
                height=400
            )

# ============= PAGINA 3: FACTORI DE INFLUENȚĂ =============
elif page == " Factori de Influență":
    st.title(" Factori de Influență")
    st.markdown("### Ce factori contribuie cel mai mult la faliment sau stabilitate?")

    # Calculare corelații
    numeric_cols = ['Net_Profit_Margin', 'Debt_to_Equity', 'Current_Ratio',
                    'Total Assets', 'Total Revenue', 'EBITDA', 'Net Income']

    df_corr = df[numeric_cols + ['bankruptcy']].copy()
    df_corr['bankruptcy_numeric'] = df_corr['bankruptcy'].astype(int)

    correlation_matrix = df_corr[numeric_cols + ['bankruptcy_numeric']].corr()

    col1, col2 = st.columns([2, 1])

    with col1:
        # Heatmap corelații
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
        ax.set_title('Matricea de Corelații între Indicatori Financiari', fontsize=16, weight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("###  Interpretare Corelații")

        # Corelații cu bankruptcy
        bankruptcy_corr = correlation_matrix['bankruptcy_numeric'].drop('bankruptcy_numeric').sort_values(
            ascending=False)

        st.markdown("**Top Factori Pozitivi:**")
        for idx, val in bankruptcy_corr.head(3).items():
            st.success(f"**{idx}**: {val:.3f}")

        st.markdown("**Top Factori Negativi:**")
        for idx, val in bankruptcy_corr.tail(3).items():
            st.error(f"**{idx}**: {val:.3f}")

        st.markdown("---")
        st.info("""
        **Interpretare:**
        - Valori **pozitive** → cresc șansa de supraviețuire
        - Valori **negative** → cresc riscul de faliment
        - Valori apropiate de **0** → influență redusă
        """)

    # Analiza detaliată per indicator
    st.markdown("---")
    st.markdown("###  Analiza Detaliată Indicatori")

    indicator = st.selectbox(
        "Selectează indicatorul:",
        ['Net_Profit_Margin', 'Debt_to_Equity', 'Current_Ratio', 'Total Assets', 'Total Revenue']
    )

    col1, col2 = st.columns(2)

    with col1:
        # Violin plot
        fig, ax = plt.subplots(figsize=(10, 6))

        data_alive = df[df['bankruptcy'] == True][indicator].dropna()
        data_failed = df[df['bankruptcy'] == False][indicator].dropna()

        parts = ax.violinplot(
            [data_failed, data_alive],
            positions=[0, 1],
            showmeans=True,
            showextrema=True
        )

        for pc, color in zip(parts['bodies'], ['#ef4444', '#10b981']):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Falimentare', 'Active'])
        ax.set_ylabel(indicator, fontsize=12)
        ax.set_title(f'Distribuția {indicator}', fontsize=14, weight='bold')
        ax.grid(axis='y', alpha=0.3)

        st.pyplot(fig)
        plt.close()

    with col2:
        # Statistici descriptive
        st.markdown(f"####  Statistici {indicator}")
        stats_alive = df[df['bankruptcy'] == True][indicator].describe()
        stats_failed = df[df['bankruptcy'] == False][indicator].describe()

        stats_df = pd.DataFrame({
            'Active': stats_alive,
            'Falimentare': stats_failed,
            'Diferență': stats_alive - stats_failed
        })
        st.dataframe(
            stats_df.style.format('{:.2f}').background_gradient(cmap='coolwarm', axis=1),
            use_container_width=True
        )

# ============= PAGINA 4: MODEL PREDICTIV =============
elif page == " Model Predictiv":
    st.title(" Model Predictiv de Faliment")
    st.markdown("### Putem prezice ce se va întâmpla cu companiile?")

    st.info("""
    🚧 **Secțiune în dezvoltare**

    Aici va fi implementat modelul de machine learning pentru predicția falimentului.

    **Pași următori:**
    1. Pregătirea datelor (train/test split)
    2. Feature engineering
    3. Antrenarea modelului (ex: Logistic Regression, Random Forest, XGBoost)
    4. Evaluarea performanței (accuracy, precision, recall, F1-score)
    5. Vizualizarea rezultatelor (confusion matrix, ROC curve)
    """)

    # Placeholder pentru viitorul model
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy (estimat)", "85%", "+5%")
    with col2:
        st.metric("Precision (estimat)", "82%", "+3%")
    with col3:
        st.metric("Recall (estimat)", "88%", "+7%")

    st.markdown("---")
    st.markdown("###  Simulare Predicție")

    with st.form("prediction_form"):
        st.markdown("Introdu valorile pentru o companie nouă:")

        col1, col2, col3 = st.columns(3)

        with col1:
            npm = st.number_input("Net Profit Margin (%)", value=10.0, step=1.0)
        with col2:
            dte = st.number_input("Debt to Equity", value=1.5, step=0.1)
        with col3:
            cr = st.number_input("Current Ratio", value=2.0, step=0.1)

        submitted = st.form_submit_button(" Prezice Risc Faliment")

        if submitted:
            # Placeholder pentru predicție reală (bazat pe valori simple)
            # Scor de risc calculat simplu
            risk_score = 50

            if npm < 0:
                risk_score += 20
            elif npm > 10:
                risk_score -= 15

            if dte > 2:
                risk_score += 15
            elif dte < 1:
                risk_score -= 10

            if cr < 1.5:
                risk_score += 10
            elif cr > 2.5:
                risk_score -= 10

            risk_score = max(0, min(100, risk_score))

            st.markdown("---")
            st.markdown("###  Rezultat Predicție")

            col1, col2 = st.columns([1, 2])

            with col1:
                if risk_score < 30:
                    st.success(f"✅ **Risc Scăzut: {risk_score}%**")
                    st.markdown("Companie stabilă cu indicatori sănătoși")
                elif risk_score < 60:
                    st.warning(f"⚠️ **Risc Mediu: {risk_score}%**")
                    st.markdown("Necesită monitorizare atentă")
                else:
                    st.error(f"❌ **Risc Ridicat: {risk_score}%**")
                    st.markdown("Atenție mare la indicatori financiari!")

            with col2:
                # Grafic bară de risc
                fig, ax = plt.subplots(figsize=(10, 2))

                colors_map = {
                    'low': '#10b981',
                    'medium': '#f59e0b',
                    'high': '#ef4444'
                }

                if risk_score < 30:
                    color = colors_map['low']
                elif risk_score < 60:
                    color = colors_map['medium']
                else:
                    color = colors_map['high']

                ax.barh([0], [risk_score], color=color, height=0.5)
                ax.set_xlim(0, 100)
                ax.set_ylim(-0.5, 0.5)
                ax.set_xlabel('Scor Risc (%)', fontsize=12)
                ax.set_yticks([])
                ax.axvline(x=30, color='green', linestyle='--', alpha=0.5)
                ax.axvline(x=60, color='orange', linestyle='--', alpha=0.5)
                ax.grid(axis='x', alpha=0.3)

                st.pyplot(fig)
                plt.close()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    📊 Analiza Falimente Companii SUA | Developed with Streamlit & Python by Cristi Bogdan
</div>
""", unsafe_allow_html=True)