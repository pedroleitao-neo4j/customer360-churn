import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase

from neo4j_analysis import Neo4jAnalysis
from neo4j_viz.neo4j import from_neo4j, ColorSpace
from neo4j_viz import Layout

colors = {
    "Customer": "#F00B47",      # Sky Magenta (Active/Primary)
    "Location": "#00AEEF",      # Sky Cyan (Fresh/Secondary)
    "PaymentMethod": "#702082",  # Sky Deep Violet
    "Contract": "#002E5D",       # Sky Midnight Blue (Formal)
    "Service": "#616AB1",        # Sky Sky Blue/Periwinkle
    "Movie": "#E6007E",          # Sky Rose (Vivid Pink)
}

label_to_property = {
    "Customer": "customer_id",
    "Location": "city",
    "PaymentMethod": "payment_method",
    "Contract": "contract",
    "Service": "service_type",
    "Movie": "title",
}

NETWORK_GRAPH_HEIGHT = 560

st.set_page_config(page_title="Graph Analytics", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    [data-testid="stAppViewContainer"] :is(
        p, label, a, li, input, textarea, button,
        h1, h2, h3, h4, h5, h6
    ),
    [data-testid="stSidebar"] :is(
        p, label, a, li, input, textarea, button,
        h1, h2, h3, h4, h5, h6
    ) {
        font-family: 'Inter', sans-serif;
    }

    .material-icons,
    .material-icons-outlined,
    .material-symbols-outlined,
    .material-symbols-rounded,
    .material-symbols-sharp,
    [class*="material-symbols"] {
        font-family: 'Material Symbols Outlined', 'Material Icons' !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.sidebar.image("https://neo4j.bynder.com/files/6b78f1f3-0e18-500b-a742-ebf47dd5a82c?account_id=2BF7D1AD-4DB7-43DF-A1F1EE0C57549FC3&expiry=1773584731459&signature=2JLasEE5QEpdFEWgUjelNyAVltB%2Bxx%2FbqE32pVd5vydThENwppPI4874xgxZwIGQQ%2BUodfwpe4382PbNetxRAA%3D%3D&version=a0b1ec4b", width=180)
st.sidebar.caption("Customer 360 Graph Analytics")
st.sidebar.image("https://images.contentstack.io/v3/assets/blt4b099fa9cc3801a6/blt0bd785479b1140c2/Sky-spectrum-cmyk.png", width=150)
st.title("Subscriber & Network Analytics with Neo4j and Graph Data Science")
st.markdown("Moving from relational tables to Graph Data Science for actionable business insights.")

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
AURA_API_CLIENT_ID = os.getenv("AURA_API_CLIENT_ID")
AURA_API_CLIENT_SECRET = os.getenv("AURA_API_CLIENT_SECRET")
AURA_API_TEXT2CYPHER_ENDPOINT = os.getenv("AURA_API_TEXT2CYPHER_ENDPOINT")
AURA_TOKEN_URL = "https://api.neo4j.io/oauth/token"

@st.cache_data(show_spinner=False)
def get_aura_access_token(client_id: str, client_secret: str) -> str:
    response = requests.post(
        AURA_TOKEN_URL,
        data={"grant_type": "client_credentials"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        auth=(client_id, client_secret),
        timeout=15,
    )
    try:
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Failed to get Aura API access token: {response.status_code} {response.text}") from exc

    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("Aura API token response missing access_token")
    return token


def invoke_aura_agent(endpoint: str, bearer_token: str, prompt: str) -> dict:
    response = requests.post(
        endpoint,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {bearer_token}",
        },
        json={"input": prompt},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()

@st.cache_resource
def get_analysis_client():
    return Neo4jAnalysis(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)


@st.cache_data(show_spinner=False)
def load_kpi_summary():
    query = """
        MATCH (c:Customer)
        WITH count(c) AS total_customers,
             sum(CASE WHEN c.churn_label = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
                         avg(c.tenure_months) AS avg_tenure_months,
             avg(coalesce(toFloat(c.monthly_charges), 0.0) * coalesce(toFloat(c.tenure_months), 0.0)) AS avg_customer_lifetime_value
        CALL() {
            MATCH (c2:Customer)
            WHERE c2.louvain_community_id IS NOT NULL
            WITH c2.louvain_community_id AS community_id, count(*) AS community_size
            WHERE community_size >= 10
            RETURN count(*) AS unique_communities
        }
        OPTIONAL MATCH (:Customer)-[:WATCHED_MOVIE]->(:Movie)
        RETURN total_customers,
               churned_customers,
               round((toFloat(churned_customers) / CASE WHEN total_customers = 0 THEN 1 ELSE total_customers END) * 100, 2) AS churn_rate_pct,
               round(coalesce(avg_tenure_months, 0), 1) AS avg_tenure_months,
                             round(coalesce(avg_customer_lifetime_value, 0), 2) AS avg_customer_lifetime_value,
                             unique_communities,
                             count(*) AS total_movie_watches
    """
    return analysis.run_query_df(query)


def render_section_intro(what_this_shows: str, how_to_use: str, business_takeaway: str):
    intro_col1, intro_col2, intro_col3 = st.columns(3)
    intro_col1.info(f"**What this shows**\n\n{what_this_shows}")
    intro_col2.info(f"**How to use**\n\n{how_to_use}")
    intro_col3.info(f"**Business takeaway**\n\n{business_takeaway}")


def render_styled_table(df: pd.DataFrame):
    st.dataframe(df, hide_index=True, width='stretch')

analysis = get_analysis_client()

try:
    kpi_df = load_kpi_summary()
except Exception:
    kpi_df = pd.DataFrame()

if not kpi_df.empty:
    kpis = kpi_df.iloc[0]
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Customers", f"{int(kpis['total_customers']):,}", border=True)
    col2.metric("Churn Rate", f"{float(kpis['churn_rate_pct']):.2f}%", border=True)
    col3.metric("Avg Tenure (Months)", f"{float(kpis['avg_tenure_months']):.1f}", border=True)
    col4.metric("Movie Watches", f"{int(kpis['total_movie_watches']):,}", border=True)
    col5.metric("Avg CLV", f"${float(kpis['avg_customer_lifetime_value']):,.2f}", border=True)
    col6.metric("Unique Communities", f"{int(kpis['unique_communities']):,}", border=True)

st.divider()

section_options = [
    "What the Data Represents",
    "The Geography",
    "Understanding the Customer 360 Graph",
    "Similarity Networks",
    "High-Churn Service Combinations",
    "GDS: Churn Communities",
    "Geo-Spatial Outages",
    "Churn Heatmap",
    "Watched Movie Networks",
    "Movie Recommendations",
    "KNN Flight Risk Prediction",
    "Agent-Based Analysis",
]

selected_section = st.sidebar.radio("Navigation", section_options)

if selected_section == "What the Data Represents":
    st.header("What the Data Represents")
    render_section_intro(
        "How customer, product, geography, and movie behavior entities are represented in the graph.",
        "Read the schema summary first, then navigate to a section in the sidebar to explore one analysis path.",
        "A graph model makes cross-domain joins and relationship-based insights much faster to access.",
    )
    st.markdown("""
    This demo uses a [Kaggle customer churn dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) modeled as a **Customer 360** graph in Neo4j.
    We will explore how a graph-based approach allows us to easily traverse complex relationships between customers, their demographics, subscription details, and behaviors to uncover insights that are difficult to capture with traditional tabular methods.
    
    **Key Features:**
    - **Customer Nodes**: Contain demographic and subscription attributes, including a binary churn label.
    - **Location Nodes**: Represent the cities where customers reside, enabling geo-spatial analysis.
    - **Payment Method Nodes**: Capture the various payment methods used by customers, which can be linked to churn risk.
    - **Contract Nodes**: Detail the types of contracts customers have, such as month-to-month or two-year agreements.
    - **Service Nodes**: Indicate the specific services each customer subscribes to, like internet or tech support.
    - **Movie Nodes**: For customers with streaming services, we enriched the graph with synthetic movies they have watched, allowing us to explore behavioral similarities and make recommendations.
    - **Relationships**: Define how customers are connected to their locations, payment methods, contracts, and services, as well as similarity relationships between customers based on shared attributes.
    """)
    
elif selected_section == "The Geography":
    st.header("The Geography")
    render_section_intro(
        "Where customers are concentrated and which locations have higher churn rates.",
        "Inspect bubble size for customer concentration and color intensity for churn risk.",
        "Teams can prioritize interventions by location instead of using one-size-fits-all retention actions.",
    )
    st.markdown("""
        Visualizing the physical distribution of our customer base, using geo-coordinates stored in Neo4j.
    """)
    
    # Cache the query so it only runs once per session
    @st.cache_data
    def load_geography():
        query = """
        MATCH (c:Customer)-[:LIVES_AT]->(l:Location)
        WITH l, count(c) AS total_customers,
             sum(CASE WHEN c.churn_label = 'Yes' THEN 1 ELSE 0 END) AS churned_customers
        RETURN l.city AS city,
               l.longitude AS longitude,
               l.latitude AS latitude,
               total_customers,
               round((toFloat(churned_customers) / total_customers) * 100, 2) AS churn_rate_pct
        """
        return analysis.run_query_df(query)

    with st.spinner("Fetching spatial data..."):
        results_df = load_geography()
    
    if not results_df.empty:
        gdf = gpd.GeoDataFrame(
            results_df,
            geometry=gpd.points_from_xy(results_df.longitude, results_df.latitude),
            crs="EPSG:4326",
        )

        # Project to Web Mercator for Contextily
        gdf_wm = gdf.to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot using churn rate for color and customer count for size
        sizes = gdf_wm["total_customers"].fillna(0) * 2 + 20  # ensure a visible minimum size
        gdf_wm.plot(
            ax=ax,
            column="churn_rate_pct",
            cmap="YlOrRd",
            markersize=sizes,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
            legend=True,
            legend_kwds={"label": "Churn %", "shrink": 0.6},
        )

        # Add basemap using the projected CRS
        ctx.add_basemap(ax, crs=gdf_wm.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

        ax.set_axis_off()
        ax.set_title(
            "Customer Distribution by Churn Rate", fontsize=12, fontweight="light", pad=15, color="#333333"
        )

        plt.tight_layout()
        
        # Use Streamlit to render the figure (replacing plt.show())
        st.pyplot(fig, width='stretch')
        st.caption("Dot color reflects churn %, dot size reflects total customers at the location.")
    else:
        st.warning("No geospatial data found in the graph.")

elif selected_section == "Understanding the Customer 360 Graph":
    st.header("The Structural Profile of a Customer")
    render_section_intro(
        "A sampled structural neighborhood around customer nodes across key dimensions.",
        "Choose sample size on the left, run the query, and inspect customer-level connected context on the right.",
        "Customer-level graph context highlights retention and cross-sell opportunities quickly.",
    )
    st.markdown("""
    In a relational database, this view requires joining 5 different tables. In Neo4j, we simply traverse outward from the customer to see their exact product and identity footprint.
    """)

    controls_col, output_col = st.columns([1, 2])

    with controls_col:
        with st.form("customer_lookup_form"):
            customers = st.slider("Number of Customers to Sample", min_value=1, max_value=10, value=2, help="How many customers should we pull?")
            submitted = st.form_submit_button("Generate Customer Graph")

    if submitted:
        query = f"""
            MATCH (c:Customer)
            LIMIT {customers}
            MATCH p=(c)-[rels*..1]-()
            WHERE NONE(r IN rels WHERE type(r) = 'SIMILAR_TO' OR type(r) = 'WATCHED_MOVIE') // Exclude similarity and movie relationships in this view
            RETURN p
            """
        with output_col:
            with st.expander("View Cypher Query"):
                st.code(query, language="cypher")
            print(f"Generating graph for {customers} customers...")
            with st.spinner("Traversing graph..."):

                results = analysis.run_query_viz(query)
                
                if results:
                    st.success("Customer graph generated.")
                    VG = from_neo4j(results)

                    VG.color_nodes(
                        field="caption",
                        color_space=ColorSpace.DISCRETE,
                        colors=colors,
                    )

                    VG.resize_nodes(property="monthly_charges", node_radius_min_max=(10, 30))

                    analysis.set_caption_by_label(VG, label_to_property)

                    generated_html = VG.render(layout=Layout.FORCE_DIRECTED)
                    components.html(generated_html.data, height=NETWORK_GRAPH_HEIGHT)
                else:
                    st.warning("Customer not found or has no connections.")
                
elif selected_section == "Similarity Networks":
    st.header("Similarity Networks")
    render_section_intro(
        "A local customer similarity graph based on shared customer attributes.",
        "Pick a customer and traversal depth on the left, then inspect neighboring similarity clusters on the right.",
        "Similarity neighborhoods help target interventions to at-risk groups with common profiles.",
    )
    st.markdown("""
    By connecting customers who share similar attributes (e.g., same payment method, contract type or certain demographic data), we can identify clusters of customers who share risk factors, preferences, or behaviors.
    """)

    controls_col, output_col = st.columns([1, 2])

    with controls_col:
        with st.form("similarity_network_form"):
            customer_id = st.text_input("Enter a Customer ID to Explore", value="7517-SAWMO", help="Try '7517-SAWMO' for a sample profile")
            customers = st.slider("Number of Similar Customers to Sample", min_value=3, max_value=10, value=10, help="How many similar customers should we pull?")
            depth = st.slider("Degrees of Similarity to Traverse", min_value=1, max_value=4, value=2, help="How many degrees of similarity should we explore?")
            max_customers = st.slider("Max Customers per Degree", min_value=5, max_value=20, value=10, help="How many customers should we pull at each degree of similarity?")
            submitted_sim = st.form_submit_button("Show Similarity Network")
        
    if submitted_sim:
        query = f"""
            // Get exactly n immediate connections
            MATCH l1 = (source:Customer {{customer_id: '{customer_id}'}})-[:SIMILAR_TO]-(target:Customer)
            WITH l1, target LIMIT {customers}
            // For EACH of those n targets, get n extended paths
            CALL(target) {{
                WITH target
                MATCH l2 = (target)-[:SIMILAR_TO*..{depth}]-(other:Customer)
                // Prevent traversing back to the source node
                WHERE other.customer_id <> '{customer_id}' 
                RETURN l2 LIMIT {max_customers}
            }}
            RETURN l1, l2
            """
        with output_col:
            with st.expander("View Cypher Query"):
                st.code(query, language="cypher")
            with st.spinner("Extracting similarity network..."):
                results_sim = analysis.run_query_viz(query)
                
                if results_sim:
                    st.success("Similarity network generated.")
                    VG = from_neo4j(results_sim)

                    VG.color_nodes(
                        field="caption",
                        color_space=ColorSpace.DISCRETE,
                        colors=colors,
                    )

                    VG.resize_nodes(property="monthly_charges", node_radius_min_max=(10, 30))

                    analysis.set_caption_by_label(VG, label_to_property)

                    html_sim = VG.render(layout=Layout.FORCE_DIRECTED, initial_zoom=0.75)
                    components.html(html_sim.data, height=NETWORK_GRAPH_HEIGHT)
                else:
                    st.warning("No similarity relationships found in the graph.")

elif selected_section == "High-Churn Service Combinations":
    st.header("Toxic Service Combinations")
    render_section_intro(
        "Service pair combinations associated with higher churn and a sample subgraph of impacted customers.",
        "Set the top pairs and sample size on the left, then inspect metrics and the network sample on the right.",
        "High-risk bundle combinations can inform packaging, pricing, and proactive retention campaigns.",
    )
    st.markdown("""
        We can query the graph to identify which pairs of services drive the highest churn rates, and explore a structural sample of the affected customer base.
        Similarly, we can identify triplets, or higher-order combinations of services.
    """)

    controls_col, output_col = st.columns([1, 2])

    with controls_col:
        with st.form("service_pair_form"):
            top_n = st.slider("Number of Top Service Pairs to Analyze", min_value=1, max_value=5, value=3, help="How many top service pairs should we analyze?")
            customers = st.slider("Number of Affected Customers to Sample", min_value=5, max_value=20, value=5, help="How many affected customers should we pull for each pair?")
            submitted_service_pairs = st.form_submit_button("Analyze Service Pairs")
    
    if submitted_service_pairs:
        
        query_metrics = f"""
            MATCH (s1:Service)<-[:SUBSCRIBES_TO]-(c:Customer)-[:SUBSCRIBES_TO]->(s2:Service)
            WHERE s1.service_type < s2.service_type
            WITH s1.service_type AS Service_A, 
                 s2.service_type AS Service_B, 
                 count(c) AS Total_Subscribers,
                 sum(CASE WHEN c.churn_label = 'Yes' THEN 1.0 ELSE 0.0 END) AS Churned,
                 sum(CASE WHEN c.churn_label = 'No' THEN 1.0 ELSE 0.0 END) AS Retained
            WHERE Total_Subscribers > 50 AND Retained > 0
            RETURN Service_A, Service_B, Total_Subscribers,
                   round((Churned / Total_Subscribers) * 100, 1) AS Churn_Rate_Pct
            ORDER BY Churn_Rate_Pct DESC
            LIMIT {top_n}
            """
        query_graph = f"""
            // Find the top riskiest service pairs mathematically
            MATCH (s1:Service)<-[:SUBSCRIBES_TO]-(c:Customer)-[:SUBSCRIBES_TO]->(s2:Service)
            WHERE s1.service_type < s2.service_type
            WITH s1, s2, count(c) AS Total_Subscribers,
                 sum(CASE WHEN c.churn_label = 'Yes' THEN 1.0 ELSE 0.0 END) AS Churned
            WHERE Total_Subscribers > 50
            WITH s1, s2, (Churned / Total_Subscribers) AS Churn_Rate
            ORDER BY Churn_Rate DESC
            LIMIT {top_n}
            
            // For each of those top pairs, grab a sample of customers to visualize
            CALL(s1, s2) {{
                WITH s1, s2
                MATCH p=(s1)<-[:SUBSCRIBES_TO]-(c:Customer)-[:SUBSCRIBES_TO]->(s2)
                RETURN p
                LIMIT {customers}
            }}
            RETURN p
            """
        with output_col:
            with st.expander("View Cypher Query for Metrics"):
                st.code(query_metrics, language="cypher")
            with st.expander("View Cypher Query for Graph"):
                st.code(query_graph, language="cypher")

            with st.spinner("Calculating combination risk metrics..."):
                metrics_df = analysis.run_query_df(query_metrics)
                if not metrics_df.empty:
                    st.success("Service combination metrics generated.")
                    render_styled_table(metrics_df)
                else:
                    st.warning("No service combination metrics available for the selected parameters.")

            with st.spinner("Extracting visual subgraphs for top pairs..."):
                results_pairs = analysis.run_query_viz(query_graph)

                if results_pairs:
                    VG = from_neo4j(results_pairs)

                    VG.color_nodes(
                        field="caption",
                        color_space=ColorSpace.DISCRETE,
                        colors=colors,
                    )

                    VG.resize_nodes(property="monthly_charges", node_radius_min_max=(10, 30))

                    analysis.set_caption_by_label(VG, label_to_property)

                    st.info("Customer node size is proportional to monthly charges.")

                    html_pairs = VG.render(layout=Layout.FORCE_DIRECTED, initial_zoom=0.8)
                    components.html(html_pairs.data, height=NETWORK_GRAPH_HEIGHT)
                else:
                    st.warning("Could not extract a graph sample for these services.")

elif selected_section == "GDS: Churn Communities":
    st.header("Behavioral Communities & Structural Risk")
    render_section_intro(
        "Community structure from Louvain clustering over customer similarity relationships.",
        "Adjust the number of clusters and minimum cluster size, then compare structure and risk side by side.",
        "Community-level risk helps prioritize interventions for groups with shared product and behavior patterns.",
    )
    st.markdown("""
        Using the **Louvain algorithm**, we previously grouped customers strictly by their structural similarity (Shared Services, Contracts, Payment Methods, Demographics).
        We then overlaid their actual churn rates to find toxic product combinations which lead to high-churn. This is a classic example of how graph-based clustering can reveal hidden patterns of risk that are invisible to traditional tabular methods.
    """)
    
    controls_col, output_col = st.columns([1, 2])

    with controls_col:
        with st.form("louvain_form"):
            clusters_to_show = st.slider("Number of Louvain Clusters to Show", min_value=3, max_value=15, value=8, help="How many Louvain clusters should we visualize?")
            min_cluster_size = st.slider("Minimum Customers per Cluster", min_value=5, max_value=30, value=20, help="What should be the minimum number of customers in a cluster to be included in the analysis?")
            submitted_louvain = st.form_submit_button("Show Louvain Clusters & Risk Profiles")
    
    if submitted_louvain:
        query_louvain = f"""
                MATCH (c:Customer)
                WHERE c.louvain_community_id IS NOT NULL
                WITH c.louvain_community_id AS comm_id, count(c) AS comm_size
                ORDER BY comm_size DESC LIMIT {clusters_to_show}
                CALL(comm_id) {{
                    WITH comm_id
                    MATCH p=(c1:Customer)-[rel:SIMILAR_TO]->(c2:Customer)
                    WHERE c1.louvain_community_id = comm_id AND c2.louvain_community_id = comm_id
                    RETURN p LIMIT 100 
                }}
                RETURN p
                """
        profile_query = f"""
                MATCH (c:Customer)
                WHERE c.louvain_community_id IS NOT NULL
                WITH c.louvain_community_id AS community_id,
                    count(c) AS total_customers,
                    sum(CASE WHEN c.churn_label = 'Yes' THEN 1 ELSE 0 END) AS churned_customers
                WHERE total_customers > {min_cluster_size}
                RETURN toString(community_id) AS community_id,
                    total_customers,
                    round((toFloat(churned_customers) / total_customers) * 100, 1) AS churn_rate_pct
                ORDER BY total_customers DESC LIMIT {clusters_to_show}
                """
        with output_col:
            with st.expander("View Cypher Query for Louvain Clusters"):
                st.code(query_louvain, language="cypher")
            with st.expander("View Cypher Query for Community Risk Profiles"):
                st.code(profile_query, language="cypher")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("The Sub-Graph Topology")
                with st.spinner("Rendering Louvain clusters..."):
                    results_louvain = analysis.run_query_viz(query_louvain)
                    if results_louvain:
                        st.success("Louvain community topology generated.")
                        VG = from_neo4j(results_louvain)
                        VG.color_nodes(
                            property="louvain_community_id",
                            color_space=ColorSpace.DISCRETE,
                            colors = [
                                "#F00B47",
                                "#00AEEF",
                                "#702082",
                                "#002E5D",
                                "#616AB1",
                                "#9A4F9F",
                                "#007BBF",
                                "#E6007E",
                            ]
                        )
                        VG.resize_relationships(property="weight")
                        html_louvain = VG.render(layout=Layout.FORCE_DIRECTED, initial_zoom=0.15)
                        components.html(html_louvain.data, height=NETWORK_GRAPH_HEIGHT)
                    else:
                        st.warning("No Louvain topology could be generated for the selected parameters.")

            with col2:
                st.subheader("The Risk Landscape")
                with st.spinner("Aggregating risk profiles..."):
                    comm_df = analysis.run_query_df(profile_query)

                    if comm_df.empty:
                        st.warning("No community risk profile data available for the selected filters.")
                    else:
                        sky_palette_12 = [
                            "#F00B47",
                            "#E6007E",
                            "#9A4F9F",
                            "#702082",
                            "#533285",
                            "#616AB1",
                            "#002E5D",
                            "#005B9A",
                            "#007BBF",
                            "#0097D7",
                            "#00AEEF",
                            "#80D6F7"
                        ]

                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.scatterplot(
                            data=comm_df, x="community_id", y="churn_rate_pct",
                            size="total_customers", sizes=(100, 1500), hue="churn_rate_pct",
                            palette=sns.color_palette(sky_palette_12, as_cmap=True), edgecolor="black", alpha=0.8, ax=ax
                        )
                        ax.set_xlabel("Louvain Community ID")
                        ax.set_ylabel("Churn Rate (%)")
                        ax.axhline(26.5, color='red', linestyle='--', linewidth=1, label="Average Churn")
                        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
                        st.pyplot(fig, width='stretch')


elif selected_section == "Geo-Spatial Outages":
    st.header("Physical Infrastructure: Potential Outage Hotspots")
    render_section_intro(
        "Areas where active customers are geographically close to churned customers sharing tech support usage.",
        "Set the distance radius on the left and review hotspot intensity on the map output.",
        "Localized risk mapping helps field teams prioritize preventive maintenance and outreach.",
    )
    st.markdown("""
        We intersect behavioral data (Tech Support subcribers) with Neo4j Spatial functions (Distance < *n* km) to find localized network degradation before the customer even calls in.
        With a graph approach, we can easily anchor on the specific churned customers who were affected by the outage, then traverse to find active customers who live nearby and share
        the same service subscription (Tech Support) to identify high-risk geographic hotspots using geo-spatial features in Neo4j.
    """)
    
    controls_col, output_col = st.columns([1, 2])

    with controls_col:
        with st.form("geo_hotspot_form"):
            km_radius = st.slider("Radius to Define 'Nearby' (km)", min_value=0.5, max_value=5.0, value=2.0, step=0.5, help="How far should we look for nearby customers?")
            submitted_geo = st.form_submit_button("Find Geo Hotspots")
        
    if submitted_geo:
        
        geo_query = f"""
            // Anchor on churned customers who were using Tech Support
            MATCH (churned:Customer {{churn_label: 'Yes'}})-[:SUBSCRIBES_TO]->(:Service {{service_type: 'tech_support'}})
            MATCH (churned)-[:LIVES_AT]->(bad_loc:Location)
            // Find active customers who also use Tech Support...
            MATCH (active:Customer {{churn_label: 'No'}})-[:SUBSCRIBES_TO]->(:Service {{service_type: 'tech_support'}})
            MATCH (active)-[:LIVES_AT]->(nearby_loc:Location)
            // ...who live physically near the churned customers (within 2km)
            WITH churned, bad_loc, active, nearby_loc, 
                point.distance(bad_loc.location_point, nearby_loc.location_point) AS distance_meters
            // Filter for distance > 0 to exclude people living in the exact same household (we handled that in previous queries)
            WHERE distance_meters > 0 AND distance_meters < {km_radius} * 1000
            // Aggregate to find "High-Risk Geographic Hotspots"
            RETURN nearby_loc.zip_code AS At_Risk_Zip, 
                nearby_loc.city AS City,
                nearby_loc.latitude AS latitude,
                nearby_loc.longitude AS longitude,
                count(DISTINCT churned) AS Nearby_Tech_Churners, 
                count(DISTINCT active) AS At_Risk_Active_Neighbors,
                round(avg(distance_meters)) AS Avg_Distance_To_Churn_Meters
            ORDER BY Nearby_Tech_Churners DESC, At_Risk_Active_Neighbors DESC
            """
            
        with output_col:
            with st.expander("View Cypher Query"):
                st.code(geo_query, language="cypher")

            with st.spinner("Calculating spatial bounds..."):
                geo_df = analysis.run_query_df(geo_query)

            if not geo_df.empty:
                st.success("Geo-spatial hotspots generated.")
                gdf = gpd.GeoDataFrame(
                    geo_df,
                    geometry=gpd.points_from_xy(geo_df.longitude, geo_df.latitude),
                    crs="EPSG:4326",
                )

                # Project to Web Mercator to prevent map distortion with contextily tiles
                gdf_wm = gdf.to_crs(epsg=3857)

                fig, ax = plt.subplots(figsize=(6, 4))

                # Map both the color and the size of the bubble to the severity of the risk
                gdf_wm.plot(
                    ax=ax,
                    column="At_Risk_Active_Neighbors",  # Color the dots based on the count
                    cmap="YlOrRd",  # Use a Yellow -> Orange -> Red heatmap palette
                    markersize=gdf_wm["At_Risk_Active_Neighbors"]
                    * 50,  # Multiply by 50 so the bubbles are visibly large
                    alpha=0.8,  # Slight transparency
                    edgecolors="#000000",  # Add a thin dark border so overlapping bubbles are distinct
                    linewidth=1.0,
                    legend=True,
                    legend_kwds={
                        "label": "At-Risk Active Neighbors",
                        "shrink": 0.6,
                    },  # Adds a nice colorbar scale
                )

                # Add basemap
                ctx.add_basemap(
                    ax, crs=gdf_wm.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik
                )

                ax.set_axis_off()  # Hides the coordinates and borders
                ax.set_title(
                    "Tech Support Outage Hotspots",
                    fontsize=12,
                    fontweight="light",
                    pad=15,
                    color="#333333",
                )

                plt.tight_layout()
                st.pyplot(fig, width='stretch')
            else:
                st.warning("No geospatial risk hotspots found based on the current data.")

elif selected_section == "Churn Heatmap":
    st.header("Churn Heatmap by Segment")
    render_section_intro(
        "Churn rates by contract and payment method segments, optionally filtered by city.",
        "Pick city and minimum segment size on the left, then inspect high-risk segment cells in the heatmap.",
        "Segment-level churn patterns can refine offers, payment incentives, and contract migration strategies.",
    )
    st.markdown("""
        A typical segmentation analysis might look at churn rates by contract type, or by payment method. With a graph, we can easily do a multi-dimensional segmentation to
        compare churn rates across contract and payment method segments to spot combinations with higher risk.
    """)

    @st.cache_data(show_spinner=False)
    def load_cities():
        city_df = analysis.run_query_df(
            """
            MATCH (:Customer)-[:LIVES_AT]->(l:Location)
            WHERE l.city IS NOT NULL
            RETURN DISTINCT l.city AS city
            ORDER BY city
            """
        )
        return ["All Cities"] + city_df["city"].tolist() if not city_df.empty else ["All Cities"]

    city_options = load_cities()
    
    controls_col, output_col = st.columns([1, 2])

    with controls_col:
        with st.form("heatmap_form"):
            selected_city = st.selectbox("City", city_options, help="Filter the heatmap to a specific city or show all cities.")
            minimum_customers = st.slider("Minimum Customers per Segment", min_value=10, max_value=50, value=20, step=10, help="What should be the minimum number of customers in a segment to be included in the heatmap?")
            submitted_heatmap = st.form_submit_button("Generate Churn Heatmap")
        
    if submitted_heatmap:
        sanitized_city = selected_city.replace("'", "\\'")
        city_filter = "" if selected_city == "All Cities" else f"WHERE l.city = '{sanitized_city}'"

        churn_heatmap_query = f"""
            MATCH (c:Customer)-[:HAS_CONTRACT]->(ct:Contract)
            MATCH (c)-[:PAYS_WITH]->(p:PaymentMethod)
            MATCH (c)-[:LIVES_AT]->(l:Location)
            {city_filter}
            WITH ct.contract AS contract, p.payment_method AS payment_method,
                count(c) AS total_customers,
                sum(CASE WHEN c.churn_label = 'Yes' THEN 1 ELSE 0 END) AS churned_customers
            WHERE contract IS NOT NULL AND payment_method IS NOT NULL AND total_customers > {minimum_customers}
            RETURN contract, payment_method, total_customers,
                round((toFloat(churned_customers) / total_customers) * 100, 2) AS churn_rate_pct
            ORDER BY churn_rate_pct DESC
        """

        with output_col:
            with st.expander("View Cypher Query"):
                st.code(churn_heatmap_query, language="cypher")

            with st.spinner("Building churn heatmap..."):
                heatmap_df = analysis.run_query_df(churn_heatmap_query)

            if heatmap_df.empty:
                st.warning("No churn data available for the selected segments.")
            else:
                st.success("Churn heatmap generated.")
                pivot_df = heatmap_df.pivot(index="contract", columns="payment_method", values="churn_rate_pct")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5, cbar_kws={"label": "Churn %"}, ax=ax)
                ax.set_xlabel("Payment Method")
                ax.set_ylabel("Contract")
                ax.set_title("Churn Rate by Contract x Payment Method")
                st.pyplot(fig, width='stretch')

                st.caption(f"Cells show churn % for segments with more than {minimum_customers} customers.")

elif selected_section == "Watched Movie Networks":
    st.header("Watched Movie Networks")
    render_section_intro(
        "Shared movie watching behavior between similar customer pairs.",
        "Set sample size on the left and inspect the customer-movie-customer bridge network on the right.",
        "Behavioral overlap helps identify micro-segments for personalized engagement.",
    )
    st.markdown("""
        We enriched synthetic data with watched movies for customers with Streaming services, let us visualize a network of shared watched movies across customers."
    """)

    controls_col, output_col = st.columns([1, 2])

    with controls_col:
        with st.form("movie_network_form"):
            customers = st.slider("Number of Customer Pairs to Sample", min_value=1, max_value=10, value=2, help="How many customer pairs should we pull?")
            submitted_movies = st.form_submit_button("Generate Movie Network")
        
    if submitted_movies:
        query = f"""
            MATCH (source:Customer)<-[s:SIMILAR_TO]->(target:Customer)
            LIMIT {customers}
            MATCH p=(source)-[:WATCHED_MOVIE]-(:Movie)-[:WATCHED_MOVIE]-(target)
            RETURN p,s
            """
        with output_col:
            with st.expander("View Cypher Query"):
                st.code(query, language="cypher")
            print(f"Generating movie network for {customers} customer pairs...")
            with st.spinner("Traversing graph..."):

                results = analysis.run_query_viz(query)

                if results:
                    st.success("Watched movie network generated.")
                    VG = from_neo4j(results)

                    VG.color_nodes(
                        field="caption",
                        color_space=ColorSpace.DISCRETE,
                        colors=colors,
                    )

                    analysis.set_caption_by_label(VG, label_to_property)

                    generated_html = VG.render(layout=Layout.FORCE_DIRECTED)
                    components.html(generated_html.data, height=NETWORK_GRAPH_HEIGHT)
                else:
                    st.warning("Customers not found or has no connections.")

elif selected_section == "Movie Recommendations":
    st.header("Movie Recommendations")
    render_section_intro(
        "Top movie candidates based on similar customers' watches and ratings.",
        "Enter a customer_id and recommendation count on the left, then review ranked recommendations on the right.",
        "Recommendation ranking combines behavioral similarity and preference signal for better relevance.",
    )
    st.markdown("""
        A simple recommendation based on what similar customers watched and rated, excluding movies this customer already watched or rated.
    """)

    controls_col, output_col = st.columns([1, 2])

    with controls_col:
        with st.form("movie_recommendation_form"):
            customer_id = st.text_input(
                "Enter a Customer ID",
                value="7517-SAWMO",
                help="Provide a customer_id to get movie recommendations.",
            )
            recommendation_count = st.slider(
                "Number of Recommendations",
                min_value=1,
                max_value=10,
                value=5,
                help="How many movie recommendations should we return?",
            )
            submitted_recommendations = st.form_submit_button("Recommend Movies")

    if submitted_recommendations:
        sanitized_customer_id = customer_id.replace("'", "\\'")
        recommendation_query = f"""
            MATCH (c:Customer {{customer_id: '{sanitized_customer_id}'}})
            OPTIONAL MATCH (c)-[:WATCHED_MOVIE]->(seen:Movie)
            OPTIONAL MATCH (c)-[:RATED]->(rated_seen:Movie)
            WITH c, collect(DISTINCT seen) + collect(DISTINCT rated_seen) AS seen_movies
            MATCH (c)-[:SIMILAR_TO]-(sim:Customer)-[:WATCHED_MOVIE]->(m:Movie)
            OPTIONAL MATCH (sim)-[r:RATED]->(m)
            WHERE NOT m IN seen_movies
            RETURN m.title AS movie_title,
                   count(DISTINCT sim) AS similar_customers_who_watched,
                   count(r) AS ratings_count,
                   round(avg(r.rating), 2) AS avg_rating
            ORDER BY similar_customers_who_watched DESC,
                     avg_rating DESC,
                     ratings_count DESC,
                     movie_title ASC
            LIMIT {recommendation_count}
        """

        with output_col:
            with st.expander("View Cypher Query"):
                st.code(recommendation_query, language="cypher")

            with st.spinner("Generating recommendations..."):
                recommendations_df = analysis.run_query_df(recommendation_query)

            if recommendations_df.empty:
                st.warning("No recommendations found for this customer_id. Try a different customer.")
            else:
                st.success("Movie recommendations generated.")
                render_styled_table(recommendations_df)

elif selected_section == "KNN Flight Risk Prediction":
    st.header("KNN Flight Risk Prediction")
    render_section_intro(
        "Predicted churn macro-reason for active customers based on churned NEAREST_NEIGHBOR relationships.",
        "Review the prediction table, then select a customer_id to inspect their NEAREST_NEIGHBOR subgraph.",
        "Neighborhood-based signals can support explainable retention prioritization.",
    )
    st.markdown("""
        This section runs a neighbor-voting prediction using existing `NEAREST_NEIGHBOR` edges and then separately visualizes the selected customer's neighbor context.
    """)

    prediction_query = """
// Start with Active customers
MATCH (active:Customer {churn_label: 'No'})
// Traverse the new KNN edges to find their Churned neighbors
MATCH (active)-[rel:NEAREST_NEIGHBOR]->(churned:Customer {churn_label: 'Yes'})
WHERE churned.macro_reason IS NOT NULL

// Count up the reasons among those specific neighbors
WITH active, churned.macro_reason AS predicted_reason, count(*) AS votes

// Sort to find the most common reason for each active customer
ORDER BY active.customer_id, votes DESC
WITH active, collect({reason: predicted_reason, votes: votes})[0] AS top_prediction

// Return the result!
RETURN active.customer_id AS CustomerID, 
       top_prediction.reason AS PredictedFlightRisk, 
       (toFloat(top_prediction.votes) / 5.0) AS ConfidenceScore
ORDER BY ConfidenceScore DESC, CustomerID
"""

    with st.expander("View Cypher Query"):
        st.code(prediction_query, language="cypher")

    with st.spinner("Running KNN prediction query..."):
        results_df = analysis.run_query_df(prediction_query)

    if results_df.empty:
        st.warning("No prediction results found.")
    else:
        st.success("Prediction results generated.")
        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.subheader("Prediction Results")
            render_styled_table(results_df)

        ranked_df = results_df.sort_values(
            by=["ConfidenceScore", "CustomerID"],
            ascending=[False, True],
        )
        customer_options = ranked_df["CustomerID"].dropna().astype(str).tolist()
        default_customer_id = str(ranked_df.iloc[0]["CustomerID"])

        with right_col:
            st.subheader("NEAREST_NEIGHBOR Graph")
            selected_customer_id = st.selectbox(
                "Select customer_id to visualize NEAREST_NEIGHBOR graph",
                options=customer_options,
                index=customer_options.index(default_customer_id),
            )

            graph_query = f"""
                MATCH p=(active:Customer {{customer_id: '{selected_customer_id}'}})-[:NEAREST_NEIGHBOR]->(churned:Customer {{churn_label: 'Yes'}})
                WHERE churned.macro_reason IS NOT NULL
                RETURN p
                """

            with st.expander("View NEAREST_NEIGHBOR Graph Query"):
                st.code(graph_query, language="cypher")

            with st.spinner("Rendering selected customer neighbor graph..."):
                results_knn_graph = analysis.run_query_viz(graph_query)

            if results_knn_graph:
                st.success("NEAREST_NEIGHBOR graph generated.")
                VG = from_neo4j(results_knn_graph)

                VG.color_nodes(
                    field="caption",
                    color_space=ColorSpace.DISCRETE,
                    colors=colors,
                )

                VG.resize_nodes(property="monthly_charges", node_radius_min_max=(10, 30))

                analysis.set_caption_by_label(VG, label_to_property)

                html_knn = VG.render(layout=Layout.FORCE_DIRECTED, initial_zoom=0.75)
                components.html(html_knn.data, height=NETWORK_GRAPH_HEIGHT)
            else:
                st.warning("No NEAREST_NEIGHBOR graph data found for the selected customer.")

elif selected_section == "Agent-Based Analysis":
    st.header("Agent-Based Analysis: An example GenAI use case for graph data")
    render_section_intro(
        "Natural language Q&A where an agent translates questions into graph operations.",
        "Enter a question on the left and inspect the answer plus optional tabular evidence on the right.",
        "Business users can query graph insights without writing Cypher directly.",
    )
    st.markdown("""
        We can implement an agent which uses the graph to answer business questions in natural language. This is a first step towards complex agentic systems which can solve and automate business processes.
    """)

    controls_col, output_col = st.columns([1, 2])

    with controls_col:
        with st.form("agent_form"):
            question = st.text_input("Ask a question...", value="How many unique customers do we have? How many of them have churned?", help="Try asking a question about the data. For example: 'How many unique customers do we have? How many of them have churned?'")
            submitted_agent = st.form_submit_button("Ask the Graph Agent")
        
    if submitted_agent:
        with output_col:
            if not all([AURA_API_CLIENT_ID, AURA_API_CLIENT_SECRET, AURA_API_TEXT2CYPHER_ENDPOINT]):
                st.error("Aura API credentials or endpoint are missing. Please set AURA_API_CLIENT_ID, AURA_API_CLIENT_SECRET, and AURA_API_TEXT2CYPHER_ENDPOINT.")
            else:
                with st.spinner("The agent is thinking..."):
                    try:
                        bearer_token = get_aura_access_token(AURA_API_CLIENT_ID, AURA_API_CLIENT_SECRET)
                    except Exception as exc:
                        st.error(f"Failed to get Aura API access token: {exc}")
                    else:
                        try:
                            agent_response = invoke_aura_agent(AURA_API_TEXT2CYPHER_ENDPOINT, bearer_token, question)
                        except Exception as exc:
                            st.error(f"Agent call failed: {exc}")
                        else:
                            st.subheader("Agent Response")

                            content = agent_response.get("content", []) if isinstance(agent_response, dict) else []
                            text_answers = [item.get("text") for item in content if isinstance(item, dict) and item.get("type") == "text" and item.get("text")]
                            if text_answers:
                                st.success(text_answers[-1])

                            tool_results = [
                                item.get("output", {})
                                for item in content
                                if isinstance(item, dict) and item.get("type") == "tool_result" and isinstance(item.get("output"), dict)
                            ]
                            for tool_output in tool_results:
                                result = tool_output.get("result") or {}
                                records = result.get("records") if isinstance(result, dict) else None
                                keys = result.get("keys") if isinstance(result, dict) else None
                                if keys and records:
                                    rows = [
                                        {k: rec.get(k) for k in keys}
                                        for rec in records
                                        if isinstance(rec, dict)
                                    ]
                                    if rows:
                                        render_styled_table(pd.DataFrame(rows))

                            with st.expander("Raw agent response"):
                                st.json(agent_response)
