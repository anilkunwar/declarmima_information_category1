#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEMO: Chat-Driven Scientific Visualization for DECLARMIMA RAG
=============================================================
Standalone demo showing chat-driven visualization capabilities.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
from pathlib import Path
from io import BytesIO

# =============================================
# DEMO DATA GENERATOR
# =============================================

def generate_demo_knowledge_graph():
    """Generate demo data simulating a populated knowledge graph."""

    class DemoEntity:
        def __init__(self, text, label, value, unit, doc_source, normalized):
            self.text = text
            self.label = label
            self.value = value
            self.unit = unit
            self.doc_source = doc_source
            self.normalized = normalized

    class DemoGraph:
        def __init__(self):
            self.entities = defaultdict(list)
            self.documents = {
                "paper1_laser_slm_heas.pdf": {"chunk_count": 45},
                "paper2_soldering_snagcu.pdf": {"chunk_count": 38},
                "paper3_ablation_silicon.pdf": {"chunk_count": 52},
                "paper4_lpbf_inconel718.pdf": {"chunk_count": 41},
                "paper5_hea_thermal_properties.pdf": {"chunk_count": 47},
            }
            self.entity_index = defaultdict(set)

            # Add multicomponent alloys
            for doc in ["paper1_laser_slm_heas.pdf", "paper5_hea_thermal_properties.pdf"]:
                for _ in range(np.random.randint(8, 15)):
                    self.entities["cocrfeni"].append(DemoEntity("CoCrFeNi", "MATERIAL", None, None, doc, "cocrfeni"))
                    self.entity_index["cocrfeni"].add(doc)
                for _ in range(np.random.randint(5, 10)):
                    self.entities["alcocrfeni"].append(DemoEntity("AlCoCrFeNi", "MATERIAL", None, None, doc, "alcocrfeni"))
                    self.entity_index["alcocrfeni"].add(doc)
                for _ in range(np.random.randint(3, 8)):
                    self.entities["crmnfeconi"].append(DemoEntity("CrMnFeCoNi", "MATERIAL", None, None, doc, "crmnfeconi"))
                    self.entity_index["crmnfeconi"].add(doc)

            # Add solders
            for doc in ["paper2_soldering_snagcu.pdf"]:
                for _ in range(np.random.randint(10, 18)):
                    self.entities["snagcu"].append(DemoEntity("Sn-Ag-Cu", "MATERIAL", None, None, doc, "snagcu"))
                    self.entity_index["snagcu"].add(doc)
                for _ in range(np.random.randint(5, 12)):
                    self.entities["sac305"].append(DemoEntity("SAC305", "MATERIAL", None, None, doc, "sac305"))
                    self.entity_index["sac305"].add(doc)

            # Add superalloys
            for doc in ["paper4_lpbf_inconel718.pdf"]:
                for _ in range(np.random.randint(12, 20)):
                    self.entities["inconel718"].append(DemoEntity("Inconel 718", "MATERIAL", None, None, doc, "inconel718"))
                    self.entity_index["inconel718"].add(doc)

            # Add silicon
            for doc in ["paper3_ablation_silicon.pdf"]:
                for _ in range(np.random.randint(8, 15)):
                    self.entities["silicon"].append(DemoEntity("Silicon", "MATERIAL", None, None, doc, "silicon"))
                    self.entity_index["silicon"].add(doc)

            # Add steel
            for doc in ["paper1_laser_slm_heas.pdf", "paper4_lpbf_inconel718.pdf"]:
                for _ in range(np.random.randint(3, 7)):
                    self.entities["steel"].append(DemoEntity("Steel", "MATERIAL", None, None, doc, "steel"))
                    self.entity_index["steel"].add(doc)

            # Add aluminum
            for doc in ["paper1_laser_slm_heas.pdf"]:
                for _ in range(np.random.randint(2, 5)):
                    self.entities["aluminum"].append(DemoEntity("Aluminum", "MATERIAL", None, None, doc, "aluminum"))
                    self.entity_index["aluminum"].add(doc)

            # Add laser parameters with values
            power_values = [200, 250, 280, 300, 350, 400, 450, 500]
            for doc in self.documents.keys():
                for _ in range(np.random.randint(2, 5)):
                    val = np.random.choice(power_values)
                    self.entities["power"].append(DemoEntity(f"{val}W", "power", val, "W", doc, "power"))
                    self.entity_index["power"].add(doc)

            scan_speeds = [200, 500, 800, 1000, 1200, 1500, 2000]
            for doc in self.documents.keys():
                for _ in range(np.random.randint(2, 4)):
                    val = np.random.choice(scan_speeds)
                    self.entities["scan_speed"].append(DemoEntity(f"{val}mm/s", "scan_speed", val, "mm/s", doc, "scan_speed"))
                    self.entity_index["scan_speed"].add(doc)

            wavelengths = [1064, 1030, 532, 355]
            for doc in self.documents.keys():
                val = np.random.choice(wavelengths)
                self.entities["wavelength"].append(DemoEntity(f"{val}nm", "wavelength", val, "nm", doc, "wavelength"))
                self.entity_index["wavelength"].add(doc)

            # Add methods
            methods_data = [
                ("slm", "SLM", ["paper1_laser_slm_heas.pdf", "paper4_lpbf_inconel718.pdf"]),
                ("lpbf", "LPBF", ["paper1_laser_slm_heas.pdf", "paper4_lpbf_inconel718.pdf"]),
                ("soldering", "Laser Soldering", ["paper2_soldering_snagcu.pdf"]),
                ("ablation", "Laser Ablation", ["paper3_ablation_silicon.pdf"]),
                ("ded", "DED", ["paper5_hea_thermal_properties.pdf"]),
            ]
            for norm, text, docs in methods_data:
                for doc in docs:
                    for _ in range(np.random.randint(5, 12)):
                        self.entities[norm].append(DemoEntity(text, "METHOD", None, None, doc, norm))
                        self.entity_index[norm].add(doc)

            # Add properties
            thermal_conductivities = [12.5, 15.2, 18.7, 22.3, 25.1, 28.9, 31.4]
            for doc in ["paper5_hea_thermal_properties.pdf", "paper1_laser_slm_heas.pdf"]:
                for _ in range(np.random.randint(3, 6)):
                    val = np.random.choice(thermal_conductivities)
                    self.entities["thermal_conductivity"].append(DemoEntity(f"{val} W/mK", "thermal_conductivity", val, "W/mK", doc, "thermal_conductivity"))
                    self.entity_index["thermal_conductivity"].add(doc)

            grain_sizes = [5.2, 8.1, 12.3, 15.7, 18.9, 22.4, 28.6, 35.2]
            for doc in self.documents.keys():
                for _ in range(np.random.randint(2, 5)):
                    val = np.random.choice(grain_sizes)
                    self.entities["grain_size"].append(DemoEntity(f"{val} μm", "grain_size", val, "μm", doc, "grain_size"))
                    self.entity_index["grain_size"].add(doc)

    return DemoGraph()


# =============================================
# SIMPLIFIED VISUALIZATION ENGINE
# =============================================

class SimpleChatViz:
    def __init__(self, graph):
        self.graph = graph

    def parse_query(self, query: str):
        query_lower = query.lower()
        result = {'chart_type': 'bar', 'filter': None, 'focus': None}

        # Chart type
        if any(w in query_lower for w in ['pie', 'proportion', 'percentage', 'share']):
            result['chart_type'] = 'pie'
        elif any(w in query_lower for w in ['line', 'trend', 'over time']):
            result['chart_type'] = 'line'
        elif any(w in query_lower for w in ['scatter', 'vs', 'versus', 'against']):
            result['chart_type'] = 'scatter'
        elif any(w in query_lower for w in ['heatmap', 'matrix', 'co-occurrence']):
            result['chart_type'] = 'heatmap'
        elif any(w in query_lower for w in ['radar', 'spider', 'profile']):
            result['chart_type'] = 'radar'
        elif any(w in query_lower for w in ['box', 'boxplot', 'distribution']):
            result['chart_type'] = 'box'
        elif any(w in query_lower for w in ['bubble']):
            result['chart_type'] = 'bubble'

        # Focus
        if any(w in query_lower for w in ['multicomponent', 'hea', 'high entropy', 'mpea']):
            result['focus'] = 'multicomponent'
        elif any(w in query_lower for w in ['solder', 'snagcu', 'sac']):
            result['focus'] = 'solder'
        elif any(w in query_lower for w in ['superalloy', 'inconel', 'in718']):
            result['focus'] = 'superalloy'
        elif any(w in query_lower for w in ['power', 'watt', 'w ']):
            result['focus'] = 'power'
        elif any(w in query_lower for w in ['scan speed', 'scanning speed']):
            result['focus'] = 'scan_speed'
        elif any(w in query_lower for w in ['method', 'technique', 'process']):
            result['focus'] = 'method'
        elif any(w in query_lower for w in ['property', 'thermal conductivity', 'hardness', 'grain size']):
            result['focus'] = 'property'

        # Filter
        if 'only' in query_lower or 'among' in query_lower:
            result['filter'] = 'compare'

        return result

    def generate(self, query: str):
        parsed = self.parse_query(query)

        if parsed['focus'] == 'multicomponent':
            if parsed['filter'] == 'compare':
                return self._multicomponent_vs_all(parsed['chart_type'])
            else:
                return self._multicomponent_only(parsed['chart_type'])
        elif parsed['focus'] == 'solder':
            return self._material_focus('solder', parsed['chart_type'])
        elif parsed['focus'] == 'superalloy':
            return self._material_focus('superalloy', parsed['chart_type'])
        elif parsed['focus'] == 'power':
            return self._parameter_chart('power', parsed['chart_type'])
        elif parsed['focus'] == 'scan_speed':
            return self._parameter_chart('scan_speed', parsed['chart_type'])
        elif parsed['focus'] == 'method':
            return self._method_chart(parsed['chart_type'])
        elif parsed['focus'] == 'property':
            return self._property_chart(parsed['chart_type'])
        else:
            return self._all_materials(parsed['chart_type'])

    def _multicomponent_vs_all(self, chart_type):
        categories = defaultdict(int)
        for ent_norm, entities in self.graph.entities.items():
            cat = self._categorize_material(ent_norm)
            categories[cat] += len(entities)

        df = pd.DataFrame([
            {'category': k.replace('_', ' ').title(), 'count': v}
            for k, v in categories.items()
        ])
        df['is_multicomponent'] = df['category'] == 'Multicomponent Alloy'

        if chart_type == 'pie':
            fig = px.pie(df, names='category', values='count', 
                        title='Material Distribution: Multicomponent Alloys vs Others',
                        color='is_multicomponent',
                        color_discrete_map={True: '#e11d48', False: '#64748b'},
                        hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label')
        else:
            fig = px.bar(df, x='category', y='count', 
                        title='Material Distribution: Multicomponent Alloys vs Others',
                        color='is_multicomponent',
                        color_discrete_map={True: '#e11d48', False: '#64748b'})
            fig.update_traces(texttemplate='%{y}', textposition='outside')

        fig.update_layout(template='plotly_white', showlegend=False)
        return fig, df

    def _multicomponent_only(self, chart_type):
        mc_alloys = {}
        for ent_norm, entities in self.graph.entities.items():
            if self._categorize_material(ent_norm) == 'multicomponent_alloy':
                mc_alloys[ent_norm.upper()] = len(entities)

        df = pd.DataFrame([
            {'alloy': k, 'mentions': v}
            for k, v in mc_alloys.items()
        ])

        if chart_type == 'pie':
            fig = px.pie(df, names='alloy', values='mentions', title='Multicomponent Alloy Distribution')
        else:
            fig = px.bar(df, x='alloy', y='mentions', title='Multicomponent Alloy Mentions',
                        color='alloy', color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_traces(texttemplate='%{y}', textposition='outside')

        fig.update_layout(template='plotly_white', showlegend=False)
        return fig, df

    def _material_focus(self, focus, chart_type):
        categories = defaultdict(int)
        for ent_norm, entities in self.graph.entities.items():
            cat = self._categorize_material(ent_norm)
            categories[cat] += len(entities)

        df = pd.DataFrame([
            {'category': k.replace('_', ' ').title(), 'count': v}
            for k, v in categories.items()
        ])
        df['highlight'] = df['category'] == focus.replace('_', ' ').title()

        if chart_type == 'pie':
            fig = px.pie(df, names='category', values='count', title=f'{focus.title()} vs Other Materials',
                        color='highlight', color_discrete_map={True: '#e11d48', False: '#64748b'}, hole=0.4)
        else:
            fig = px.bar(df, x='category', y='count', title=f'{focus.title()} vs Other Materials',
                        color='highlight', color_discrete_map={True: '#e11d48', False: '#64748b'})
            fig.update_traces(texttemplate='%{y}', textposition='outside')

        fig.update_layout(template='plotly_white', showlegend=False)
        return fig, df

    def _parameter_chart(self, param, chart_type):
        data = []
        for ent_norm, entities in self.graph.entities.items():
            if ent_norm == param:
                for e in entities:
                    if e.value is not None:
                        data.append({
                            'value': e.value,
                            'unit': e.unit,
                            'document': Path(e.doc_source).stem
                        })

        df = pd.DataFrame(data)
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", showarrow=False, font=dict(size=20))
            return fig, df

        if chart_type == 'box':
            fig = px.box(df, y='value', title=f'{param.replace("_", " ").title()} Distribution')
        elif chart_type == 'scatter':
            fig = px.scatter(df, x='document', y='value', color='document',
                           title=f'{param.replace("_", " ").title()} by Document',
                           size='value')
        else:
            fig = px.histogram(df, x='value', title=f'{param.replace("_", " ").title()} Distribution',
                             color='document', barmode='group')

        fig.update_layout(template='plotly_white')
        return fig, df

    def _method_chart(self, chart_type):
        methods = defaultdict(int)
        for ent_norm, entities in self.graph.entities.items():
            if self._categorize_method(ent_norm) != 'other':
                methods[self._categorize_method(ent_norm)] += len(entities)

        df = pd.DataFrame([
            {'method': k, 'count': v}
            for k, v in methods.items()
        ])

        if chart_type == 'pie':
            fig = px.pie(df, names='method', values='count', title='Laser Processing Methods', hole=0.4)
        else:
            fig = px.bar(df, x='method', y='count', title='Laser Processing Methods',
                        color='method', color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_traces(texttemplate='%{y}', textposition='outside')

        fig.update_layout(template='plotly_white', showlegend=False)
        return fig, df

    def _property_chart(self, chart_type):
        props = defaultdict(list)
        for ent_norm, entities in self.graph.entities.items():
            if ent_norm in ['thermal_conductivity', 'grain_size', 'interfacial_energy']:
                for e in entities:
                    if e.value is not None:
                        props[ent_norm.replace('_', ' ').title()].append(e.value)

        categories = list(props.keys())
        values = [np.mean(v) if v else 0 for v in props.values()]

        df = pd.DataFrame([
            {'property': k, 'mean_value': np.mean(v) if v else 0, 'count': len(v)}
            for k, v in props.items()
        ])

        if chart_type == 'radar':
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Mean Values'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title='Property Profile',
                template='plotly_white'
            )
        elif chart_type == 'box':
            box_data = []
            for prop, vals in props.items():
                for v in vals:
                    box_data.append({'property': prop, 'value': v})
            df_box = pd.DataFrame(box_data)
            fig = px.box(df_box, x='property', y='value', title='Property Distributions')
        else:
            fig = px.bar(df, x='property', y='mean_value', title='Mean Property Values',
                        color='property', color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')

        fig.update_layout(template='plotly_white', showlegend=False)
        return fig, df

    def _all_materials(self, chart_type):
        categories = defaultdict(int)
        for ent_norm, entities in self.graph.entities.items():
            cat = self._categorize_material(ent_norm)
            categories[cat] += len(entities)

        df = pd.DataFrame([
            {'category': k.replace('_', ' ').title(), 'count': v}
            for k, v in categories.items()
        ])

        if chart_type == 'pie':
            fig = px.pie(df, names='category', values='count', title='All Materials', hole=0.4)
        else:
            fig = px.bar(df, x='category', y='count', title='All Materials',
                        color='category', color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_traces(texttemplate='%{y}', textposition='outside')

        fig.update_layout(template='plotly_white', showlegend=False)
        return fig, df

    def _categorize_material(self, ent_norm):
        ent_lower = ent_norm.lower()
        if any(kw in ent_lower for kw in ['hea', 'mpea', 'multicomponent', 'cocrfeni', 'alcocrfeni', 'crmnfeconi']):
            return 'multicomponent_alloy'
        if any(kw in ent_lower for kw in ['solder', 'snagcu', 'sac']):
            return 'solder'
        if any(kw in ent_lower for kw in ['inconel', 'superalloy']):
            return 'superalloy'
        if 'steel' in ent_lower:
            return 'steel'
        if 'titanium' in ent_lower or 'ti-' in ent_lower:
            return 'titanium'
        if 'aluminum' in ent_lower or 'al-' in ent_lower:
            return 'aluminum'
        if 'copper' in ent_lower or ent_lower == 'cu':
            return 'copper'
        if 'silicon' in ent_lower or ent_lower == 'si':
            return 'silicon'
        if any(kw in ent_lower for kw in ['ceramic', 'alumina', 'zirconia']):
            return 'ceramic'
        if any(kw in ent_lower for kw in ['polymer', 'pmma', 'polyimide']):
            return 'polymer'
        return 'other'

    def _categorize_method(self, ent_norm):
        ent_lower = ent_norm.lower()
        if any(kw in ent_lower for kw in ['slm', 'lpbf', 'powder bed']):
            return 'SLM/LPBF'
        if any(kw in ent_lower for kw in ['ded', 'cladding', 'deposition']):
            return 'DED/Cladding'
        if 'soldering' in ent_lower:
            return 'Laser Soldering'
        if 'ablation' in ent_lower:
            return 'Laser Ablation'
        if 'welding' in ent_lower:
            return 'Laser Welding'
        if any(kw in ent_lower for kw in ['structuring', 'texturing', 'lipss']):
            return 'Surface Structuring'
        if 'annealing' in ent_lower:
            return 'Laser Annealing'
        return 'other'


# =============================================
# STREAMLIT APP
# =============================================

def main():
    st.set_page_config(
        page_title="🔬 Chat-Driven Scientific Viz Demo",
        page_icon="📊",
        layout="wide"
    )

    st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        background: linear-gradient(90deg, #1e40af, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
    }
    .viz-card {
        background: #f8fafc;
        border-left: 4px solid #7c3aed;
        padding: 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🔬 Chat-Driven Scientific Visualization</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#64748b;margin-bottom:2rem">
    Type natural language queries to generate scientific charts from your knowledge graph.<br>
    <strong>Examples:</strong> <em>"Plot multicomponent alloys among all materials"</em> or <em>"Show pie chart of laser power distribution"</em>
    </div>
    """, unsafe_allow_html=True)

    # Initialize demo graph and session state
    if 'demo_graph' not in st.session_state:
        st.session_state.demo_graph = generate_demo_knowledge_graph()
        st.session_state.viz_engine = SimpleChatViz(st.session_state.demo_graph)
        st.session_state.viz_query = ""   # separate from widget key

    # Layout
    col_query, col_examples = st.columns([2, 1])

    with col_query:
        st.markdown("### 💬 Visualization Query")

        def update_query():
            st.session_state.viz_query = st.session_state.viz_query_widget

        query = st.text_input(
            "Enter your visualization request:",
            placeholder="e.g., 'Plot multicomponent alloys among all materials as a pie chart'",
            key="viz_query_widget",
            value=st.session_state.viz_query,
            on_change=update_query
        )

        col_gen, col_clear = st.columns([1, 1])
        with col_gen:
            generate = st.button("📈 Generate Visualization", type="primary", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.viz_query = ""
                st.session_state.pop('last_fig', None)
                st.session_state.pop('last_df', None)
                st.rerun()

    with col_examples:
        st.markdown("### 💡 Example Queries")
        examples = [
            "Plot multicomponent alloys among all materials",
            "Show pie chart of laser processing methods",
            "Plot laser power distribution as box plot",
            "Compare solder alloys vs other materials",
            "Show radar chart of material properties",
            "Plot scan speed vs document scatter",
            "Show multicomponent alloys only",
            "Plot grain size distribution",
        ]
        for ex in examples:
            if st.button(f"▶ {ex}", key=f"ex_{ex[:20]}", use_container_width=True):
                st.session_state.viz_query = ex
                st.rerun()

    # Generate visualization
    if generate and st.session_state.viz_query:
        with st.spinner("🔍 Parsing query and generating chart..."):
            try:
                fig, df = st.session_state.viz_engine.generate(st.session_state.viz_query)
                st.session_state.last_fig = fig
                st.session_state.last_df = df
                st.session_state.last_query = st.session_state.viz_query
            except Exception as e:
                st.error(f"Error: {e}")

    # Display results
    if st.session_state.get('last_fig'):
        st.markdown("---")

        viz_col, info_col = st.columns([3, 1])

        with viz_col:
            st.markdown(f"#### 📊 Result for: *{st.session_state.get('last_query', '')}*")
            st.plotly_chart(st.session_state.last_fig, use_container_width=True)

            # Download as interactive HTML (no extra dependencies)
            buf = BytesIO()
            st.session_state.last_fig.write_html(buf, include_plotlyjs='cdn')
            buf.seek(0)
            st.download_button(
                "📥 Download as HTML",
                data=buf,
                file_name="chat_viz.html",
                mime="text/html",
                use_container_width=True,
                help="Download an interactive Plotly HTML file that can be opened in any browser"
            )

        with info_col:
            st.markdown("#### 📋 Data Summary")
            df = st.session_state.get('last_df')
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True, height=300)
                st.caption(f"Total records: {len(df)}")
            else:
                st.info("No tabular data for this visualization")

            # Show query parsing info
            parsed = st.session_state.viz_engine.parse_query(st.session_state.get('last_query', ''))
            with st.expander("🔍 Parsed Parameters"):
                for k, v in parsed.items():
                    if v:
                        st.markdown(f"**{k}:** `{v}`")

    # Knowledge graph stats
    st.markdown("---")
    st.markdown("### 📦 Knowledge Graph Statistics")

    stats_cols = st.columns(4)
    graph = st.session_state.demo_graph

    with stats_cols[0]:
        total_entities = sum(len(v) for v in graph.entities.values())
        st.metric("Total Entities", total_entities)
    with stats_cols[1]:
        st.metric("Unique Entities", len(graph.entities))
    with stats_cols[2]:
        st.metric("Documents", len(graph.documents))
    with stats_cols[3]:
        mc_count = sum(len(v) for k, v in graph.entities.items() 
                      if 'cocrfeni' in k or 'alcocrfeni' in k or 'crmnfeconi' in k)
        st.metric("Multicomponent Mentions", mc_count)

    # Entity preview
    with st.expander("🔍 Browse Knowledge Graph Entities"):
        entity_data = []
        for ent_norm, entities in graph.entities.items():
            entity_data.append({
                'Entity': ent_norm,
                'Mentions': len(entities),
                'Documents': len(set(e.doc_source for e in entities)),
                'Has Values': any(e.value is not None for e in entities)
            })
        df_entities = pd.DataFrame(entity_data).sort_values('Mentions', ascending=False)
        st.dataframe(df_entities, use_container_width=True)


if __name__ == "__main__":
    main()
