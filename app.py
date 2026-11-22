from pathlib import Path
import json, html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
from streamlit_folium import st_folium
import statsmodels.api as sm
from io import BytesIO

NAVY = '#06264A'
GREEN = '#079DB5'
TURQUOISE = '#20B5C6'
LAVENDER = '#8B83CF'
RED = '#C84459'  # Reserved for fatal victims and adverse rate changes
PALETTE = [NAVY, TEAL, LAVENDER]

st.set_page_config(page_title='Cook County Health | Chicago Community Insights', page_icon='🏥', layout='wide')
ROOT=Path(__file__).parent/'data'
@st.cache_data
def load():
    d=pd.read_csv(ROOT/'chicago_community_area_year_merged.csv')
    h=pd.read_csv(ROOT/'trauma_hospitals.csv')
    geo=json.loads((ROOT/'dashboard_map.geojson').read_text())
    d.columns=d.columns.str.strip()
    for c in ['shot_total','fatal','nonfatal','population','pct_poverty','pct_unemployed','shot_rate_per_10k','per_capita_income','profile_trauma_drive_min','profile_clinics','pct_rent_burdened','eviction_filing_rate']:
        if c in d: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['ca_num','ca_name','year']).drop_duplicates(['ca_num','year'],keep='first')
    d['ca_num']=d.ca_num.astype(int);d['year']=d.year.astype(int)
    for f in geo['features']: f['properties']['ca_num']=int(f['properties']['ca_num'])
    return d,h,geo
D,H,G=load()
# Optional official county boundary overlay. The shooting dataset remains Chicago-only.
# Source: Cook County GIS ArcGIS service, Simplified Cook County Boundary (layer 199).
@st.cache_data(ttl=86400, show_spinner=False)
def load_cook_county_boundary():
    local_boundary = ROOT / 'cook_county_boundary.geojson'
    try:
        if local_boundary.is_file():
            geo = json.loads(local_boundary.read_text(encoding='utf-8'))
        else:
            from urllib.request import Request, urlopen
            from urllib.parse import urlencode
            service = ('https://services.arcgis.com/F7DSX1DSNSiWmOqh/ArcGIS/rest/services/'
                       'Simplified_Cook_County_Boundary/FeatureServer/199/query')
            query = urlencode({'where': '1=1', 'outFields': '*',
                               'returnGeometry': 'true', 'outSR': '4326', 'f': 'geojson'})
            request = Request(service + '?' + query, headers={'User-Agent': 'CookCountyHealthDashboard/1.0'})
            with urlopen(request, timeout=10) as response:
                geo = json.load(response)
        if (geo.get('type') == 'FeatureCollection' and geo.get('features')
                and all(feature.get('geometry') for feature in geo['features'])):
            return geo, None
        return None, 'The county GIS service did not provide a valid boundary.'
    except Exception as exc:
        return None, f'Could not load county boundary: {exc}'

st.markdown("""<style>
/* Cook County Health-inspired theme: respect Streamlit light/dark surface variables. */
:root { --cch-navy:#06264A; --cch-teal:#079DB5; --cch-red:#C84459; }
.block-container { padding-top:1.2rem; max-width:1500px; padding-bottom:2.5rem; }
h1,h2,h3 { letter-spacing:-.025em; }
[data-testid="stSidebar"] {
 background:linear-gradient(170deg,#06264A 0%,#09365C 73%,#0B546C 100%)!important;
 border-right:3px solid #20B5C6;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] label { color:#EDF9FC!important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color:#C7E8EF!important; }
[data-testid="stSidebar"] [data-baseweb="select"] { background:var(--secondary-background-color); border-radius:8px; }
.cch-brand-wrap { border-bottom:2px solid rgba(32,181,198,.7); padding:3px 0 15px; margin-bottom:16px; }
.cch-brand-wrap img { width:100%; max-width:240px; height:auto; border-radius:9px; display:block; }
.cch-brand-wrap .brand-note { font-size:12px; color:#C7E8EF; margin-top:8px; }
[data-testid="stMetric"] {
 border:1px solid rgba(7,157,181,.32); border-left:4px solid #079DB5;
 border-radius:13px; padding:15px 17px;
 background:var(--secondary-background-color);
}
[data-testid="stMetricLabel"] { font-size:.95rem; font-weight:600!important; }
[data-testid="stMetricValue"] { font-size:2.05rem; font-weight:750!important; }
/* Accessible navigation tabs: explicit colors in light AND dark modes.
   Streamlit renders BaseWeb tabs with nested Markdown, which may inherit
   the page theme unless both the tab and its descendants are targeted. */
[data-testid="stTabs"] [role="tablist"],
[data-testid="stTabs"] [data-baseweb="tab-list"] {
 display:flex!important; gap:7px!important; padding:8px!important;
 border-radius:14px!important; background:#06264A!important;
 border:1px solid #1B4C73!important; overflow-x:auto!important;
 scrollbar-width:thin;
}
[data-testid="stTabs"] button[role="tab"],
[data-testid="stTabs"] [data-baseweb="tab"] {
 flex:1 0 auto!important; padding:11px 14px!important; min-height:46px!important;
 border-radius:9px!important; border:1px solid #2D587A!important;
 background:#123C64!important; background-color:#123C64!important;
 color:#FFFFFF!important; -webkit-text-fill-color:#FFFFFF!important;
 opacity:1!important; font-weight:650!important;
 text-shadow:none!important;
}
/* Override any Streamlit theme rules on Markdown labels inside tabs. */
[data-testid="stTabs"] [role="tab"] *,
[data-testid="stTabs"] [data-baseweb="tab"] *,
[data-testid="stTabs"] [role="tab"] [data-testid="stMarkdownContainer"] *,
[data-testid="stTabs"] [role="tab"] p,
[data-testid="stTabs"] [role="tab"] span,
[data-testid="stTabs"] [role="tab"] div {
 color:#FFFFFF!important; -webkit-text-fill-color:#FFFFFF!important;
 opacity:1!important;
}
[data-testid="stTabs"] [role="tab"]:hover,
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
 background:#176486!important; background-color:#176486!important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"],
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
 background:#087F93!important; background-color:#087F93!important;
 border-color:#6CE3EF!important;
}
[data-testid="stTabs"] [role="tab"]:focus-visible {
 outline:3px solid #79EAF2!important; outline-offset:2px!important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display:none!important; }
[data-testid="stTabs"] [role="tabpanel"] { padding-top:1.1rem!important; }
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"] { border-radius:12px; overflow:hidden; }
@media(max-width:900px) {
 [data-testid="stTabs"] [role="tab"] { font-size:.82rem!important; padding:9px 11px!important; }
}
</style>""",unsafe_allow_html=True)

def chart_style(fig, height=None):
    fig.update_layout(font=dict(family='Arial',size=13),
        margin=dict(l=10,r=15,t=25,b=20),
        legend_title_text='',
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showgrid=True,gridcolor='rgba(128,128,128,.17)',zeroline=False)
    fig.update_yaxes(showgrid=False,zeroline=False)
    if height: fig.update_layout(height=height)
    return fig


def fmt(x,dec=0):
    return 'N/A' if pd.isna(x) else f'{x:,.{dec}f}'
def get(row,col): return row.get(col,np.nan)
def pct(x): return f'{fmt(x,1)}%' if pd.notna(x) else 'N/A'
def metric(row,col): return fmt(get(row,col),1 if col in ['shot_rate_per_10k','profile_trauma_drive_min'] else 0)

# Official CCH profile artwork: circular Cook County seal + organization wordmark.
# The URL is an external logo asset; replace with a local approved image for offline use.
CCH_BRAND_IMAGE = 'https://pbs.twimg.com/profile_images/1842254877193596928/7cNH2u0x.jpg'
st.sidebar.markdown(
    f'<div class="cch-brand-wrap"><img src="{CCH_BRAND_IMAGE}" alt="Cook County Health branding" />'
    '<div class="brand-note">Chicago Community Health Insights</div></div>',
    unsafe_allow_html=True,
)
year=st.sidebar.selectbox('Reporting year',sorted(D.year.unique(),reverse=True))
years_available=sorted(D.year.unique().tolist())
range_start,range_end=st.sidebar.select_slider(
    'Historical comparison range', options=years_available,
    value=(years_available[0],years_available[-1]),
    help='Controls the historical trends and the PDF report. The reporting-year map and KPIs remain annual.',
)
D_window=D[D.year.between(range_start,range_end)].copy()

current=D[D.year==year].copy()
areas=sorted(current.ca_name.unique())
if 'selected_area' not in st.session_state: st.session_state.selected_area='All Chicago'
if st.session_state.selected_area not in ['All Chicago']+areas: st.session_state.selected_area='All Chicago'
area=st.sidebar.selectbox('Focus community area',['All Chicago']+areas,key='selected_area')
st.sidebar.caption(f'{D.ca_num.nunique()} Chicago community areas · {D.year.min()}–{D.year.max()} shooting data')
st.title('Chicago Gun Violence & Healthcare Dashboard')
st.caption('Community-level evidence · exploratory associations, not causal predictions')
t1,t2,t3,t4=st.tabs(['🗺️ Overview & Map','📊 Socioeconomic Factors & Regression','🏥 Healthcare & Hospital Impact','🏘️ Neighborhood Comparison'])

row=current[current.ca_name==area].iloc[0] if area!='All Chicago' else None
selected=current if row is None else current[current.ca_name==area]
shot=selected.shot_total.sum();fatal=selected.fatal.sum();nonfatal=selected.nonfatal.sum();pop=selected.population.sum()
rate=shot/pop*10000 if pop>0 else np.nan

with t1:
    st.subheader(f'{area} · {year}')
    a,b,c,d=st.columns(4)
    a.metric('Shooting victims',fmt(shot));b.metric('Fatal victims',fmt(fatal));c.metric('Nonfatal victims',fmt(nonfatal));d.metric('Victims per 10k',fmt(rate,1))
    st.markdown('#### Community insights at a glance')
    valid_rate=current.dropna(subset=['shot_rate_per_10k'])
    if not valid_rate.empty:
        highest=valid_rate.loc[valid_rate.shot_rate_per_10k.idxmax()]
        insight1=(f"**Highest recorded victim rate:** {highest.ca_name} "
                  f"({highest.shot_rate_per_10k:.1f} per 10k in {year}).")
    else:
        insight1='**Highest recorded victim rate:** No data available.'
    prior=D[D.year.eq(year-1)][['ca_num','shot_total','population']].copy()
    prior['prior_rate']=np.where(prior.population.gt(0),prior.shot_total/prior.population*10000,np.nan)
    changes=current[['ca_num','ca_name','shot_rate_per_10k']].merge(
        prior[['ca_num','prior_rate']],on='ca_num',how='inner').dropna()
    if not changes.empty:
        changes['rate_drop']=changes['prior_rate']-changes['shot_rate_per_10k']
        best=changes.loc[changes.rate_drop.idxmax()]
        insight2=(f"**Largest year-over-year rate decrease:** {best.ca_name} "
                  f"({best.rate_drop:+.1f} per 10k decrease vs. {year-1}).") if best.rate_drop>0 else                  f'**Year-over-year comparison:** No community area shows a lower rate than {year-1} in the comparable data.'
    else:
        insight2=f'**Year-over-year comparison:** Comparable {year-1} rates are unavailable.'
    valid_drive=current.dropna(subset=['profile_trauma_drive_min'])
    if not valid_drive.empty:
        longest=valid_drive.loc[valid_drive.profile_trauma_drive_min.idxmax()]
        insight3=(f"**Longest estimated trauma-center drive:** {longest.ca_name} "
                  f"({longest.profile_trauma_drive_min:.1f} min; not EMS response time).")
    else:
        insight3='**Estimated trauma-center drive:** No data available.'
    for insight in (insight1,insight2,insight3):
        st.markdown('- '+insight)
    st.caption('Automatically calculated from your selected year; descriptive observations, not validated public-health recommendations.')
    left,right=st.columns([1.55,1],gap='large')
    with left:
        st.subheader('Where are shooting victims concentrated?')
        choices={'Victims per 10k':'shot_rate_per_10k','Total shooting victims':'shot_total','Poverty (%)':'pct_poverty','Unemployment (%)':'pct_unemployed','Trauma drive time (min)':'profile_trauma_drive_min'}
        choice=st.radio('Map shading',list(choices),horizontal=True)
        field=choices[choice]
        values=current.set_index('ca_num')[field].to_dict()
        v=pd.Series(list(values.values())).dropna()
        vmin=float(v.min()) if len(v) else 0; vmax=float(v.max()) if len(v) else 1
        import branca.colormap as bcm
        cmap=bcm.LinearColormap(['#fff7bc','#fec44f','#f03b20','#800026'],vmin=vmin,vmax=max(vmax,vmin+.01),caption=choice)
        show_trauma_on_main=st.checkbox('Show listed trauma hospitals on this map',value=False,key='main_trauma_markers')
        show_county=st.checkbox('Show Cook County boundary', value=True, key='cook_county_boundary')
        county_view=st.checkbox('Zoom out to all of Cook County', value=False, key='cook_county_view',
                                help='Displays county geography; shooting figures still describe Chicago only.')
        mp=folium.Map(location=[41.84,-87.68],zoom_start=10,tiles='OpenStreetMap',control_scale=True)
        county_geo, county_error = (load_cook_county_boundary() if (show_county or county_view)
                                    else (None, None))
        if county_geo:
            if county_view:
                # GeoJson bounds are computed from the actual county geometry, not estimated coordinates.
                bounds=folium.GeoJson(county_geo).get_bounds()
                if bounds:
                    mp.fit_bounds(bounds, padding=(15,15))
        elif county_error:
            st.info('Cook County boundary temporarily unavailable. The Chicago community map still works. '
                    'You can also place cook_county_boundary.geojson in the data/ folder for offline use.')
        st.caption('Purple dashed line = Cook County boundary; colored polygons = Chicago community areas with available data. '
                   'No county-wide shooting estimates are implied.')
        lookup=current.set_index('ca_num')
        def style(feature):
            cid=int(feature['properties']['ca_num']);val=values.get(cid,np.nan)
            return {'fillColor':cmap(val) if pd.notna(val) else '#cbd5e1','color':'#202c3b' if row is not None and int(row.ca_num)==cid else '#657181','weight':3 if row is not None and int(row.ca_num)==cid else .8,'fillOpacity':.78}
        for feature in G['features']:
            cid=int(feature['properties']['ca_num'])
            if cid not in lookup.index: continue
            r=lookup.loc[cid]
            fields=[('Shooting victims',fmt(r.shot_total)),('Fatal victims',fmt(r.fatal)),('Nonfatal victims',fmt(r.nonfatal)),('Victims per 10k',fmt(r.shot_rate_per_10k,1)),('Poverty (%)',pct(r.pct_poverty)),('Unemployment (%)',pct(r.pct_unemployed)),('Income per person', '$'+fmt(r.per_capita_income)),('Eviction filing rate',fmt(get(r,'eviction_filing_rate'),2)),('Estimated trauma drive (min)',fmt(get(r,'profile_trauma_drive_min'),1)),('Clinics (source estimate)',fmt(get(r,'profile_clinics'))),('Nearest listed trauma center',str(get(r,'profile_nearest_center'))),('Medicaid reliance (%)',pct(get(r,'pct_on_medicaid')))]
            details=''.join(f'<tr><td style="padding:3px 8px">{html.escape(k)}</td><td><b>{html.escape(str(val))}</b></td></tr>' for k,val in fields)
            popup=folium.Popup(f'<div style="font:13px Arial;width:315px"><h3>{html.escape(str(r.ca_name))} · {year}</h3><table>{details}</table><p style="font-size:11px">Source definitions and years require verification. Travel time is estimated, not EMS response time.</p></div>',max_width=380)
            folium.GeoJson(feature,style_function=style,highlight_function=lambda f:{'weight':3,'color':'#111827','fillOpacity':.88},tooltip=f'{r.ca_name}: {choice} {fmt(get(r,field),1)}',popup=popup).add_to(mp)
        # Draw the county outline on top of the Chicago polygons for visibility.
        if show_county and county_geo:
            folium.GeoJson(
                county_geo, name='Cook County boundary (outline)',
                style_function=lambda f: {'color':'#6D4AC6','weight':4,'opacity':1,
                                          'fillOpacity':0,'dashArray':'9,5'},
                tooltip='Cook County boundary'
            ).add_to(mp)
        if show_trauma_on_main and {'hospital','lat','lon'}.issubset(H.columns):
            trauma_overlay=folium.FeatureGroup(name='Listed hospitals',show=True)
            for _, hosp in H.iterrows():
                lat=pd.to_numeric(hosp.get('lat'),errors='coerce')
                lon=pd.to_numeric(hosp.get('lon'),errors='coerce')
                if pd.isna(lat) or pd.isna(lon):
                    continue
                name=str(hosp.get('hospital','Listed hospital'))
                is_stroger=('stroger' in name.lower() or 'cook county hospital' in name.lower())
                folium.CircleMarker(
                    location=[float(lat),float(lon)],radius=7 if is_stroger else 5,
                    color=NAVY if is_stroger else TEAL,fill=True,fill_opacity=.95,
                    tooltip=name,
                    popup=folium.Popup(html.escape(name)+'<br>Location from hospital dataset',max_width=280)
                ).add_to(trauma_overlay)
            trauma_overlay.add_to(mp)
        # Custom vertical map legend, positioned INSIDE the map on its right edge.
        # Do not add cmap to the map: its default horizontal legend sits on top.
        def legend_number(x):
            return f'{x:,.0f}' if abs(x) >= 100 else f'{x:,.1f}'

        legend_ticks = [vmax - (vmax - vmin) * fraction for fraction in (0, .25, .5, .75, 1)]
        legend_labels = ''.join(
            f'<span>{html.escape(legend_number(number))}</span>'
            for number in legend_ticks
        )
        vertical_legend = f"""
        <style>
          #cch-map-legend {{
            position:fixed; z-index:9999; top:50%; right:14px;
            transform:translateY(-50%); box-sizing:border-box;
            background:transparent; border:none; border-radius:0; padding:0;
            box-shadow:none; font:12px Arial,sans-serif; color:#06264A;
          }}
          #cch-map-legend .legend-title {{
            font-weight:700; font-size:11px; line-height:1.25;
            margin-bottom:8px; text-shadow:0 1px 2px rgba(255,255,255,.9);
          }}
          #cch-map-legend .legend-body {{display:flex;gap:6px;height:320px;align-items:stretch;}}
          #cch-map-legend .legend-gradient {{
            width:12px; min-width:12px; height:100%; border-radius:3px;
            border:1px solid rgba(6,38,74,.18);
            background:linear-gradient(to bottom,#800026 0%,#f03b20 34%,#fec44f 68%,#fff7bc 100%);
          }}
          #cch-map-legend .legend-ticks {{
            display:flex;flex-direction:column;justify-content:space-between;
            text-align:left;white-space:nowrap;font-size:11px;
            text-shadow:0 1px 2px rgba(255,255,255,.92);
          }}
          @media(max-width:560px) {{
            #cch-map-legend {{right:8px;}}
            #cch-map-legend .legend-body {{height:240px;}}
            #cch-map-legend .legend-title {{font-size:10px;}}
            #cch-map-legend .legend-gradient {{width:10px; min-width:10px;}}
          }}
        </style>
        <div id="cch-map-legend" aria-label="Vertical map legend">
          <div class="legend-title">{html.escape(choice)}</div>
          <div class="legend-body">
            <div class="legend-gradient"></div>
            <div class="legend-ticks">{legend_labels}</div>
          </div>
        </div>
        """
        mp.get_root().html.add_child(folium.Element(vertical_legend))
        st_folium(mp,height=520,use_container_width=True,key=f'community_map_{year}_{choice}',returned_objects=['last_active_drawing'])
        st.caption('Click any Chicago community polygon for its detailed popup. The dashed purple boundary shows Cook County, including suburban areas not covered by these shooting datasets. Use the left selector to update all charts.')
    with right:
        st.subheader('Top 10 community areas')
        top=current.nlargest(10,'shot_total').sort_values('shot_total')
        fig=px.bar(top,x='shot_total',y='ca_name',orientation='h',labels={'shot_total':'Shooting victims','ca_name':''},height=490)
        fig.update_traces(marker_color=TEAL);fig.update_layout(margin=dict(l=0,r=0,t=5,b=0),showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    if row is not None:
        st.subheader(f'{area}: socioeconomic & healthcare snapshot')
        cols=st.columns(4)
        for col,(label,key) in zip(cols,[('Poverty', 'pct_poverty'),('Unemployment','pct_unemployed'),('Estimated trauma drive (min)','profile_trauma_drive_min'),('Clinics (estimate)','profile_clinics')]):
            col.metric(label,fmt(get(row,key),1))
    st.subheader('How have shootings changed over time?')
    trend=D if row is None else D[D.ca_name==area]
    trend=trend.groupby('year',as_index=False)[['fatal','nonfatal']].sum()
    fig=px.bar(trend,x='year',y=['fatal','nonfatal'],barmode='stack',color_discrete_map={'fatal':RED,'nonfatal':TEAL},labels={'value':'Victims','variable':'Outcome'},height=330)
    fig.update_layout(xaxis=dict(dtick=1),margin=dict(l=0,r=0,t=5,b=0));st.plotly_chart(fig,use_container_width=True)

with t2:
    st.subheader('Socioeconomic factors and gun violence')
    indicators={'Poverty (%)':'pct_poverty','Unemployment (%)':'pct_unemployed','Income per person ($)':'per_capita_income','Rent burden (%)':'pct_rent_burdened','Eviction filing rate':'eviction_filing_rate'}
    factor=st.selectbox('Compare shooting rates with',list(indicators));col=indicators[factor]
    # Short, plain-English definitions; calculated rates come from the source data.
    # We do not recompute them without the original numerator, denominator, or source metadata.
    factor_explanations = {
        'Poverty (%)': (
            'People living below the poverty threshold.',
            '(People below the poverty threshold ÷ people covered by the poverty measure) × 100.'
        ),
        'Unemployment (%)': (
            'People in the labor force who are unemployed and looking for work.',
            '(Unemployed people ÷ total labor force) × 100.'
        ),
        'Income per person ($)': (
            'Average income per person, not per household.',
            'Total income of residents ÷ total population.'
        ),
        'Rent burden (%)': (
            'Renter households spending a large share of income on housing, usually 30% or more.',
            '(Rent-burdened renter households ÷ renter households assessed) × 100.'
        ),
        'Eviction filing rate': (
            'How frequently eviction cases are filed against renter households.',
            'Eviction filings ÷ renter households; the dataset source determines whether the result is per 100, per 1,000, or another unit.'
        ),
    }
    meaning, calculation = factor_explanations[factor]
    st.info(f'**{factor} — Meaning:** {meaning}  \n**How it is calculated:** {calculation}')
    st.caption('Definitions are general. Confirm each indicator’s original source, reference year and denominator before interpreting its reported values.')
    with st.expander('See definitions and calculations for all five factors'):
        for factor_name, (short_meaning, short_formula) in factor_explanations.items():
            st.markdown(f'**{factor_name}:** {short_meaning}  \n*Calculation:* {short_formula}')
    valid=current.dropna(subset=[col,'shot_rate_per_10k'])
    fig=px.scatter(valid,x=col,y='shot_rate_per_10k',size='shot_total',hover_name='ca_name',color='shot_rate_per_10k',color_continuous_scale=[(0,'#DDF2F6'),(.5,TEAL),(1,RED)],labels={col:factor,'shot_rate_per_10k':'Shooting victims per 10,000'},height=410)
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Each bubble is one community area. Association does not establish that the factor causes shootings.')
    st.subheader('Exploratory multivariate regression')
    st.write('Poisson count regression with a population offset and year indicators. Results are descriptive and require validation; repeated neighborhood observations and historical indicator timing can affect inference.')
    predictors=['pct_poverty','pct_unemployed']
    model_data=D[['ca_name','year','shot_total','population']+predictors].dropna().copy()
    model_data=model_data[(model_data.population>0)&(model_data.shot_total>=0)]
    try:
        X=pd.concat([model_data[predictors].astype(float),pd.get_dummies(model_data.year.astype(str),prefix='year',drop_first=True,dtype=float)],axis=1)
        X=sm.add_constant(X)
        model=sm.GLM(model_data.shot_total.astype(float),X,family=sm.families.Poisson(),offset=np.log(model_data.population.astype(float))).fit(cov_type='HC1')
        model_data['estimated']=model.predict(X,offset=np.log(model_data.population))
        dispersion=model.pearson_chi2/model.df_resid
        m1,m2,m3=st.columns(3)
        m1.metric('Observations',f'{len(model_data):,}');m2.metric('Pearson dispersion',fmt(dispersion,2));m3.metric('Predictors','Poverty + unemployment')
        if dispersion>1.5:
            st.warning('Overdispersion detected. Poisson standard errors and uncertainty may be unreliable.')
        with st.expander('What do Poisson regression and rate ratios mean?'):
            st.write('Poisson regression estimates how a count (shooting victims) is associated with measured factors while accounting for the population at risk. A rate ratio of 1.10 corresponds to a fitted rate about 10% higher for a one-percentage-point increase in that factor, holding the other predictors and year indicators constant. A ratio of 0.90 corresponds to about 10% lower. These are not causal effects or reliable forecasts.')
            st.write('Because some socioeconomic estimates are repeated across years and shooting counts may be overdispersed, the model should not be used for clinical, emergency, or policy decisions without independent validation.')
        if dispersion>1.5:
            with st.expander('Exploratory negative-binomial comparison'):
                try:
                    from statsmodels.discrete.discrete_model import NegativeBinomial
                    nb=NegativeBinomial(model_data.shot_total.astype(float),X,
                                        offset=np.log(model_data.population.astype(float))).fit(
                                        disp=False,maxiter=250)
                    st.write(f'Poisson AIC: {model.aic:,.1f} | Negative-binomial AIC: {nb.aic:,.1f}')
                    st.caption('Lower AIC suggests better in-sample fit under the fitted model assumptions, not validated prediction. Repeated community profiles and year effects require careful review.')
                    nb_ratios=pd.DataFrame({'Factor':predictors,
                        'Negative-binomial rate ratio':np.exp(nb.params[predictors].values)})
                    st.dataframe(nb_ratios.round(3),hide_index=True,use_container_width=True)
                except Exception as nb_error:
                    st.info(f'Negative-binomial comparison unavailable for this dataset: {nb_error}')
        result=pd.DataFrame({'Factor':['Poverty (+1 percentage point)','Unemployment (+1 percentage point)'],'Rate ratio':np.exp(model.params[predictors].to_numpy()),'95% CI lower':np.exp(model.conf_int().loc[predictors,0].to_numpy()),'95% CI upper':np.exp(model.conf_int().loc[predictors,1].to_numpy())})
        st.dataframe(result.round(3),hide_index=True,use_container_width=True)
        display=model_data[model_data.year==year].copy()
        fig=px.scatter(display,x='estimated',y='shot_total',hover_name='ca_name',labels={'estimated':'Model-estimated victims','shot_total':'Observed victims'},height=400)
        upper=max(display.estimated.max(),display.shot_total.max());fig.add_shape(type='line',x0=0,y0=0,x1=upper,y1=upper,line=dict(color=RED,dash='dash'))
        st.plotly_chart(fig,use_container_width=True)
        st.subheader('What-If explorer — model association only')
        chosen=area if area!='All Chicago' else st.selectbox('Community area for scenario',areas)
        base=current[current.ca_name==chosen].iloc[0]
        if pd.notna(base.pct_poverty) and pd.notna(base.pct_unemployed):
            c1,c2=st.columns(2)
            p=c1.slider('Hypothetical poverty (%)',0.,100.,float(base.pct_poverty),.5)
            u=c2.slider('Hypothetical unemployment (%)',0.,100.,float(base.pct_unemployed),.5)
            factor_change=np.exp(model.params['pct_poverty']*(p-base.pct_poverty)+model.params['pct_unemployed']*(u-base.pct_unemployed))
            st.metric('Relative fitted rate vs. baseline',f'{factor_change:.2f}×')
            st.caption('Changing sliders does not demonstrate that an intervention would change shootings. This is a fitted statistical association, not a causal forecast.')
    except Exception as exc:
        st.error(f'Regression could not be estimated: {exc}')

with t3:
    st.subheader('Healthcare access and hospital impact')
    st.caption('Travel times are estimates, not measured ambulance response or transport times. Hospital costs are hospital-wide, not shooting-specific.')
    if row is not None:
        a,b,c=st.columns(3)
        a.metric('Nearest listed center',str(get(row,'profile_nearest_center')))
        b.metric('Estimated trauma drive (min)',fmt(get(row,'profile_trauma_drive_min'),1))
        c.metric('Clinics (estimate)',fmt(get(row,'profile_clinics')))
    # Interactive hospital map: highlight Stroger without duplicating its CSV entry.
    st.markdown("#### Chicago trauma hospitals")
    show_other = st.checkbox("Show other listed hospitals", value=True, key="show_other_hospitals")
    focus_stroger = st.checkbox("Zoom to Stroger Hospital", value=False, key="focus_stroger")
    stroger_lat, stroger_lon = 41.8564, -87.6663  # Approximate map coordinates
    hospital_map = folium.Map(
        location=[stroger_lat, stroger_lon] if focus_stroger else [41.84, -87.68],
        zoom_start=13 if focus_stroger else 10,
        tiles="OpenStreetMap", control_scale=True
    )
    other_layer = folium.FeatureGroup(name="Other listed hospitals", show=show_other)
    stroger_layer = folium.FeatureGroup(name="Cook County / Stroger Hospital", show=True)

    # Use one dedicated marker for Stroger, even if the CSV already contains it.
    for _, hospital in H.iterrows():
        hospital_name = str(hospital.get("hospital", "Unknown hospital"))
        if any(alias in hospital_name.lower() for alias in
               ("stroger", "cook county hospital")):
            continue
        if not show_other:
            continue
        lat = pd.to_numeric(hospital.get("lat"), errors="coerce")
        lon = pd.to_numeric(hospital.get("lon"), errors="coerce")
        if pd.isna(lat) or pd.isna(lon):
            continue
        hospital_type = html.escape(str(hospital.get("type", "Not specified")))
        trauma = html.escape(str(hospital.get("adult_level1_trauma", "Not specified")))
        popup_html = (
            f'<div style="font-family:Arial;min-width:220px">'
            f'<b>{html.escape(hospital_name)}</b><hr>'
            f'Type: {hospital_type}<br>Adult Level I (dataset): {trauma}</div>'
        )
        folium.Marker(
            [float(lat), float(lon)], tooltip=hospital_name,
            popup=folium.Popup(popup_html, max_width=320),
            icon=folium.Icon(color="cadetblue", icon="plus-sign")
        ).add_to(other_layer)

    stroger_popup = """
    <div style="font-family:Arial,sans-serif;min-width:275px;line-height:1.6;color:#06264A">
      <div style="background:#06264A;color:white;padding:10px;border-radius:7px 7px 0 0">
        <strong>COOK COUNTY HEALTH</strong><br>
        John H. Stroger Jr. Hospital
      </div>
      <div style="padding:10px">
        <b>Adult Level I Trauma Center</b><br>
        1969 W. Ogden Avenue<br>Chicago, IL 60612<br>
        Public teaching hospital · Emergency and trauma care<br>
        <a href="https://cookcountyhealth.org/locations/john-h-stroger-jr-hospital-of-cook-county/" target="_blank" rel="noopener noreferrer">
          Cook County Health — official website</a>
      </div>
    </div>
    """
    folium.Marker(
        [stroger_lat, stroger_lon],
        tooltip="★ Cook County Health — Stroger Hospital",
        popup=folium.Popup(stroger_popup, max_width=360),
        icon=folium.Icon(color="darkblue", icon="plus-sign")
    ).add_to(stroger_layer)
    other_layer.add_to(hospital_map)
    stroger_layer.add_to(hospital_map)
    folium.LayerControl(collapsed=False).add_to(hospital_map)
    st_folium(
        hospital_map, height=470, use_container_width=True,
        key=f"hospital_map_{focus_stroger}_{show_other}", returned_objects=[]
    )
    st.caption("Navy: Stroger Hospital · Turquoise-blue: other hospitals in the dataset. "
               "Map coordinates are approximate; confirm locations and trauma designations before publication.")

    st.divider()
    st.subheader("Cook County Health | John H. Stroger Jr. Hospital")
    st.caption("Featured hospital · Chicago, Illinois")
    k1, k2, k3 = st.columns(3)
    k1.metric("Trauma designation", "Adult Level I")
    k2.metric("Healthcare system", "Cook County Health")
    k3.metric("Location", "Chicago, IL")
    st.markdown(
        "**John H. Stroger Jr. Hospital of Cook County** is a public teaching "
        "hospital providing emergency and trauma services. "
        "**Address:** 1969 W. Ogden Avenue, Chicago, IL 60612. "
        "[Official Stroger Hospital page](https://cookcountyhealth.org/locations/john-h-stroger-jr-hospital-of-cook-county/)"
    )
    st.info("The neighborhood shooting totals in this dashboard are not counts of "
            "patients treated at Stroger Hospital. Hospital-specific firearm injury "
            "counts have not been verified or added.")
    stroger_data = H[H["hospital"].astype(str).str.contains(
        r"Stroger|Cook County Hospital", case=False, regex=True, na=False
    )]
    finance_cols = [c for c in (
        "hospital", "total_costs", "net_patient_revenue",
        "charity_care_cost", "uncompensated_care_cost"
    ) if c in stroger_data.columns]
    if not stroger_data.empty and len(finance_cols) > 1:
        st.markdown("**Stroger financial figures available in the existing dataset**")
        st.dataframe(stroger_data[finance_cols], hide_index=True, use_container_width=True)
        st.caption("These are hospital-wide financial fields, not firearm-injury-specific costs. "
                   "Verify reporting periods and original sources before presenting.")
    else:
        st.caption("No Stroger-specific financial record is available in the current hospital CSV.")

    st.subheader('Hospital financial comparison')
    fields=['hospital','type','total_costs','net_patient_revenue','charity_care_cost','uncompensated_care_cost']
    st.dataframe(H[[c for c in fields if c in H.columns]],hide_index=True,use_container_width=True)
    financial=H.dropna(subset=['total_costs']).sort_values('total_costs')
    if len(financial):
        fig=px.bar(financial,x='total_costs',y='hospital',orientation='h',labels={'total_costs':'Hospital-wide total costs ($)','hospital':''},height=400)
        st.plotly_chart(fig,use_container_width=True)
    st.subheader('Estimated access compared with shooting rates')
    access_preview=current.dropna(subset=['profile_trauma_drive_min','shot_rate_per_10k'])
    if not access_preview.empty:
        hospital_scatter=px.scatter(access_preview,x='profile_trauma_drive_min',
            y='shot_rate_per_10k',hover_name='ca_name',
            labels={'profile_trauma_drive_min':'Estimated drive to listed trauma center (minutes)',
                    'shot_rate_per_10k':'Shooting victims per 10k'},
            color_discrete_sequence=[TEAL],height=360)
        st.plotly_chart(chart_style(hospital_scatter),use_container_width=True)
    st.caption('Driving minutes are dataset estimates, not ambulance transport or EMS response times.')
    st.subheader('Neighborhood healthcare comparison')
    st.dataframe(current[['ca_name','shot_rate_per_10k','profile_nearest_center','profile_trauma_drive_min','profile_clinics']].sort_values('shot_rate_per_10k',ascending=False),hide_index=True,use_container_width=True)

with t4:
    st.subheader('Neighborhood Comparison')
    st.markdown('Compare neighborhood safety trends, socioeconomic conditions and estimated trauma-center access in one place.')
    st.caption(f'Explore community differences across Chicago in {year}. Compare rates rather than counts when populations differ.')

    measures = {
        'Shooting victims per 10,000': 'shot_rate_per_10k',
        'Total shooting victims': 'shot_total',
        'Poverty (%)': 'pct_poverty',
        'Unemployment (%)': 'pct_unemployed',
        'Income per person ($)': 'per_capita_income',
        'Estimated trauma drive (min)': 'profile_trauma_drive_min',
    }
    c_filter, c_sort = st.columns([1.6, 1])
    with c_filter:
        neighborhoods = st.multiselect('Neighborhoods to include (leave empty for all)', areas, default=[area] if area != 'All Chicago' else [], key='compare_filter')
    with c_sort:
        measure_label = st.selectbox('Compare and rank by', list(measures), key='compare_measure')
    measure = measures[measure_label]
    subset = current[current.ca_name.isin(neighborhoods)].copy() if neighborhoods else current.copy()
    subset = subset.dropna(subset=[measure]).sort_values(measure, ascending=False)

    if subset.empty:
        st.info('No neighborhood records are available for this selection.')
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric('Areas compared', f'{len(subset):,}')
        k2.metric('Median ' + measure_label, fmt(subset[measure].median(), 1 if measure != 'shot_total' else 0))
        k3.metric('Highest: ' + str(subset.iloc[0]['ca_name']), fmt(subset.iloc[0][measure], 1 if measure != 'shot_total' else 0))

        chart_col, scatter_col = st.columns(2, gap='large')
        with chart_col:
            st.markdown('**Highest 10 areas by selected indicator**')
            top = subset.head(10).sort_values(measure)
            bar = px.bar(top, x=measure, y='ca_name', orientation='h',
                         labels={measure:measure_label, 'ca_name':''},
                         color=measure, color_continuous_scale=[(0,'#DDF2F6'),(.5,TEAL),(1,RED)], height=400)
            bar.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(l=0,r=8,t=5,b=0))
            st.plotly_chart(bar, use_container_width=True)
        with scatter_col:
            st.markdown('**Poverty vs. shooting victim rate**')
            scatter_data = subset.dropna(subset=['pct_poverty','shot_rate_per_10k'])
            if scatter_data.empty:
                st.info('Data unavailable for this chart.')
            else:
                scatter = px.scatter(scatter_data, x='pct_poverty', y='shot_rate_per_10k',
                                     hover_name='ca_name', color='pct_unemployed',
                                     color_continuous_scale=[(0,'#DDF2F6'),(.5,TEAL),(1,RED)],
                                     labels={'pct_poverty':'Poverty (%)',
                                             'shot_rate_per_10k':'Victims per 10,000',
                                             'pct_unemployed':'Unemployment (%)'},
                                     height=400)
                scatter.update_layout(margin=dict(l=0,r=8,t=5,b=0))
                st.plotly_chart(scatter, use_container_width=True)
        st.caption('Charts show descriptive comparisons, not evidence of cause and effect. Socioeconomic source years may differ from the reporting year.')

        st.markdown('**Compare two community areas side by side**')
        compare_names = sorted(subset.ca_name.unique())
        if len(compare_names) >= 2:
            col_a, col_b = st.columns(2)
            first = col_a.selectbox('Community A', compare_names, index=0, key='compare_a')
            second = col_b.selectbox('Community B', compare_names, index=1, key='compare_b')
            pair = current[current.ca_name.isin([first, second])].copy()
            pair_display = pair[['ca_name','shot_total','shot_rate_per_10k','pct_poverty','pct_unemployed','per_capita_income']].set_index('ca_name').T
            pair_display.index = ['Shooting victims','Victims per 10,000','Poverty (%)','Unemployment (%)','Income per person ($)']
            st.dataframe(pair_display.style.format('{:,.1f}', na_rep='N/A'), use_container_width=True)
            if first == second:
                st.caption('Choose two different communities to see a side-by-side comparison.')
        else:
            st.info('Select at least two neighborhoods above to compare them side by side.')

        st.divider()
        st.subheader('Year-over-year change in shooting victim rates')
        st.caption('Changes compare each community with the immediately preceding calendar year. Rates use the population denominator in each respective year; missing data is not treated as zero.')
        previous = D[D.year == year - 1][['ca_num','shot_total','population']].copy()
        previous = previous.rename(columns={'shot_total':'previous_victims','population':'previous_population'})
        yoy = subset.merge(previous,on='ca_num',how='left')
        yoy['prior_rate'] = np.where(yoy['previous_population'] > 0,
                                      yoy['previous_victims'] / yoy['previous_population'] * 10000, np.nan)
        yoy['rate_change'] = yoy['shot_rate_per_10k'] - yoy['prior_rate']
        yoy['rate_change_pct'] = np.where(yoy['prior_rate'] > 0,
                                          100 * yoy['rate_change'] / yoy['prior_rate'],np.nan)
        yoy_valid = yoy.dropna(subset=['shot_rate_per_10k','prior_rate','rate_change'])
        if yoy_valid.empty:
            st.info(f'No comparable {year - 1} data is available for the chosen communities. Change the reporting year or selected areas.')
        else:
            improved = int((yoy_valid.rate_change < 0).sum())
            increased = int((yoy_valid.rate_change > 0).sum())
            unchanged = int((yoy_valid.rate_change == 0).sum())
            yc1,yc2,yc3 = st.columns(3)
            yc1.metric('Rates decreased', f'{improved} areas')
            yc2.metric('Rates increased', f'{increased} areas')
            yc3.metric('No change', f'{unchanged} areas')
            top_changes = yoy_valid.reindex(yoy_valid.rate_change.abs().sort_values(ascending=False).index).head(15)
            top_changes = top_changes.sort_values('rate_change')
            diff = go.Figure(go.Bar(x=top_changes.rate_change, y=top_changes.ca_name,
                                    orientation='h',
                                    marker_color=[TEAL if v < 0 else RED if v > 0 else NAVY for v in top_changes.rate_change],
                                    customdata=top_changes[['prior_rate','shot_rate_per_10k']].to_numpy(),
                                    hovertemplate='%{y}<br>Change: %{x:+.2f} victims/10k<br>Prior: %{customdata[0]:.2f}<br>Current: %{customdata[1]:.2f}<extra></extra>'))
            diff.update_layout(xaxis_title=f'Change in victims per 10,000 ({year - 1} → {year})',yaxis_title='',height=460)
            st.plotly_chart(chart_style(diff),use_container_width=True)
            st.caption('Teal indicates a lower victim rate than the previous year; red indicates a higher rate. The chart ranks areas by largest absolute rate change.')
            yoy_table = yoy_valid[['ca_name','prior_rate','shot_rate_per_10k','rate_change','rate_change_pct']].copy()
            yoy_table.columns=['Community area',f'{year-1} rate',f'{year} rate','Change per 10,000','Change (%)']
            st.dataframe(yoy_table.sort_values('Change per 10,000',ascending=False).round(2),
                         hide_index=True,use_container_width=True)

        st.divider()
        st.subheader('Two-community trend comparison')
        st.caption(f'Track annual shooting victim rates in the selected {range_start}–{range_end} historical comparison range.')
        trend_areas = sorted(D.ca_name.dropna().unique().tolist())
        ta,tb = st.columns(2)
        first_trend = ta.selectbox('Trend: community A',trend_areas,
             index=trend_areas.index(area) if area in trend_areas else 0,key='trend_a')
        default_b = next((i for i,n in enumerate(trend_areas) if n != first_trend),0)
        second_trend = tb.selectbox('Trend: community B',trend_areas,index=default_b,key='trend_b')
        view_metric=st.radio('Trend measure', ['Victims per 10,000','Total shooting victims'],
                             horizontal=True,key='trend_metric')
        trend_column='shot_rate_per_10k' if view_metric=='Victims per 10,000' else 'shot_total'
        if first_trend==second_trend:
            st.info('Choose different communities for two separate trend lines.')
        historical=D_window[D_window.ca_name.isin([first_trend,second_trend])].sort_values('year').copy()
        historical=historical.dropna(subset=[trend_column])
        if historical.empty:
            st.info('No annual data is available for this trend comparison.')
        else:
            lines=px.line(historical,x='year',y=trend_column,color='ca_name',markers=True,
                color_discrete_map={first_trend:TEAL,second_trend:RED},
                labels={'year':'Reporting year','ca_name':'Community area',trend_column:view_metric})
            lines.update_layout(height=420,xaxis=dict(dtick=1),hovermode='x unified')
            st.plotly_chart(chart_style(lines),use_container_width=True)
            st.caption('A change in victim rate does not establish why a community changed. The trend uses the available yearly records only.')

        st.markdown('**Citywide historical trend for your selected range**')
        city_history=D_window.dropna(subset=['shot_total','population']).groupby('year',as_index=False)[['shot_total','population']].sum()
        city_history=city_history[city_history.population.gt(0)].copy()
        city_history['rate']=city_history.shot_total/city_history.population*10000
        if not city_history.empty:
            city_chart=px.line(city_history,x='year',y='rate',markers=True,
                labels={'year':'Year','rate':'Citywide victims per 10k'},
                color_discrete_sequence=[TEAL],height=290)
            st.plotly_chart(chart_style(city_chart),use_container_width=True)
            st.caption('Citywide yearly rates = summed victims ÷ summed population × 10,000. These are annual values, not cumulative totals.')

        st.divider()
        st.subheader('Healthcare accessibility versus shooting victim rates')
        st.caption('Compare neighborhood-level estimated driving minutes to the nearest listed trauma center with shooting victim rates. Travel estimates are not EMS response or transport times.')
        access = subset.dropna(subset=['profile_trauma_drive_min','shot_rate_per_10k']).copy()
        if access.empty:
            st.info('Estimated trauma travel data is unavailable for these communities.')
        else:
            ha,hb,hc = st.columns(3)
            ha.metric('Areas with access data',f'{len(access):,}')
            hb.metric('Median estimated drive',f'{access.profile_trauma_drive_min.median():.1f} min')
            hc.metric('Longest estimated drive',f'{access.profile_trauma_drive_min.max():.1f} min')
            healthcare = px.scatter(access,x='profile_trauma_drive_min',y='shot_rate_per_10k',
                                    hover_name='ca_name',size='shot_total',
                                    color='shot_rate_per_10k',
                                    color_continuous_scale=[(0,'#DDF2F6'),(.5,TEAL),(1,RED)],
                                    labels={'profile_trauma_drive_min':'Estimated trauma center drive (minutes)',
                                            'shot_rate_per_10k':'Shooting victims per 10,000'},
                                    hover_data={'profile_nearest_center':True} if 'profile_nearest_center' in access.columns else None,
                                    height=440)
            st.plotly_chart(chart_style(healthcare),use_container_width=True)
            access_table_columns = ['ca_name','shot_rate_per_10k','profile_trauma_drive_min']
            if 'profile_nearest_center' in access.columns:
                access_table_columns.append('profile_nearest_center')
            access_table=access[access_table_columns].sort_values('profile_trauma_drive_min',ascending=False)
            st.dataframe(access_table.rename(columns={
                'ca_name':'Community area','shot_rate_per_10k':'Victims per 10,000',
                'profile_trauma_drive_min':'Estimated drive (min)',
                'profile_nearest_center':'Nearest listed trauma center'}),
                hide_index=True,use_container_width=True)
            st.caption('Drive-time estimates and location listings need source verification. No causal relationship or emergency-response performance is implied.')

        st.markdown('**Full neighborhood data**')
        display_cols = {
            'ca_name':'Community area',
            'shot_total':'Shooting victims',
            'shot_rate_per_10k':'Victims per 10,000',
            'pct_poverty':'Poverty (%)',
            'pct_unemployed':'Unemployment (%)',
            'per_capita_income':'Income per person ($)',
            'profile_trauma_drive_min':'Estimated trauma drive (min)'
        }
        full = subset[list(display_cols)].rename(columns=display_cols)
        st.dataframe(full, hide_index=True, use_container_width=True,
                     column_config={
                         'Shooting victims':st.column_config.NumberColumn(format='%d'),
                         'Victims per 10,000':st.column_config.NumberColumn(format='%.1f'),
                         'Poverty (%)':st.column_config.NumberColumn(format='%.1f%%'),
                         'Unemployment (%)':st.column_config.NumberColumn(format='%.1f%%'),
                         'Income per person ($)':st.column_config.NumberColumn(format='$%.0f'),
                         'Estimated trauma drive (min)':st.column_config.NumberColumn(format='%.1f')
                     })
        st.download_button('Download comparison CSV',
                           full.to_csv(index=False).encode('utf-8'),
                           file_name=f'cook_county_neighborhood_comparison_{year}.csv',
                           mime='text/csv', key='download_comparison')


# Client-ready PDF report using the same selection and currently loaded dataset.
def create_client_pdf(data, all_history, hospitals, reporting_year, community, start_year, end_year):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import inch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    navy=colors.HexColor(NAVY)
    teal=colors.HexColor(TEAL)
    out=BytesIO()
    pdf=SimpleDocTemplate(out,pagesize=(612,792),leftMargin=42,rightMargin=42,
                          topMargin=42,bottomMargin=42)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CCHTitle',parent=styles['Title'],textColor=navy,fontSize=20,leading=24,spaceAfter=12))
    styles.add(ParagraphStyle(name='CCHHeading',parent=styles['Heading2'],textColor=navy,spaceBefore=14,spaceAfter=7))
    styles.add(ParagraphStyle(name='CCHBody',parent=styles['BodyText'],fontSize=9,leading=13,spaceAfter=7))
    story=[Paragraph('Cook County Health | Community Insights',styles['CCHTitle']),
           Paragraph(f'{html.escape(community)} | Reporting year {reporting_year} | Historical view {start_year}–{end_year}',styles['CCHBody'])]
    result=data if community=='All Chicago' else data[data.ca_name.eq(community)]
    patients=int(result.shot_total.sum())
    fatal_n=int(result.fatal.sum())
    pop_sum=result.population.sum()
    patient_rate=patients/pop_sum*10000 if pd.notna(pop_sum) and pop_sum>0 else np.nan
    kpi_data=[['Shooting victims','Fatal victims','Victims per 10k'],
              [f'{patients:,}',f'{fatal_n:,}',fmt(patient_rate,1)]]
    kpi_table=Table(kpi_data,colWidths=[175]*3,rowHeights=[27,35])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),navy),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('BACKGROUND',(0,1),(-1,1),colors.HexColor('#EAF7FA')),
        ('TEXTCOLOR',(0,1),(-1,1),navy),('FONTSIZE',(0,0),(-1,-1),10),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
    story.extend([kpi_table,Spacer(1,12),Paragraph('Selected-year observations',styles['CCHHeading'])])
    valid=data.dropna(subset=['shot_rate_per_10k'])
    if community=='All Chicago' and not valid.empty:
        top=valid.sort_values('shot_rate_per_10k',ascending=False).iloc[0]
        story.append(Paragraph(f'Highest observed community rate: {html.escape(str(top.ca_name))} ({top.shot_rate_per_10k:.1f} victims per 10k).',styles['CCHBody']))
    elif community!='All Chicago' and not result.empty:
        r=result.iloc[0]
        if pd.notna(r.get('profile_trauma_drive_min',np.nan)):
            story.append(Paragraph(f'Estimated trauma-center driving time: {r.profile_trauma_drive_min:.1f} minutes. Not EMS transport time.',styles['CCHBody']))

    history=all_history[all_history.year.between(start_year,end_year)]
    history=history if community=='All Chicago' else history[history.ca_name.eq(community)]
    history=history.dropna(subset=['shot_total','population']).groupby('year',as_index=False)[['shot_total','population']].sum()
    history=history[history.population.gt(0)].copy()
    if not history.empty:
        history['victim_rate']=history.shot_total/history.population*10000
        fig,ax=plt.subplots(figsize=(7.1,2.5))
        ax.plot(history.year,history.victim_rate,marker='o',color=TEAL,linewidth=2)
        ax.set_ylabel('Victims per 10k'); ax.set_xlabel('Year');ax.set_title('Annual shooting victim rate')
        ax.grid(axis='y',alpha=.25);fig.tight_layout()
        img=BytesIO();fig.savefig(img,format='png',dpi=140,bbox_inches='tight');plt.close(fig);img.seek(0)
        story.extend([Paragraph('Historical trend',styles['CCHHeading']),Image(img,width=505,height=178)])
    if community=='All Chicago':
        top=data.dropna(subset=['shot_rate_per_10k']).sort_values('shot_rate_per_10k',ascending=False).head(6)
        if not top.empty:
            fig,ax=plt.subplots(figsize=(7.0,2.8))
            ax.barh(top.ca_name.iloc[::-1],top.shot_rate_per_10k.iloc[::-1],color=TEAL)
            ax.set_xlabel('Victims per 10k');fig.tight_layout()
            img2=BytesIO();fig.savefig(img2,format='png',dpi=130,bbox_inches='tight');plt.close(fig);img2.seek(0)
            story.extend([Paragraph('Six highest community rates',styles['CCHHeading']),Image(img2,width=505,height=202)])
    story.extend([Spacer(1,9),Paragraph('Data definitions and limitations',styles['CCHHeading']),
        Paragraph('Source inputs: chicago_community_area_year_merged.csv, trauma_hospitals.csv, and dashboard_map.geojson. Original agencies, publication dates, profile-variable reference years, hospital designations and travel-time methods require verification against primary sources.',styles['CCHBody']),
        Paragraph('Victims are counts of injured or killed people, not distinct incidents. Socioeconomic profile variables may repeat across years. Estimated driving times are not EMS response times. Associations are not causal predictions. Hospital financial figures (not shown here) are hospital-wide, not firearm-specific.',styles['CCHBody']),
        Paragraph('Research prototype | Not medical, emergency-response, or validated policy guidance.',styles['CCHBody'])])
    pdf.build(story)
    out.seek(0)
    return out.getvalue()

st.divider()
st.subheader('Client report & data quality')
with st.expander('Data source coverage and quality checks',expanded=False):
    st.markdown('**Files currently used**')
    st.code('data/chicago_community_area_year_merged.csv\ndata/trauma_hospitals.csv\ndata/dashboard_map.geojson',language='text')
    st.caption('These filenames identify the bundled inputs, not verified originating agencies or collection methods.')
    qa_fields=['shot_total','population','shot_rate_per_10k','pct_poverty','pct_unemployed','per_capita_income','profile_trauma_drive_min']
    available=[field for field in qa_fields if field in current.columns]
    missing_data=pd.DataFrame({'Measure':available,
            'Available areas':[int(current[field].notna().sum()) for field in available],
            'Missing areas':[int(current[field].isna().sum()) for field in available]})
    st.dataframe(missing_data,hide_index=True,use_container_width=True)
    st.caption(f'{len(current)} rows in selected year {year}; {D.ca_num.nunique()} unique community area IDs across {D.year.min()}–{D.year.max()}.')
    dupes=int(D.duplicated(['ca_num','year']).sum())
    bad_pops=int((D.population<=0).sum()) if 'population' in D else 0
    st.write(f'Duplicate community-year keys after loading: {dupes}; non-positive population rows: {bad_pops}.')
    st.warning('Verify original data agency, collection period, profile indicator reference years, hospital designations, travel-time calculation, and hospital financial reporting period before publication. Missing records are not assumed to be zero.')

with st.expander('Download client-ready PDF',expanded=False):
    st.write(f'Generate a report for **{area}**, reporting year **{year}**, trend period **{range_start}–{range_end}**. Includes a KPI summary, charts, and limitations.')
    report_selection=(str(area),int(year),int(range_start),int(range_end))
    if st.button('Generate PDF report',key='generate_report'):
        try:
            st.session_state['pdf_report_bytes']=create_client_pdf(current,D,H,year,area,range_start,range_end)
            st.session_state['pdf_report_selection']=report_selection
        except ImportError:
            st.error('PDF export requires reportlab and matplotlib. Add both packages to requirements.txt.')
        except Exception as export_error:
            st.error(f'Could not create the PDF report for this selection: {export_error}')
    if (st.session_state.get('pdf_report_selection') == report_selection
            and st.session_state.get('pdf_report_bytes')):
        st.download_button('Download Cook County Health PDF report',
            data=st.session_state['pdf_report_bytes'],
            file_name=f'cook_county_health_{area.lower().replace(" ","_")}_{year}.pdf',
            mime='application/pdf',key='download_pdf')
    else:
        st.caption('Click Generate PDF report first. Reports are rebuilt when you change the selection.')

with st.expander('Data definitions, limitations & download'):
    st.write('Merged dataset: one row per Chicago community area and shooting year. Profile variables are repeated across years and must not be treated as independent annual observations. Historical 2008–2012 indicators are labeled separately.')
    st.write('Poverty definition, socioeconomic source years, hospital designations, and cost-report figures require primary-source verification. Shooting counts represent victims, not necessarily distinct incidents.')
    st.download_button('Download merged data',D.to_csv(index=False).encode(),'chicago_community_area_year_merged.csv','text/csv')
    st.download_button('Download selected year',current.to_csv(index=False).encode(),f'chicago_{year}.csv','text/csv')
st.caption('Research prototype · Not emergency-response guidance or a validated causal prediction model')
