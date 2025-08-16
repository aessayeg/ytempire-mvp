"""
Advanced Data Visualization Service
Provides sophisticated data visualization capabilities for YTEmpire platform
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import asyncio
import hashlib

# Optional imports for advanced visualizations
try:
    import plotly.graph_objs as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from app.core.config import settings
from app.models.video import Video
from app.models.channel import Channel
from app.models.analytics import Analytics

logger = logging.getLogger(__name__)


class VisualizationType(Enum):
    """Types of visualizations available"""

    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    FUNNEL = "funnel"
    SANKEY = "sankey"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    RADAR = "radar"
    BOX_PLOT = "box_plot"
    VIOLIN_PLOT = "violin_plot"
    WATERFALL = "waterfall"
    CANDLESTICK = "candlestick"
    GEOGRAPHIC_MAP = "geographic_map"
    NETWORK_GRAPH = "network_graph"


@dataclass
class VisualizationConfig:
    """Configuration for a visualization"""

    type: VisualizationType
    title: str
    data_source: str
    filters: Dict[str, Any] = field(default_factory=dict)
    dimensions: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    styling: Dict[str, Any] = field(default_factory=dict)
    interactive: bool = True
    real_time: bool = False
    cache_ttl: int = 300  # seconds


@dataclass
class DashboardLayout:
    """Layout configuration for dashboards"""

    name: str
    grid: List[List[str]]  # Grid layout with visualization IDs
    responsive: bool = True
    refresh_interval: int = 60  # seconds
    filters: List[Dict[str, Any]] = field(default_factory=list)


class AdvancedDataVisualizationService:
    """Service for advanced data visualization"""

    def __init__(self):
        self.visualizations: Dict[str, VisualizationConfig] = {}
        self.dashboards: Dict[str, DashboardLayout] = {}
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self._initialize_default_visualizations()

    def _initialize_default_visualizations(self):
        """Initialize default visualization templates"""
        # Revenue tracking visualization
        self.register_visualization(
            VisualizationConfig(
                type=VisualizationType.LINE_CHART,
                title="Revenue Trend Analysis",
                data_source="revenue_metrics",
                dimensions=["date", "channel"],
                metrics=["revenue", "views", "cpm"],
                styling={"theme": "plotly_dark", "colors": ["#00ff41", "#00d4ff"]},
                real_time=True,
            )
        )

        # Video performance heatmap
        self.register_visualization(
            VisualizationConfig(
                type=VisualizationType.HEATMAP,
                title="Video Performance Heatmap",
                data_source="video_analytics",
                dimensions=["hour_of_day", "day_of_week"],
                metrics=["views", "engagement_rate"],
                styling={"colorscale": "viridis"},
            )
        )

        # Content funnel analysis
        self.register_visualization(
            VisualizationConfig(
                type=VisualizationType.FUNNEL,
                title="Content Generation Funnel",
                data_source="content_pipeline",
                dimensions=["stage"],
                metrics=["count", "conversion_rate"],
                styling={"orientation": "horizontal"},
            )
        )

        # Channel network graph
        self.register_visualization(
            VisualizationConfig(
                type=VisualizationType.NETWORK_GRAPH,
                title="Channel Relationship Network",
                data_source="channel_relationships",
                dimensions=["source", "target"],
                metrics=["weight", "interaction_type"],
                interactive=True,
            )
        )

    def register_visualization(self, config: VisualizationConfig) -> str:
        """Register a new visualization configuration"""
        viz_id = self._generate_viz_id(config)
        self.visualizations[viz_id] = config
        logger.info(f"Registered visualization: {config.title} (ID: {viz_id})")
        return viz_id

    def _generate_viz_id(self, config: VisualizationConfig) -> str:
        """Generate unique ID for visualization"""
        content = f"{config.type.value}_{config.title}_{config.data_source}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    async def create_visualization(
        self,
        viz_id: str,
        db: AsyncSession,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> Dict[str, Any]:
        """Create a visualization based on configuration"""
        if viz_id not in self.visualizations:
            raise ValueError(f"Visualization {viz_id} not found")

        config = self.visualizations[viz_id]

        # Check cache
        if not config.real_time:
            cached_data = self._get_cached_visualization(viz_id)
            if cached_data:
                return cached_data

        # Fetch data
        data = await self._fetch_visualization_data(config, db, time_range)

        # Generate visualization
        if PLOTLY_AVAILABLE:
            viz_output = await self._create_plotly_visualization(config, data)
        else:
            viz_output = await self._create_fallback_visualization(config, data)

        # Cache result
        if not config.real_time:
            self._cache_visualization(viz_id, viz_output, config.cache_ttl)

        return viz_output

    async def _fetch_visualization_data(
        self,
        config: VisualizationConfig,
        db: AsyncSession,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> pd.DataFrame:
        """Fetch data for visualization"""
        # Map data source to query
        if config.data_source == "revenue_metrics":
            return await self._fetch_revenue_data(db, config, time_range)
        elif config.data_source == "video_analytics":
            return await self._fetch_video_analytics(db, config, time_range)
        elif config.data_source == "content_pipeline":
            return await self._fetch_pipeline_data(db, config, time_range)
        elif config.data_source == "channel_relationships":
            return await self._fetch_channel_network(db, config, time_range)
        else:
            return pd.DataFrame()

    async def _fetch_revenue_data(
        self,
        db: AsyncSession,
        config: VisualizationConfig,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> pd.DataFrame:
        """Fetch revenue metrics data"""
        if not time_range:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            time_range = (start_date, end_date)

        # Simulate revenue data (replace with actual query)
        dates = pd.date_range(start=time_range[0], end=time_range[1], freq="D")
        data = {
            "date": dates,
            "revenue": np.random.uniform(100, 1000, len(dates)),
            "views": np.random.randint(1000, 50000, len(dates)),
            "cpm": np.random.uniform(1, 10, len(dates)),
        }

        df = pd.DataFrame(data)

        # Apply filters if any
        for filter_key, filter_value in config.filters.items():
            if filter_key in df.columns:
                df = df[df[filter_key] == filter_value]

        return df

    async def _fetch_video_analytics(
        self,
        db: AsyncSession,
        config: VisualizationConfig,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> pd.DataFrame:
        """Fetch video analytics data"""
        # Create sample heatmap data
        hours = list(range(24))
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        data = []
        for hour in hours:
            for day_idx, day in enumerate(days):
                views = np.random.randint(100, 5000)
                engagement = np.random.uniform(0.1, 0.9)
                data.append(
                    {
                        "hour_of_day": hour,
                        "day_of_week": day,
                        "day_index": day_idx,
                        "views": views,
                        "engagement_rate": engagement,
                    }
                )

        return pd.DataFrame(data)

    async def _fetch_pipeline_data(
        self,
        db: AsyncSession,
        config: VisualizationConfig,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> pd.DataFrame:
        """Fetch content pipeline data"""
        stages = [
            "Trend Analysis",
            "Script Generation",
            "Voice Synthesis",
            "Video Assembly",
            "Quality Check",
            "Publishing",
        ]

        data = []
        remaining = 1000
        for stage in stages:
            conversion_rate = np.random.uniform(0.7, 0.95)
            count = int(remaining * conversion_rate)
            data.append(
                {"stage": stage, "count": count, "conversion_rate": conversion_rate}
            )
            remaining = count

        return pd.DataFrame(data)

    async def _fetch_channel_network(
        self,
        db: AsyncSession,
        config: VisualizationConfig,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> pd.DataFrame:
        """Fetch channel network data"""
        # Create sample network data
        channels = [f"Channel_{i}" for i in range(10)]
        data = []

        for i, source in enumerate(channels):
            # Create connections to other channels
            num_connections = np.random.randint(1, 4)
            targets = np.random.choice(
                [c for c in channels if c != source],
                size=num_connections,
                replace=False,
            )

            for target in targets:
                data.append(
                    {
                        "source": source,
                        "target": target,
                        "weight": np.random.uniform(0.1, 1.0),
                        "interaction_type": np.random.choice(
                            ["collaboration", "competition", "similar_content"]
                        ),
                    }
                )

        return pd.DataFrame(data)

    async def _create_plotly_visualization(
        self, config: VisualizationConfig, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create Plotly visualization"""
        if not PLOTLY_AVAILABLE:
            return await self._create_fallback_visualization(config, data)

        fig = None

        if config.type == VisualizationType.LINE_CHART:
            fig = self._create_line_chart(config, data)
        elif config.type == VisualizationType.BAR_CHART:
            fig = self._create_bar_chart(config, data)
        elif config.type == VisualizationType.HEATMAP:
            fig = self._create_heatmap(config, data)
        elif config.type == VisualizationType.FUNNEL:
            fig = self._create_funnel(config, data)
        elif config.type == VisualizationType.SCATTER_PLOT:
            fig = self._create_scatter_plot(config, data)
        elif config.type == VisualizationType.SANKEY:
            fig = self._create_sankey(config, data)
        elif config.type == VisualizationType.TREEMAP:
            fig = self._create_treemap(config, data)
        elif config.type == VisualizationType.NETWORK_GRAPH:
            fig = self._create_network_graph(config, data)
        else:
            fig = self._create_default_chart(config, data)

        if fig:
            # Apply styling
            fig.update_layout(
                title=config.title,
                template=config.styling.get("theme", "plotly_dark"),
                height=config.styling.get("height", 500),
                width=config.styling.get("width", 800),
            )

            return {
                "type": "plotly",
                "figure": fig.to_dict(),
                "config": {"displayModeBar": config.interactive},
            }

        return await self._create_fallback_visualization(config, data)

    def _create_line_chart(
        self, config: VisualizationConfig, data: pd.DataFrame
    ) -> go.Figure:
        """Create line chart visualization"""
        fig = go.Figure()

        for metric in config.metrics:
            if metric in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data[config.dimensions[0]],
                        y=data[metric],
                        mode="lines+markers",
                        name=metric,
                        line=dict(width=2),
                    )
                )

        return fig

    def _create_heatmap(
        self, config: VisualizationConfig, data: pd.DataFrame
    ) -> go.Figure:
        """Create heatmap visualization"""
        # Pivot data for heatmap
        pivot_data = data.pivot(
            index=config.dimensions[0],
            columns=config.dimensions[1],
            values=config.metrics[0],
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale=config.styling.get("colorscale", "viridis"),
                colorbar=dict(title=config.metrics[0]),
            )
        )

        return fig

    def _create_funnel(
        self, config: VisualizationConfig, data: pd.DataFrame
    ) -> go.Figure:
        """Create funnel chart visualization"""
        fig = go.Figure(
            go.Funnel(
                y=data[config.dimensions[0]],
                x=data[config.metrics[0]],
                textinfo="value+percent initial",
                marker=dict(
                    color=config.styling.get("colors", px.colors.sequential.Viridis)
                ),
            )
        )

        return fig

    def _create_bar_chart(
        self, config: VisualizationConfig, data: pd.DataFrame
    ) -> go.Figure:
        """Create bar chart visualization"""
        fig = go.Figure()

        for metric in config.metrics:
            if metric in data.columns:
                fig.add_trace(
                    go.Bar(x=data[config.dimensions[0]], y=data[metric], name=metric)
                )

        return fig

    def _create_scatter_plot(
        self, config: VisualizationConfig, data: pd.DataFrame
    ) -> go.Figure:
        """Create scatter plot visualization"""
        fig = go.Figure()

        if len(config.metrics) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=data[config.metrics[0]],
                    y=data[config.metrics[1]],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=data.get(config.metrics[2], 1)
                        if len(config.metrics) > 2
                        else 1,
                        colorscale="viridis",
                        showscale=True,
                    ),
                    text=data.get(config.dimensions[0], ""),
                )
            )

        return fig

    def _create_sankey(
        self, config: VisualizationConfig, data: pd.DataFrame
    ) -> go.Figure:
        """Create Sankey diagram"""
        # Prepare node and link data
        all_nodes = list(set(data["source"].tolist() + data["target"].tolist()))
        node_indices = {node: i for i, node in enumerate(all_nodes)}

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=all_nodes,
                        color="blue",
                    ),
                    link=dict(
                        source=[node_indices[s] for s in data["source"]],
                        target=[node_indices[t] for t in data["target"]],
                        value=data["weight"],
                    ),
                )
            ]
        )

        return fig

    def _create_treemap(
        self, config: VisualizationConfig, data: pd.DataFrame
    ) -> go.Figure:
        """Create treemap visualization"""
        fig = go.Figure(
            go.Treemap(
                labels=data[config.dimensions[0]],
                values=data[config.metrics[0]],
                parents=data.get(config.dimensions[1], [""] * len(data)),
                textinfo="label+value+percent root",
            )
        )

        return fig

    def _create_network_graph(
        self, config: VisualizationConfig, data: pd.DataFrame
    ) -> go.Figure:
        """Create network graph visualization"""
        # Create edge traces
        edge_trace = []
        for _, row in data.iterrows():
            edge_trace.append(
                go.Scatter(
                    x=[row["source"], row["target"]],
                    y=[0, 1],  # Simple layout
                    mode="lines",
                    line=dict(width=row["weight"] * 2),
                    showlegend=False,
                )
            )

        # Create node traces
        nodes = list(set(data["source"].tolist() + data["target"].tolist()))
        node_trace = go.Scatter(
            x=list(range(len(nodes))),
            y=[0.5] * len(nodes),
            mode="markers+text",
            text=nodes,
            textposition="top center",
            marker=dict(size=20, color="lightblue"),
        )

        fig = go.Figure(data=edge_trace + [node_trace])
        return fig

    def _create_default_chart(
        self, config: VisualizationConfig, data: pd.DataFrame
    ) -> go.Figure:
        """Create default chart when specific type not implemented"""
        return self._create_bar_chart(config, data)

    async def _create_fallback_visualization(
        self, config: VisualizationConfig, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create fallback visualization when Plotly not available"""
        # Return data in table format
        return {
            "type": "table",
            "title": config.title,
            "columns": list(data.columns),
            "data": data.to_dict("records"),
            "summary": {"rows": len(data), "columns": len(data.columns)},
        }

    def _get_cached_visualization(self, viz_id: str) -> Optional[Dict[str, Any]]:
        """Get cached visualization if available"""
        if viz_id in self.cache:
            cached_data, cached_time = self.cache[viz_id]
            if datetime.now() - cached_time < timedelta(seconds=300):
                return cached_data
        return None

    def _cache_visualization(self, viz_id: str, data: Dict[str, Any], ttl: int):
        """Cache visualization result"""
        self.cache[viz_id] = (data, datetime.now())

    async def create_dashboard(
        self, dashboard_name: str, layout: DashboardLayout, db: AsyncSession
    ) -> Dict[str, Any]:
        """Create a complete dashboard with multiple visualizations"""
        dashboard_data = {
            "name": dashboard_name,
            "layout": layout.grid,
            "visualizations": {},
            "filters": layout.filters,
            "refresh_interval": layout.refresh_interval,
        }

        # Create all visualizations in the layout
        for row in layout.grid:
            for viz_id in row:
                if viz_id and viz_id != "empty":
                    try:
                        viz_data = await self.create_visualization(viz_id, db)
                        dashboard_data["visualizations"][viz_id] = viz_data
                    except Exception as e:
                        logger.error(f"Error creating visualization {viz_id}: {e}")
                        dashboard_data["visualizations"][viz_id] = {
                            "type": "error",
                            "message": str(e),
                        }

        return dashboard_data

    def export_visualization(
        self, viz_data: Dict[str, Any], format: str = "json"
    ) -> Any:
        """Export visualization in various formats"""
        if format == "json":
            return json.dumps(viz_data)
        elif format == "html" and viz_data.get("type") == "plotly":
            # Generate HTML with embedded Plotly
            return self._generate_html_export(viz_data)
        elif format == "csv" and viz_data.get("type") == "table":
            df = pd.DataFrame(viz_data["data"])
            return df.to_csv(index=False)
        else:
            return viz_data

    def _generate_html_export(self, viz_data: Dict[str, Any]) -> str:
        """Generate HTML export for Plotly visualization"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <title>YTEmpire Visualization</title>
        </head>
        <body>
            <div id="visualization"></div>
            <script>
                var figure = {figure_json};
                Plotly.newPlot('visualization', figure.data, figure.layout);
            </script>
        </body>
        </html>
        """

        figure_json = json.dumps(viz_data.get("figure", {}))
        return html_template.replace("{figure_json}", figure_json)

    def get_visualization_list(self) -> List[Dict[str, Any]]:
        """Get list of available visualizations"""
        return [
            {
                "id": viz_id,
                "title": config.title,
                "type": config.type.value,
                "data_source": config.data_source,
                "real_time": config.real_time,
            }
            for viz_id, config in self.visualizations.items()
        ]


# Singleton instance
advanced_visualization_service = AdvancedDataVisualizationService()
