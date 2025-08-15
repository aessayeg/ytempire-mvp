"""
Data Analytics API Endpoints
Integrates all Data Team P2 components into the main application
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
import numpy as np

from app.db.session import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.services.advanced_data_visualization import (
    advanced_visualization_service,
    VisualizationType,
    VisualizationConfig
)
from app.services.custom_report_builder import (
    custom_report_builder,
    ReportFormat,
    ReportFrequency
)
from app.services.data_marketplace_integration import (
    data_marketplace,
    DataCategory,
    MarketplaceProvider
)
from app.services.advanced_forecasting_models import (
    advanced_forecasting,
    ForecastModel,
    ForecastMetric,
    ForecastConfig
)

router = APIRouter()


# ============== VISUALIZATION ENDPOINTS ==============

@router.get("/visualizations")
async def get_visualizations(
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get list of available visualizations"""
    return advanced_visualization_service.get_visualization_list()


@router.post("/visualizations/create")
async def create_visualization(
    viz_id: str,
    time_range: Optional[Dict[str, str]] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Create a visualization"""
    try:
        # Parse time range if provided
        if time_range:
            start = datetime.fromisoformat(time_range.get('start', ''))
            end = datetime.fromisoformat(time_range.get('end', ''))
            time_tuple = (start, end)
        else:
            time_tuple = None
        
        result = await advanced_visualization_service.create_visualization(
            viz_id, db, time_tuple
        )
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/visualizations/register")
async def register_custom_visualization(
    config: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Register a custom visualization"""
    try:
        viz_config = VisualizationConfig(
            type=VisualizationType[config['type']],
            title=config['title'],
            data_source=config['data_source'],
            dimensions=config.get('dimensions', []),
            metrics=config.get('metrics', []),
            interactive=config.get('interactive', True),
            real_time=config.get('real_time', False)
        )
        viz_id = advanced_visualization_service.register_visualization(viz_config)
        return {"status": "success", "viz_id": viz_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/dashboards/create")
async def create_dashboard(
    name: str,
    layout: List[List[str]],
    refresh_interval: int = 60,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Create a dashboard with multiple visualizations"""
    try:
        from app.services.advanced_data_visualization import DashboardLayout
        
        dashboard_layout = DashboardLayout(
            name=name,
            grid=layout,
            refresh_interval=refresh_interval
        )
        
        result = await advanced_visualization_service.create_dashboard(
            name, dashboard_layout, db
        )
        return {"status": "success", "dashboard": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============== REPORT BUILDER ENDPOINTS ==============

@router.get("/reports/templates")
async def get_report_templates(
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get available report templates"""
    return custom_report_builder.get_templates()


@router.post("/reports/generate")
async def generate_report(
    template_id: str,
    format: Optional[str] = "pdf",
    filters: Optional[Dict[str, Any]] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Generate a report from template"""
    try:
        format_enum = ReportFormat[format.upper()]
        result = await custom_report_builder.create_report(
            template_id, db, filters, format_enum
        )
        return {
            "status": "success",
            "report": result['metadata'],
            "report_id": result['report_id']
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/reports/schedule")
async def schedule_report(
    template_id: str,
    frequency: str,
    recipients: List[str],
    time_of_day: Optional[str] = "09:00",
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Schedule automatic report generation"""
    try:
        frequency_enum = ReportFrequency[frequency.upper()]
        schedule_id = custom_report_builder.schedule_report(
            template_id, frequency_enum, recipients, time_of_day
        )
        return {"status": "success", "schedule_id": schedule_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/reports/scheduled")
async def get_scheduled_reports(
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get list of scheduled reports"""
    return custom_report_builder.get_scheduled_reports()


@router.post("/reports/custom-template")
async def create_custom_template(
    name: str,
    sections: List[Dict[str, Any]],
    format: str = "pdf",
    frequency: str = "on_demand",
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Create a custom report template"""
    try:
        format_enum = ReportFormat[format.upper()]
        frequency_enum = ReportFrequency[frequency.upper()]
        
        template_id = custom_report_builder.create_custom_template(
            name, sections, format_enum, frequency_enum
        )
        return {"status": "success", "template_id": template_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============== DATA MARKETPLACE ENDPOINTS ==============

@router.get("/marketplace/products")
async def browse_marketplace_products(
    category: Optional[str] = None,
    provider: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Browse available data products in marketplace"""
    try:
        category_enum = DataCategory[category.upper()] if category else None
        provider_enum = MarketplaceProvider[provider.upper()] if provider else None
        
        price_range = None
        if min_price is not None and max_price is not None:
            from decimal import Decimal
            price_range = (Decimal(str(min_price)), Decimal(str(max_price)))
        
        products = await data_marketplace.browse_products(
            category_enum, provider_enum, price_range
        )
        
        return [
            {
                "id": p.id,
                "name": p.name,
                "provider": p.provider.value,
                "category": p.category.value,
                "description": p.description,
                "price": float(p.price),
                "price_model": p.price_model,
                "format": p.format.value,
                "rating": p.rating,
                "reviews_count": p.reviews_count
            }
            for p in products
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/marketplace/subscribe")
async def subscribe_to_data_product(
    product_id: str,
    duration_days: Optional[int] = 30,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Subscribe to a data product"""
    try:
        subscription_id = await data_marketplace.subscribe_to_product(
            product_id, duration_days
        )
        return {"status": "success", "subscription_id": subscription_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/marketplace/fetch")
async def fetch_marketplace_data(
    subscription_id: str,
    filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = 100,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Fetch data from a subscribed product"""
    try:
        data = await data_marketplace.fetch_data(
            subscription_id, filters, limit
        )
        return {
            "status": "success",
            "records_count": len(data.get('records', [])),
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/marketplace/subscriptions/{subscription_id}/status")
async def get_subscription_status(
    subscription_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get subscription status and usage"""
    try:
        return data_marketplace.get_subscription_status(subscription_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/marketplace/analytics")
async def get_marketplace_analytics(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get marketplace usage analytics"""
    return data_marketplace.get_marketplace_analytics()


# ============== FORECASTING ENDPOINTS ==============

@router.post("/forecast/create")
async def create_forecast(
    model_type: str,
    metric: str,
    horizon_days: int,
    channel_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Create a forecast using specified model"""
    try:
        # Generate sample historical data (in production, fetch from DB)
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        values = np.cumsum(np.random.randn(90)) + 1000
        
        if metric == "revenue":
            values = values * 10  # Scale for revenue
        elif metric == "views":
            values = values * 100  # Scale for views
        
        historical_data = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        # Create forecast config
        config = ForecastConfig(
            model_type=ForecastModel[model_type.upper()],
            metric=ForecastMetric[metric.upper()],
            horizon=horizon_days
        )
        
        # Generate forecast
        result = await advanced_forecasting.create_forecast(
            config, historical_data, db
        )
        
        # Convert to JSON-serializable format
        predictions = result.predictions['forecast'].tolist()
        dates = [d.isoformat() for d in result.predictions.index]
        
        response = {
            "status": "success",
            "model": result.model_type.value,
            "metric": result.metric.value,
            "predictions": {
                "dates": dates,
                "values": predictions
            },
            "accuracy_metrics": result.accuracy_metrics,
            "model_params": result.model_params
        }
        
        # Add confidence intervals if available
        if result.confidence_intervals is not None:
            response["confidence_intervals"] = {
                "lower": result.confidence_intervals['lower'].tolist(),
                "upper": result.confidence_intervals['upper'].tolist()
            }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/forecast/compare")
async def compare_forecast_models(
    metric: str,
    horizon_days: int = 30,
    models: Optional[List[str]] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Compare multiple forecasting models"""
    try:
        # Generate sample historical data
        dates = pd.date_range(end=datetime.now(), periods=120, freq='D')
        values = np.cumsum(np.random.randn(120)) + 1000
        historical_data = pd.DataFrame({'date': dates, 'value': values})
        
        # Parse models
        if models:
            model_list = [ForecastModel[m.upper()] for m in models]
        else:
            model_list = None
        
        # Compare models
        comparison = await advanced_forecasting.compare_models(
            historical_data,
            ForecastMetric[metric.upper()],
            horizon_days,
            model_list
        )
        
        return {
            "status": "success",
            "comparison": comparison
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/forecast/recommendations")
async def get_model_recommendations(
    data_length: int,
    has_seasonality: bool = False,
    has_trend: bool = False,
    is_stationary: bool = False,
    has_outliers: bool = False,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get model recommendations based on data characteristics"""
    try:
        characteristics = {
            'length': data_length,
            'seasonality': has_seasonality,
            'trend': has_trend,
            'stationary': is_stationary,
            'outliers': has_outliers
        }
        
        recommendations = advanced_forecasting.get_model_recommendations(
            characteristics
        )
        
        return {
            "status": "success",
            "recommended_models": [m.value for m in recommendations],
            "characteristics": characteristics
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============== INTEGRATED ANALYTICS ENDPOINT ==============

@router.get("/analytics/overview")
async def get_analytics_overview(
    channel_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get comprehensive analytics overview integrating all services"""
    try:
        # This endpoint demonstrates integration of all services
        overview = {
            "timestamp": datetime.now().isoformat(),
            "services_status": {
                "visualization": "active",
                "reporting": "active",
                "marketplace": "active",
                "forecasting": "active"
            },
            "available_features": {
                "visualizations": len(advanced_visualization_service.get_visualization_list()),
                "report_templates": len(custom_report_builder.get_templates()),
                "marketplace_products": len(data_marketplace.products),
                "forecast_models": len(ForecastModel)
            },
            "usage_stats": {
                "marketplace": data_marketplace.get_marketplace_analytics(),
                "scheduled_reports": len(custom_report_builder.get_scheduled_reports())
            }
        }
        
        # Add sample forecast
        if channel_id:
            # Generate sample forecast for the channel
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            values = np.random.uniform(100, 1000, 30)
            historical_data = pd.DataFrame({'date': dates, 'value': values})
            
            config = ForecastConfig(
                model_type=ForecastModel.LINEAR_REGRESSION,
                metric=ForecastMetric.REVENUE,
                horizon=7
            )
            
            forecast_result = await advanced_forecasting.create_forecast(
                config, historical_data, db
            )
            
            overview["sample_forecast"] = {
                "next_7_days": forecast_result.predictions['forecast'].tolist()
            }
        
        return overview
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))