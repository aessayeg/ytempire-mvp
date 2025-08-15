"""
A/B Testing Experiments API Endpoints
Manage experiments and track conversions
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from pydantic import BaseModel, Field
import logging

from app.db.session import get_db
from app.api.v1.endpoints.auth import get_current_verified_user
from app.models.user import User
from app.services.ab_testing_service import (
    ab_testing_service,
    ExperimentVariant,
    ExperimentStatus
)

logger = logging.getLogger(__name__)

router = APIRouter()


class VariantConfig(BaseModel):
    """Variant configuration model"""
    name: str
    allocation: float = Field(..., ge=0, le=100)
    config: Dict[str, Any] = {}
    is_control: bool = False


class CreateExperimentRequest(BaseModel):
    """Request model for creating experiment"""
    name: str
    description: str
    hypothesis: str
    variants: List[VariantConfig]
    target_metric: str
    secondary_metrics: Optional[List[str]] = None
    target_audience: Optional[Dict[str, Any]] = None
    duration_days: Optional[int] = None
    min_sample_size: Optional[int] = 100


class ExperimentResponse(BaseModel):
    """Response model for experiment"""
    experiment_id: int
    name: str
    description: str
    status: str
    variants: List[Dict[str, Any]]
    target_metric: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    winner_variant: Optional[str]


class VariantAssignmentResponse(BaseModel):
    """Response model for variant assignment"""
    variant: Optional[str]
    experiment_active: bool
    config: Dict[str, Any] = {}
    excluded: bool = False


class ConversionTrackingRequest(BaseModel):
    """Request model for tracking conversion"""
    experiment_id: int
    metric_name: str
    metric_value: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


class ExperimentResultsResponse(BaseModel):
    """Response model for experiment results"""
    experiment_id: int
    name: str
    status: str
    variants: List[Dict[str, Any]]
    winner: Optional[str]
    required_sample_size: int
    can_conclude: bool


@router.post("/", response_model=Dict[str, Any])
async def create_experiment(
    request: CreateExperimentRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Create a new A/B test experiment.
    
    Requires admin privileges to create experiments.
    """
    try:
        if not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can create experiments"
            )
        
        # Convert to ExperimentVariant objects
        variants = [
            ExperimentVariant(
                name=v.name,
                allocation=v.allocation,
                config=v.config,
                is_control=v.is_control
            )
            for v in request.variants
        ]
        
        result = await ab_testing_service.create_experiment(
            db=db,
            name=request.name,
            description=request.description,
            hypothesis=request.hypothesis,
            variants=variants,
            target_metric=request.target_metric,
            target_audience=request.target_audience,
            duration_days=request.duration_days,
            min_sample_size=request.min_sample_size
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating experiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create experiment"
        )


@router.post("/{experiment_id}/start", response_model=Dict[str, Any])
async def start_experiment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Start an experiment.
    
    Changes experiment status from draft to running.
    """
    try:
        if not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can start experiments"
            )
        
        result = await ab_testing_service.start_experiment(db, experiment_id)
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error starting experiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start experiment"
        )


@router.get("/assignment/{experiment_id}", response_model=VariantAssignmentResponse)
async def get_variant_assignment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> VariantAssignmentResponse:
    """
    Get or assign variant for current user.
    
    Returns the variant assignment for the user in the specified experiment.
    If no assignment exists, one will be created based on allocation rules.
    """
    try:
        assignment = await ab_testing_service.get_variant_assignment(
            db=db,
            experiment_id=experiment_id,
            user_id=current_user.id
        )
        
        return VariantAssignmentResponse(**assignment)
        
    except Exception as e:
        logger.error(f"Error getting variant assignment: {str(e)}")
        return VariantAssignmentResponse(
            variant=None,
            experiment_active=False,
            config={}
        )


@router.post("/track-conversion", response_model=Dict[str, Any])
async def track_conversion(
    request: ConversionTrackingRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Track a conversion event for an experiment.
    
    Records when a user completes the target action for an experiment.
    """
    try:
        success = await ab_testing_service.track_conversion(
            db=db,
            experiment_id=request.experiment_id,
            user_id=current_user.id,
            metric_name=request.metric_name,
            metric_value=request.metric_value,
            metadata=request.metadata
        )
        
        return {"success": success, "tracked": success}
        
    except Exception as e:
        logger.error(f"Error tracking conversion: {str(e)}")
        return {"success": False, "error": str(e)}


@router.get("/{experiment_id}/results", response_model=ExperimentResultsResponse)
async def get_experiment_results(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> ExperimentResultsResponse:
    """
    Get comprehensive experiment results with statistical analysis.
    
    Returns:
    - Sample sizes and conversion rates for each variant
    - Statistical significance (p-values)
    - Confidence intervals
    - Winner determination
    - Required sample size for conclusive results
    """
    try:
        if not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can view experiment results"
            )
        
        results = await ab_testing_service.get_experiment_results(db, experiment_id)
        return ExperimentResultsResponse(**results)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting experiment results: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get experiment results"
        )


@router.post("/{experiment_id}/conclude", response_model=Dict[str, Any])
async def conclude_experiment(
    experiment_id: int,
    winner_variant: Optional[str] = Body(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Conclude an experiment and optionally declare a winner.
    
    Stops the experiment and finalizes results.
    """
    try:
        if not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can conclude experiments"
            )
        
        result = await ab_testing_service.conclude_experiment(
            db=db,
            experiment_id=experiment_id,
            winner_variant=winner_variant
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error concluding experiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to conclude experiment"
        )


@router.get("/active", response_model=List[Dict[str, Any]])
async def get_active_experiments(
    include_assignments: bool = Query(False, description="Include user's variant assignments"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> List[Dict[str, Any]]:
    """
    Get list of all active experiments.
    
    Returns experiments currently in 'running' status.
    Optionally includes the current user's variant assignments.
    """
    try:
        user_id = current_user.id if include_assignments else None
        experiments = await ab_testing_service.get_active_experiments(db, user_id)
        return experiments
        
    except Exception as e:
        logger.error(f"Error getting active experiments: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get active experiments"
        )


@router.get("/", response_model=List[ExperimentResponse])
async def list_experiments(
    status: Optional[str] = Query(None, description="Filter by status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> List[ExperimentResponse]:
    """
    List all experiments with optional filtering.
    
    Requires admin privileges to view all experiments.
    """
    try:
        if not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can list experiments"
            )
        
        from app.models.experiment import Experiment
        from sqlalchemy import select
        
        query = select(Experiment)
        
        if status:
            query = query.where(Experiment.status == status)
        
        query = query.offset(skip).limit(limit).order_by(Experiment.created_at.desc())
        
        result = await db.execute(query)
        experiments = result.scalars().all()
        
        return [
            ExperimentResponse(
                experiment_id=exp.id,
                name=exp.name,
                description=exp.description,
                status=exp.status,
                variants=exp.variants,
                target_metric=exp.target_metric,
                start_date=exp.start_date,
                end_date=exp.end_date,
                winner_variant=exp.winner_variant
            )
            for exp in experiments
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing experiments: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list experiments"
        )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> ExperimentResponse:
    """
    Get details of a specific experiment.
    
    Requires admin privileges.
    """
    try:
        if not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can view experiment details"
            )
        
        from app.models.experiment import Experiment
        from sqlalchemy import select
        
        query = select(Experiment).where(Experiment.id == experiment_id)
        result = await db.execute(query)
        experiment = result.scalar_one_or_none()
        
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found"
            )
        
        return ExperimentResponse(
            experiment_id=experiment.id,
            name=experiment.name,
            description=experiment.description,
            status=experiment.status,
            variants=experiment.variants,
            target_metric=experiment.target_metric,
            start_date=experiment.start_date,
            end_date=experiment.end_date,
            winner_variant=experiment.winner_variant
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get experiment"
        )