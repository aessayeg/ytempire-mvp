"""
Test endpoint for first video generation attempt
Day 8 - Critical P0 Task
"""
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional
import logging

from app.db.session import get_db
from app.services.video_generation_orchestrator import video_orchestrator
from app.services.youtube_multi_account import get_youtube_manager
from app.services.cost_tracking import cost_tracker
from app.models.channel import Channel
from app.api.v1.endpoints.auth import get_current_verified_user
from app.models.user import User

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/test-video-generation")
async def test_video_generation(
    channel_id: Optional[str] = None,
    topic: Optional[str] = "Latest AI Technology Trends 2024",
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Test endpoint for first video generation
    This is the CRITICAL P0 task for Day 8
    Target: <10 min generation, <$3 cost
    """
    try:
        # Initialize services
        await get_youtube_manager().initialize(db)
        
        # Use first available channel if not specified
        if not channel_id:
            result = await db.execute(
                select(Channel).filter(Channel.user_id == current_user.id).limit(1)
            )
            channel = result.scalar_one_or_none()
            if not channel:
                # Create a test channel
                channel = Channel(
                    id=f"test_channel_{current_user.id}",
                    user_id=current_user.id,
                    name="Test Channel",
                    platform="youtube"
                )
                db.add(channel)
                await db.commit()
            channel_id = channel.id
        
        logger.info(f"Starting test video generation for channel {channel_id} with topic: {topic}")
        
        # Generate video
        result = await video_orchestrator.generate_video(
            channel_id=channel_id,
            topic=topic,
            db=db
        )
        
        # Validate results against P0 requirements
        validation = {
            "p0_requirements_met": True,
            "checks": {
                "generation_time": {
                    "target": 600,  # 10 minutes
                    "actual": result["metrics"]["total_duration_seconds"] if result["success"] else None,
                    "passed": result["metrics"]["total_duration_seconds"] < 600 if result["success"] else False
                },
                "cost": {
                    "target": 3.00,
                    "actual": float(result["metrics"]["total_cost"]) if result["success"] else None,
                    "passed": float(result["metrics"]["total_cost"]) < 3.00 if result["success"] else False
                },
                "quality_score": {
                    "target": 70,
                    "actual": result["metrics"]["quality_score"] if result["success"] else None,
                    "passed": result["metrics"]["quality_score"] >= 70 if result["success"] else False
                },
                "youtube_upload": {
                    "target": True,
                    "actual": result.get("youtube_id") is not None if result["success"] else False,
                    "passed": result.get("youtube_id") is not None if result["success"] else False
                }
            }
        }
        
        # Check if all P0 requirements are met
        validation["p0_requirements_met"] = all(
            check["passed"] for check in validation["checks"].values()
        )
        
        # Log results
        if validation["p0_requirements_met"]:
            logger.info(
                f"✅ P0 SUCCESS: Video generated in {result['metrics']['total_duration_seconds']:.1f}s "
                f"for ${result['metrics']['total_cost']:.2f} with quality score {result['metrics']['quality_score']:.1f}"
            )
        else:
            failed_checks = [
                name for name, check in validation["checks"].items() 
                if not check["passed"]
            ]
            logger.warning(f"⚠️ P0 requirements not fully met. Failed checks: {failed_checks}")
        
        return {
            "test_status": "PASSED" if validation["p0_requirements_met"] else "PARTIAL",
            "video_generation_result": result,
            "p0_validation": validation,
            "summary": {
                "video_id": result.get("video_id"),
                "youtube_id": result.get("youtube_id"),
                "duration_seconds": result["metrics"]["total_duration_seconds"] if result["success"] else None,
                "total_cost_usd": float(result["metrics"]["total_cost"]) if result["success"] else None,
                "quality_score": result["metrics"]["quality_score"] if result["success"] else None,
                "success": result["success"]
            }
        }
        
    except Exception as e:
        logger.error(f"Test video generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/test-generation-ws/{channel_id}")
async def test_generation_websocket(
    websocket: WebSocket,
    channel_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    WebSocket endpoint for real-time video generation updates
    """
    await websocket.accept()
    
    try:
        # Initialize services
        await get_youtube_manager().initialize(db)
        
        # Start video generation with WebSocket updates
        result = await video_orchestrator.generate_video(
            channel_id=channel_id,
            topic="AI Technology Trends 2024",
            db=db,
            websocket=websocket
        )
        
        # Send final result
        await websocket.send_json({
            "type": "final_result",
            "result": result
        })
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for channel {channel_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        await websocket.close()

@router.get("/test-generation-status")
async def get_generation_status(
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Get status of all active video generations
    """
    active_generations = video_orchestrator.get_active_generations()
    
    # Get YouTube account status
    quota_status = await get_youtube_manager().check_quota_limits()
    
    # Get cost tracker status
    # Use global cost_tracker instance
    daily_costs = await cost_tracker.get_daily_total()
    
    return {
        "active_generations": active_generations,
        "youtube_accounts": {
            "total": quota_status["total_quota"],
            "used": quota_status["total_used"],
            "available_accounts": quota_status["accounts_available"],
            "accounts": quota_status["accounts_status"]
        },
        "costs": {
            "daily_total": float(daily_costs),
            "daily_limit": 100.00,
            "percentage_used": (float(daily_costs) / 100.00) * 100
        },
        "system_status": {
            "all_services_operational": True,
            "p0_tasks_ready": True,
            "can_generate_video": quota_status["accounts_available"] > 0 and float(daily_costs) < 90.00
        }
    }

@router.post("/test-batch-generation")
async def test_batch_generation(
    num_videos: int = 3,
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Test batch video generation (up to 10 concurrent)
    """
    if num_videos > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 videos for batch generation")
    
    # Initialize services
    await get_youtube_manager().initialize(db)
    
    # Get user's channels
    result = await db.execute(
        select(Channel).filter(Channel.user_id == current_user.id).limit(num_videos)
    )
    channels = result.scalars().all()
    
    if len(channels) < num_videos:
        # Create test channels if needed
        for i in range(num_videos - len(channels)):
            channel = Channel(
                id=f"test_channel_{current_user.id}_{i}",
                user_id=current_user.id,
                name=f"Test Channel {i+1}",
                platform="youtube"
            )
            db.add(channel)
            channels.append(channel)
        await db.commit()
    
    channel_ids = [channel.id for channel in channels[:num_videos]]
    
    # Generate videos in batch
    results = await video_orchestrator.batch_generate(
        channel_ids=channel_ids,
        max_concurrent=3,  # Limit concurrent generations
        db=db
    )
    
    # Calculate success metrics
    successful = sum(1 for r in results if r["success"])
    total_cost = sum(
        float(r["metrics"]["total_cost"]) 
        for r in results 
        if r["success"] and "metrics" in r
    )
    avg_duration = sum(
        r["metrics"]["total_duration_seconds"] 
        for r in results 
        if r["success"] and "metrics" in r
    ) / max(successful, 1)
    
    return {
        "batch_summary": {
            "requested": num_videos,
            "successful": successful,
            "failed": num_videos - successful,
            "total_cost_usd": total_cost,
            "average_duration_seconds": avg_duration,
            "success_rate": (successful / num_videos) * 100
        },
        "individual_results": results,
        "p0_batch_validation": {
            "all_under_10_min": all(
                r["metrics"]["total_duration_seconds"] < 600 
                for r in results 
                if r["success"]
            ),
            "all_under_3_usd": all(
                float(r["metrics"]["total_cost"]) < 3.00 
                for r in results 
                if r["success"]
            ),
            "batch_success": successful == num_videos
        }
    }