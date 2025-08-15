"""
Content Library API Endpoints
Template management, script library, and voice preset functionality
"""
from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import uuid
import json

from app.db.session import get_db
from app.api.v1.endpoints.auth import get_current_verified_user
from app.models.user import User
from app.services.cost_tracking import cost_tracker

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class TemplateType(str, Enum):
    """Template types"""
    SCRIPT = "script"
    THUMBNAIL = "thumbnail"
    TITLE = "title"
    DESCRIPTION = "description"
    TAGS = "tags"
    WORKFLOW = "workflow"


class TemplateCategory(str, Enum):
    """Template categories"""
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    GAMING = "gaming"
    TECH = "tech"
    LIFESTYLE = "lifestyle"
    BUSINESS = "business"
    NEWS = "news"
    MUSIC = "music"
    SPORTS = "sports"
    COMEDY = "comedy"


class VoiceProvider(str, Enum):
    """Voice synthesis providers"""
    ELEVENLABS = "elevenlabs"
    GOOGLE_TTS = "google_tts"
    AZURE_TTS = "azure_tts"
    AWS_POLLY = "aws_polly"


class TemplateRequest(BaseModel):
    """Request to create a template"""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field("", max_length=500)
    template_type: TemplateType
    category: TemplateCategory
    content: Dict[str, Any] = Field(..., description="Template content/structure")
    tags: List[str] = Field(default=[], description="Template tags")
    is_public: bool = Field(False, description="Make template public")
    variables: List[str] = Field(default=[], description="Template variables")


class ScriptTemplate(BaseModel):
    """Script template structure"""
    hook: str = Field(..., description="Opening hook")
    introduction: str = Field(..., description="Introduction section")
    main_content: List[str] = Field(..., description="Main content sections")
    conclusion: str = Field(..., description="Conclusion")
    call_to_action: str = Field(..., description="Call to action")
    estimated_duration: int = Field(..., description="Estimated duration in seconds")


class VoicePresetRequest(BaseModel):
    """Request to create voice preset"""
    name: str = Field(..., min_length=1, max_length=50)
    provider: VoiceProvider
    voice_id: str = Field(..., description="Provider-specific voice ID")
    settings: Dict[str, Any] = Field({}, description="Voice settings (speed, pitch, etc.)")
    sample_text: str = Field("", description="Sample text for preview")
    is_favorite: bool = Field(False)


class TemplateFilter(BaseModel):
    """Template filtering options"""
    template_type: Optional[TemplateType] = None
    category: Optional[TemplateCategory] = None
    tags: List[str] = Field(default=[])
    is_public: Optional[bool] = None
    search_query: str = Field("", description="Search in name/description")


@router.post("/templates")
async def create_template(
    request: TemplateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Create a new content template
    """
    try:
        template_id = f"tmpl_{uuid.uuid4().hex[:8]}"
        
        # Validate template content based on type
        if request.template_type == TemplateType.SCRIPT:
            _validate_script_template(request.content)
        
        # Track cost
        background_tasks.add_task(
            cost_tracker.track_cost,
            user_id=str(current_user.id),
            service="content_library",
            operation="create_template",
            cost=0.001,
            metadata={"template_type": request.template_type.value}
        )
        
        # Store template (in production, this would be in database)
        template_data = {
            "id": template_id,
            "name": request.name,
            "description": request.description,
            "type": request.template_type.value,
            "category": request.category.value,
            "content": request.content,
            "tags": request.tags,
            "variables": request.variables,
            "is_public": request.is_public,
            "user_id": str(current_user.id),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "usage_count": 0
        }
        
        # In production: await db_store_template(template_data)
        
        return {
            "success": True,
            "template_id": template_id,
            "message": "Template created successfully",
            "template": template_data
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to create template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create template: {str(e)}"
        )


@router.get("/templates")
async def list_templates(
    template_type: Optional[TemplateType] = Query(None),
    category: Optional[TemplateCategory] = Query(None),
    tags: List[str] = Query(default=[]),
    is_public: Optional[bool] = Query(None),
    search: str = Query("", description="Search in name/description"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    List templates with filtering and search
    """
    try:
        # In production, this would query the database
        # For now, return mock data
        templates = _get_mock_templates(
            user_id=str(current_user.id),
            template_type=template_type,
            category=category,
            tags=tags,
            is_public=is_public,
            search=search,
            limit=limit,
            offset=offset
        )
        
        return {
            "success": True,
            "templates": templates,
            "total": len(templates),
            "filters_applied": {
                "type": template_type.value if template_type else None,
                "category": category.value if category else None,
                "tags": tags,
                "is_public": is_public,
                "search": search
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list templates: {str(e)}"
        )


@router.get("/templates/{template_id}")
async def get_template(
    template_id: str,
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Get a specific template by ID
    """
    try:
        # In production: template = await db_get_template(template_id)
        template = _get_mock_template(template_id, str(current_user.id))
        
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        # Check access permissions
        if not template["is_public"] and template["user_id"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to private template"
            )
        
        return {
            "success": True,
            "template": template
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get template: {str(e)}"
        )


@router.put("/templates/{template_id}")
async def update_template(
    template_id: str,
    request: TemplateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Update an existing template
    """
    try:
        # Check template ownership
        template = _get_mock_template(template_id, str(current_user.id))
        
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        if template["user_id"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Can only update your own templates"
            )
        
        # Validate content
        if request.template_type == TemplateType.SCRIPT:
            _validate_script_template(request.content)
        
        # Track cost
        background_tasks.add_task(
            cost_tracker.track_cost,
            user_id=str(current_user.id),
            service="content_library",
            operation="update_template",
            cost=0.0005,
            metadata={"template_id": template_id}
        )
        
        # Update template
        updated_template = {
            **template,
            "name": request.name,
            "description": request.description,
            "type": request.template_type.value,
            "category": request.category.value,
            "content": request.content,
            "tags": request.tags,
            "variables": request.variables,
            "is_public": request.is_public,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        return {
            "success": True,
            "message": "Template updated successfully",
            "template": updated_template
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update template: {str(e)}"
        )


@router.delete("/templates/{template_id}")
async def delete_template(
    template_id: str,
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, str]:
    """
    Delete a template
    """
    try:
        template = _get_mock_template(template_id, str(current_user.id))
        
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        if template["user_id"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Can only delete your own templates"
            )
        
        # In production: await db_delete_template(template_id)
        
        return {
            "success": True,
            "message": "Template deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete template: {str(e)}"
        )


@router.post("/templates/{template_id}/use")
async def use_template(
    template_id: str,
    variables: Dict[str, str] = {},
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Use a template with variable substitution
    """
    try:
        template = _get_mock_template(template_id, str(current_user.id))
        
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        # Check access permissions
        if not template["is_public"] and template["user_id"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to private template"
            )
        
        # Track usage
        background_tasks.add_task(
            cost_tracker.track_cost,
            user_id=str(current_user.id),
            service="content_library",
            operation="use_template",
            cost=0.0001,
            metadata={"template_id": template_id}
        )
        
        # Apply variable substitution
        processed_content = _apply_template_variables(template["content"], variables)
        
        return {
            "success": True,
            "template_id": template_id,
            "processed_content": processed_content,
            "variables_applied": variables,
            "usage_count": template.get("usage_count", 0) + 1
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to use template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to use template: {str(e)}"
        )


@router.post("/voice-presets")
async def create_voice_preset(
    request: VoicePresetRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Create a new voice preset
    """
    try:
        preset_id = f"voice_{uuid.uuid4().hex[:8]}"
        
        # Track cost
        background_tasks.add_task(
            cost_tracker.track_cost,
            user_id=str(current_user.id),
            service="content_library",
            operation="create_voice_preset",
            cost=0.001,
            metadata={"provider": request.provider.value}
        )
        
        # Store voice preset
        preset_data = {
            "id": preset_id,
            "name": request.name,
            "provider": request.provider.value,
            "voice_id": request.voice_id,
            "settings": request.settings,
            "sample_text": request.sample_text,
            "is_favorite": request.is_favorite,
            "user_id": str(current_user.id),
            "created_at": datetime.utcnow().isoformat(),
            "usage_count": 0
        }
        
        return {
            "success": True,
            "preset_id": preset_id,
            "message": "Voice preset created successfully",
            "preset": preset_data
        }
        
    except Exception as e:
        logger.error(f"Failed to create voice preset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create voice preset: {str(e)}"
        )


@router.get("/voice-presets")
async def list_voice_presets(
    provider: Optional[VoiceProvider] = Query(None),
    favorites_only: bool = Query(False),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    List voice presets
    """
    try:
        # Mock voice presets data
        presets = _get_mock_voice_presets(str(current_user.id), provider, favorites_only)
        
        return {
            "success": True,
            "voice_presets": presets,
            "total": len(presets),
            "filters": {
                "provider": provider.value if provider else None,
                "favorites_only": favorites_only
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to list voice presets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list voice presets: {str(e)}"
        )


@router.get("/voice-presets/providers")
async def get_voice_providers(
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Get available voice providers and their capabilities
    """
    try:
        providers = {
            "elevenlabs": {
                "name": "ElevenLabs",
                "description": "High-quality AI voice synthesis",
                "cost_per_1k_chars": 0.30,
                "quality_rating": 95,
                "languages": ["en", "es", "fr", "de", "it", "pt", "pl"],
                "features": ["emotion_control", "voice_cloning", "speech_synthesis"],
                "sample_voices": [
                    {"id": "adam", "name": "Adam", "gender": "male", "accent": "american"},
                    {"id": "bella", "name": "Bella", "gender": "female", "accent": "american"},
                    {"id": "charlie", "name": "Charlie", "gender": "male", "accent": "british"}
                ]
            },
            "google_tts": {
                "name": "Google Text-to-Speech",
                "description": "Reliable cloud-based TTS",
                "cost_per_1k_chars": 0.016,
                "quality_rating": 85,
                "languages": ["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh"],
                "features": ["neural_voices", "ssml_support", "audio_profiles"],
                "sample_voices": [
                    {"id": "en-US-Neural2-J", "name": "Neural2-J", "gender": "male", "accent": "american"},
                    {"id": "en-US-Neural2-F", "name": "Neural2-F", "gender": "female", "accent": "american"}
                ]
            },
            "azure_tts": {
                "name": "Azure Cognitive Services TTS",
                "description": "Microsoft's neural voice synthesis",
                "cost_per_1k_chars": 0.015,
                "quality_rating": 88,
                "languages": ["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "ar"],
                "features": ["neural_voices", "custom_voices", "voice_tuning"],
                "sample_voices": [
                    {"id": "en-US-JennyNeural", "name": "Jenny", "gender": "female", "accent": "american"},
                    {"id": "en-US-GuyNeural", "name": "Guy", "gender": "male", "accent": "american"}
                ]
            },
            "aws_polly": {
                "name": "Amazon Polly",
                "description": "AWS text-to-speech service",
                "cost_per_1k_chars": 0.004,
                "quality_rating": 80,
                "languages": ["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "ar", "hi"],
                "features": ["neural_voices", "lexicons", "speech_marks"],
                "sample_voices": [
                    {"id": "Joanna", "name": "Joanna", "gender": "female", "accent": "american"},
                    {"id": "Matthew", "name": "Matthew", "gender": "male", "accent": "american"}
                ]
            }
        }
        
        return {
            "success": True,
            "providers": providers,
            "recommendations": {
                "highest_quality": "elevenlabs",
                "best_value": "aws_polly",
                "most_languages": "azure_tts",
                "most_reliable": "google_tts"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get voice providers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get voice providers: {str(e)}"
        )


@router.post("/voice-presets/{preset_id}/test")
async def test_voice_preset(
    preset_id: str,
    test_text: str = "This is a test of the voice preset. How does it sound?",
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Test a voice preset with sample text
    """
    try:
        preset = _get_mock_voice_preset(preset_id, str(current_user.id))
        
        if not preset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice preset not found"
            )
        
        # Track cost
        background_tasks.add_task(
            cost_tracker.track_cost,
            user_id=str(current_user.id),
            service="voice_synthesis",
            operation="test_voice",
            cost=0.01,  # Estimated cost for voice test
            metadata={"preset_id": preset_id, "provider": preset["provider"]}
        )
        
        # In production: generate actual audio using the preset
        audio_url = f"https://storage.ytempire.com/voice-tests/{preset_id}/{uuid.uuid4().hex[:8]}.mp3"
        
        return {
            "success": True,
            "preset_id": preset_id,
            "test_text": test_text,
            "audio_url": audio_url,
            "preset_details": preset,
            "estimated_cost": 0.01,
            "duration_seconds": len(test_text) * 0.1  # Rough estimate
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to test voice preset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test voice preset: {str(e)}"
        )


@router.get("/script-library/suggestions")
async def get_script_suggestions(
    topic: str = Query(..., description="Video topic"),
    target_duration: int = Query(300, description="Target duration in seconds"),
    style: str = Query("educational", description="Content style"),
    audience: str = Query("general", description="Target audience"),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Get script suggestions based on topic and requirements
    """
    try:
        # Generate script suggestions (in production, this would use AI)
        suggestions = _generate_script_suggestions(topic, target_duration, style, audience)
        
        return {
            "success": True,
            "topic": topic,
            "target_duration": target_duration,
            "style": style,
            "audience": audience,
            "suggestions": suggestions,
            "templates_available": len([t for t in _get_mock_templates(str(current_user.id)) if t["type"] == "script"])
        }
        
    except Exception as e:
        logger.error(f"Failed to get script suggestions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get script suggestions: {str(e)}"
        )


# Helper functions

def _validate_script_template(content: Dict[str, Any]) -> None:
    """Validate script template structure"""
    required_fields = ["hook", "introduction", "main_content", "conclusion", "call_to_action"]
    
    for field in required_fields:
        if field not in content:
            raise ValueError(f"Script template missing required field: {field}")
    
    if not isinstance(content["main_content"], list):
        raise ValueError("main_content must be a list of sections")


def _get_mock_templates(
    user_id: str,
    template_type: Optional[TemplateType] = None,
    category: Optional[TemplateCategory] = None,
    tags: List[str] = [],
    is_public: Optional[bool] = None,
    search: str = "",
    limit: int = 20,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Get mock templates data"""
    # Mock data - in production this would query the database
    templates = [
        {
            "id": "tmpl_12345",
            "name": "Educational Video Script",
            "description": "Template for educational content",
            "type": "script",
            "category": "educational",
            "content": {
                "hook": "Did you know that {{fact}}?",
                "introduction": "Today we're exploring {{topic}}",
                "main_content": ["{{main_point_1}}", "{{main_point_2}}", "{{main_point_3}}"],
                "conclusion": "So remember, {{key_takeaway}}",
                "call_to_action": "Subscribe for more {{category}} content!"
            },
            "tags": ["education", "tutorial"],
            "variables": ["fact", "topic", "main_point_1", "main_point_2", "main_point_3", "key_takeaway", "category"],
            "is_public": True,
            "user_id": user_id,
            "created_at": "2024-01-10T10:00:00Z",
            "usage_count": 15
        }
    ]
    
    # Apply filters (simplified for mock)
    if template_type:
        templates = [t for t in templates if t["type"] == template_type.value]
    
    return templates[offset:offset + limit]


def _get_mock_template(template_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get a single mock template"""
    templates = _get_mock_templates(user_id)
    return next((t for t in templates if t["id"] == template_id), None)


def _get_mock_voice_presets(
    user_id: str,
    provider: Optional[VoiceProvider] = None,
    favorites_only: bool = False
) -> List[Dict[str, Any]]:
    """Get mock voice presets"""
    presets = [
        {
            "id": "voice_12345",
            "name": "Professional Male",
            "provider": "elevenlabs",
            "voice_id": "adam",
            "settings": {"speed": 1.0, "pitch": 0.0, "stability": 0.8},
            "sample_text": "This is a professional voice for business content",
            "is_favorite": True,
            "user_id": user_id,
            "created_at": "2024-01-10T10:00:00Z",
            "usage_count": 25
        }
    ]
    
    if provider:
        presets = [p for p in presets if p["provider"] == provider.value]
    
    if favorites_only:
        presets = [p for p in presets if p["is_favorite"]]
    
    return presets


def _get_mock_voice_preset(preset_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get a single mock voice preset"""
    presets = _get_mock_voice_presets(user_id)
    return next((p for p in presets if p["id"] == preset_id), None)


def _apply_template_variables(content: Dict[str, Any], variables: Dict[str, str]) -> Dict[str, Any]:
    """Apply variable substitution to template content"""
    def substitute_text(text: str) -> str:
        for var, value in variables.items():
            text = text.replace(f"{{{{{var}}}}}", value)
        return text
    
    processed = {}
    for key, value in content.items():
        if isinstance(value, str):
            processed[key] = substitute_text(value)
        elif isinstance(value, list):
            processed[key] = [substitute_text(item) if isinstance(item, str) else item for item in value]
        else:
            processed[key] = value
    
    return processed


def _generate_script_suggestions(topic: str, target_duration: int, style: str, audience: str) -> List[Dict[str, Any]]:
    """Generate script suggestions"""
    return [
        {
            "title": f"Introduction to {topic}",
            "structure": {
                "hook": f"Surprising fact about {topic}",
                "duration_breakdown": {
                    "hook": 15,
                    "introduction": 30,
                    "main_content": target_duration - 90,
                    "conclusion": 30,
                    "call_to_action": 15
                }
            },
            "key_points": [
                f"What is {topic}?",
                f"Why {topic} matters",
                f"How to get started with {topic}"
            ],
            "estimated_engagement": 0.85,
            "difficulty": "beginner"
        }
    ]