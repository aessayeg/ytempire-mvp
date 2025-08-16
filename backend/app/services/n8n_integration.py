"""
N8N Workflow Integration Service
Manages webhook communication with N8N workflows
"""
import os
import json
import logging
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """N8N Workflow types"""

    VIDEO_AUTOMATION = "video_automation"
    TREND_ANALYSIS = "trend_analysis"
    QUALITY_CHECK = "quality_check"
    CHANNEL_OPTIMIZATION = "channel_optimization"
    CONTENT_SCHEDULING = "content_scheduling"


class N8NIntegration:
    """N8N Workflow Engine Integration"""

    def __init__(self):
        self.base_url = os.getenv("N8N_URL", "http://localhost:5678")
        self.api_key = os.getenv("N8N_API_KEY", "")
        self.webhook_base = f"{self.base_url}/webhook"
        self.workflows = self._load_workflows()

    def _load_workflows(self) -> Dict[str, str]:
        """Load workflow webhook IDs"""
        return {
            WorkflowType.VIDEO_AUTOMATION: "ytempire-video-trigger",
            WorkflowType.TREND_ANALYSIS: "ytempire-trend-trigger",
            WorkflowType.QUALITY_CHECK: "ytempire-quality-trigger",
            WorkflowType.CHANNEL_OPTIMIZATION: "ytempire-channel-trigger",
            WorkflowType.CONTENT_SCHEDULING: "ytempire-schedule-trigger",
        }

    async def trigger_workflow(
        self,
        workflow_type: WorkflowType,
        data: Dict[str, Any],
        wait_for_response: bool = True,
        timeout: int = 300,
    ) -> Optional[Dict[str, Any]]:
        """Trigger an N8N workflow via webhook"""

        webhook_id = self.workflows.get(workflow_type)
        if not webhook_id:
            logger.error(f"Unknown workflow type: {workflow_type}")
            return None

        webhook_url = f"{self.webhook_base}/{webhook_id}"

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                headers = {"Content-Type": "application/json"}

                if self.api_key:
                    headers["X-N8N-API-KEY"] = self.api_key

                logger.info(f"Triggering N8N workflow: {workflow_type.value}")

                response = await client.post(webhook_url, json=data, headers=headers)

                response.raise_for_status()

                if wait_for_response:
                    result = response.json()
                    logger.info(f"Workflow completed: {workflow_type.value}")
                    return result
                else:
                    logger.info(f"Workflow triggered async: {workflow_type.value}")
                    return {"status": "triggered", "workflow": workflow_type.value}

        except httpx.TimeoutException:
            logger.error(f"Workflow timeout: {workflow_type.value}")
            return {"status": "error", "message": "Workflow execution timeout"}

        except httpx.HTTPError as e:
            logger.error(f"Workflow HTTP error: {e}")
            return {"status": "error", "message": str(e)}

        except Exception as e:
            logger.error(f"Workflow trigger failed: {e}")
            return None

    async def trigger_video_automation(
        self,
        channel_id: int,
        topic: str = None,
        style: str = "educational",
        schedule_time: datetime = None,
    ) -> Optional[Dict[str, Any]]:
        """Trigger video automation workflow"""

        data = {
            "channel_id": channel_id,
            "style": style,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if topic:
            data["topic"] = topic
        else:
            data["auto_select_topic"] = True

        if schedule_time:
            data["schedule_time"] = schedule_time.isoformat()
            data["scheduled"] = True
        else:
            data["immediate"] = True

        return await self.trigger_workflow(
            WorkflowType.VIDEO_AUTOMATION,
            data,
            wait_for_response=not schedule_time,  # Don't wait if scheduled
        )

    async def trigger_trend_analysis(
        self, categories: List[str] = None, region: str = "US", max_results: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Trigger trend analysis workflow"""

        data = {
            "categories": categories or ["all"],
            "region": region,
            "max_results": max_results,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return await self.trigger_workflow(
            WorkflowType.TREND_ANALYSIS, data, wait_for_response=True
        )

    async def trigger_quality_check(
        self, video_id: int, video_path: str, strict_mode: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Trigger quality check workflow"""

        data = {
            "video_id": video_id,
            "video_path": video_path,
            "strict_mode": strict_mode,
            "checks": [
                "content_quality",
                "audio_quality",
                "video_quality",
                "policy_compliance",
                "engagement_prediction",
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        return await self.trigger_workflow(
            WorkflowType.QUALITY_CHECK, data, wait_for_response=True
        )

    async def trigger_channel_optimization(
        self, channel_id: int, optimization_goals: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Trigger channel optimization workflow"""

        data = {
            "channel_id": channel_id,
            "optimization_goals": optimization_goals
            or [
                "increase_views",
                "improve_retention",
                "optimize_posting_time",
                "enhance_thumbnails",
            ],
            "analysis_period_days": 30,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return await self.trigger_workflow(
            WorkflowType.CHANNEL_OPTIMIZATION,
            data,
            wait_for_response=False,  # Long running task
        )

    async def trigger_content_scheduling(
        self, channel_id: int, days_ahead: int = 7, videos_per_day: int = 2
    ) -> Optional[Dict[str, Any]]:
        """Trigger content scheduling workflow"""

        data = {
            "channel_id": channel_id,
            "days_ahead": days_ahead,
            "videos_per_day": videos_per_day,
            "optimize_timing": True,
            "consider_trends": True,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return await self.trigger_workflow(
            WorkflowType.CONTENT_SCHEDULING, data, wait_for_response=True
        )

    async def batch_trigger_workflows(
        self, workflows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Trigger multiple workflows in parallel"""

        tasks = []
        for workflow in workflows:
            task = self.trigger_workflow(
                workflow["type"],
                workflow["data"],
                workflow.get("wait_for_response", True),
                workflow.get("timeout", 300),
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                output.append(
                    {
                        "workflow": workflows[i]["type"].value,
                        "status": "error",
                        "error": str(result),
                    }
                )
            else:
                output.append(
                    {
                        "workflow": workflows[i]["type"].value,
                        "status": "success",
                        "result": result,
                    }
                )

        return output

    async def get_workflow_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow execution"""

        try:
            async with httpx.AsyncClient() as client:
                headers = {}
                if self.api_key:
                    headers["X-N8N-API-KEY"] = self.api_key

                response = await client.get(
                    f"{self.base_url}/executions/{execution_id}", headers=headers
                )

                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")
            return None

    async def list_workflow_executions(
        self, workflow_id: str = None, status: str = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """List workflow executions"""

        try:
            async with httpx.AsyncClient() as client:
                headers = {}
                if self.api_key:
                    headers["X-N8N-API-KEY"] = self.api_key

                params = {"limit": limit}
                if workflow_id:
                    params["workflowId"] = workflow_id
                if status:
                    params["status"] = status

                response = await client.get(
                    f"{self.base_url}/executions", headers=headers, params=params
                )

                response.raise_for_status()
                return response.json().get("data", [])

        except Exception as e:
            logger.error(f"Failed to list workflow executions: {e}")
            return []


# Singleton instance
_n8n_instance = None


def get_n8n_integration() -> N8NIntegration:
    """Get singleton instance of N8N integration"""
    global _n8n_instance
    if _n8n_instance is None:
        _n8n_instance = N8NIntegration()
    return _n8n_instance
