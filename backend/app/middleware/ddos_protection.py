"""
DDoS Protection middleware for YTEmpire.
Implements multiple layers of DDoS mitigation including rate limiting, 
connection throttling, and pattern detection.
"""

import time
import asyncio
import hashlib
from typing import Dict, Optional, Set, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import ipaddress

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis

from app.core.audit_logging import audit_logger, AuditEventType, AuditSeverity

logger = logging.getLogger(__name__)

# ============================================================================
# DDoS Protection Configuration
# ============================================================================


class DDoSConfig:
    """DDoS protection configuration"""

    # Connection limits
    MAX_CONNECTIONS_PER_IP = 100  # Maximum concurrent connections per IP
    MAX_REQUESTS_PER_SECOND = 10  # Maximum requests per second per IP
    MAX_REQUESTS_PER_MINUTE = 300  # Maximum requests per minute per IP
    MAX_REQUESTS_PER_HOUR = 5000  # Maximum requests per hour per IP

    # Burst limits
    BURST_THRESHOLD = 50  # Requests in burst window
    BURST_WINDOW = 10  # Burst window in seconds

    # Pattern detection
    PATTERN_WINDOW = 60  # Pattern detection window in seconds
    PATTERN_THRESHOLD = 100  # Threshold for pattern detection

    # Slowloris protection
    MIN_REQUEST_INTERVAL = 0.1  # Minimum interval between requests (seconds)
    MAX_HEADER_SIZE = 8192  # Maximum header size in bytes
    MAX_BODY_SIZE = 10485760  # Maximum body size (10MB)
    REQUEST_TIMEOUT = 30  # Request timeout in seconds

    # Geographic limits
    COUNTRY_BLACKLIST = []  # List of blocked country codes
    COUNTRY_WHITELIST = []  # List of allowed country codes (empty = all allowed)

    # Challenge-response
    ENABLE_CHALLENGE = True  # Enable challenge-response for suspicious traffic
    CHALLENGE_THRESHOLD = 100  # Requests before challenge

    # Blocking durations
    TEMP_BLOCK_DURATION = 300  # 5 minutes
    MEDIUM_BLOCK_DURATION = 3600  # 1 hour
    PERMANENT_BLOCK_DURATION = 86400  # 24 hours


# ============================================================================
# DDoS Detection Engine
# ============================================================================


class DDoSDetector:
    """DDoS attack detection engine"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.connection_count: Dict[str, int] = defaultdict(int)
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.blocked_ips: Set[str] = set()
        self.challenged_ips: Set[str] = set()
        self.whitelisted_ips: Set[str] = {
            "127.0.0.1",
            "::1",  # IPv6 localhost
        }

        # Pattern detection
        self.request_patterns: Dict[str, List[float]] = defaultdict(list)
        self.syn_flood_detection: Dict[str, int] = defaultdict(int)

    async def check_request(
        self, request: Request
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Check request for DDoS patterns
        Returns: (is_allowed, block_reason, metadata)
        """

        client_ip = self._get_client_ip(request)

        # Check whitelist
        if client_ip in self.whitelisted_ips:
            return True, None, None

        # Check if IP is blocked
        if await self._is_blocked(client_ip):
            return False, "ip_blocked", {"ip": client_ip}

        # Check connection limit
        if not await self._check_connection_limit(client_ip):
            return (
                False,
                "connection_limit",
                {"ip": client_ip, "limit": DDoSConfig.MAX_CONNECTIONS_PER_IP},
            )

        # Check rate limits
        rate_check = await self._check_rate_limits(client_ip)
        if not rate_check[0]:
            return (
                False,
                f"rate_limit_{rate_check[1]}",
                {"ip": client_ip, "limit_type": rate_check[1]},
            )

        # Check for burst patterns
        if await self._detect_burst(client_ip):
            return False, "burst_detected", {"ip": client_ip}

        # Check for attack patterns
        pattern = await self._detect_attack_pattern(client_ip, request)
        if pattern:
            return (
                False,
                f"attack_pattern_{pattern}",
                {"ip": client_ip, "pattern": pattern},
            )

        # Check for slowloris attack
        if await self._detect_slowloris(client_ip, request):
            return False, "slowloris_attack", {"ip": client_ip}

        # Check for amplification attack
        if self._detect_amplification(request):
            return False, "amplification_attack", {"ip": client_ip}

        # Challenge suspicious IPs
        if await self._should_challenge(client_ip):
            return await self._issue_challenge(client_ip, request)

        # Record request
        await self._record_request(client_ip)

        return True, None, None

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for proxy headers
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    async def _is_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        if ip in self.blocked_ips:
            return True

        if self.redis_client:
            try:
                blocked = await self.redis_client.get(f"ddos:blocked:{ip}")
                return blocked is not None
            except Exception as e:
                logger.error(f"Error checking blocked IP: {e}")

        return False

    async def _check_connection_limit(self, ip: str) -> bool:
        """Check connection limit per IP"""
        self.connection_count[ip] += 1

        if self.connection_count[ip] > DDoSConfig.MAX_CONNECTIONS_PER_IP:
            logger.warning(
                f"Connection limit exceeded for {ip}: {self.connection_count[ip]}"
            )
            return False

        return True

    async def _check_rate_limits(self, ip: str) -> Tuple[bool, Optional[str]]:
        """Check various rate limits"""
        current_time = time.time()

        # Clean old entries
        self.request_history[ip] = deque(
            [t for t in self.request_history[ip] if current_time - t < 3600],
            maxlen=1000,
        )

        history = self.request_history[ip]

        # Check per-second limit
        recent_second = [t for t in history if current_time - t < 1]
        if len(recent_second) > DDoSConfig.MAX_REQUESTS_PER_SECOND:
            return False, "second"

        # Check per-minute limit
        recent_minute = [t for t in history if current_time - t < 60]
        if len(recent_minute) > DDoSConfig.MAX_REQUESTS_PER_MINUTE:
            return False, "minute"

        # Check per-hour limit
        recent_hour = [t for t in history if current_time - t < 3600]
        if len(recent_hour) > DDoSConfig.MAX_REQUESTS_PER_HOUR:
            return False, "hour"

        return True, None

    async def _detect_burst(self, ip: str) -> bool:
        """Detect burst patterns"""
        current_time = time.time()
        history = self.request_history[ip]

        # Count requests in burst window
        burst_requests = [
            t for t in history if current_time - t < DDoSConfig.BURST_WINDOW
        ]

        if len(burst_requests) > DDoSConfig.BURST_THRESHOLD:
            logger.warning(
                f"Burst detected from {ip}: {len(burst_requests)} requests in {DDoSConfig.BURST_WINDOW}s"
            )
            return True

        return False

    async def _detect_attack_pattern(self, ip: str, request: Request) -> Optional[str]:
        """Detect specific attack patterns"""

        # Check for SYN flood pattern (many connections, no data)
        if self.syn_flood_detection[ip] > 10:
            return "syn_flood"

        # Check for HTTP flood (same endpoint repeatedly)
        path = request.url.path
        if self.redis_client:
            try:
                key = f"ddos:endpoint:{ip}:{path}"
                count = await self.redis_client.incr(key)
                await self.redis_client.expire(key, 60)

                if count > 100:  # Same endpoint hit 100+ times in a minute
                    return "http_flood"
            except Exception as e:
                logger.error(f"Error detecting HTTP flood: {e}")

        # Check for random URL pattern (DDoS bots often use random URLs)
        if await self._detect_random_urls(ip, request):
            return "random_url"

        # Check for reflection attack pattern
        if self._detect_reflection_pattern(request):
            return "reflection"

        return None

    async def _detect_random_urls(self, ip: str, request: Request) -> bool:
        """Detect random URL pattern"""
        if self.redis_client:
            try:
                # Track unique URLs per IP
                key = f"ddos:urls:{ip}"
                await self.redis_client.sadd(key, request.url.path)
                await self.redis_client.expire(key, 60)

                # If too many unique URLs in short time, likely bot
                unique_urls = await self.redis_client.scard(key)
                if unique_urls > 50:  # 50+ unique URLs in a minute
                    return True
            except Exception as e:
                logger.error(f"Error detecting random URLs: {e}")

        return False

    def _detect_reflection_pattern(self, request: Request) -> bool:
        """Detect reflection/amplification attack pattern"""
        # Check for suspiciously large response-inducing requests
        if request.method == "GET":
            # Check for requests that might trigger large responses
            suspicious_params = ["size", "count", "limit", "max", "all"]
            query_params = str(request.url.query).lower()

            for param in suspicious_params:
                if param in query_params:
                    # Check if value is suspiciously large
                    try:
                        import re

                        pattern = f"{param}=(\\d+)"
                        match = re.search(pattern, query_params)
                        if match:
                            value = int(match.group(1))
                            if value > 1000:
                                return True
                    except:
                        pass

        return False

    async def _detect_slowloris(self, ip: str, request: Request) -> bool:
        """Detect Slowloris attack"""
        # Check for incomplete headers or slow data transmission
        content_length = request.headers.get("content-length")

        if content_length:
            try:
                length = int(content_length)
                if length > DDoSConfig.MAX_BODY_SIZE:
                    return True
            except:
                pass

        # Check for suspiciously slow connections
        if self.redis_client:
            try:
                key = f"ddos:slow:{ip}"
                await self.redis_client.incr(key)
                await self.redis_client.expire(key, 60)

                count = await self.redis_client.get(key)
                if int(count) > 10:  # Many slow connections from same IP
                    return True
            except Exception as e:
                logger.error(f"Error detecting Slowloris: {e}")

        return False

    def _detect_amplification(self, request: Request) -> bool:
        """Detect amplification attack attempts"""
        # Check for small requests that might trigger large responses
        if request.method == "GET":
            # Check for recursive DNS queries or similar
            if "/dns" in request.url.path or "/resolve" in request.url.path:
                return True

            # Check for requests to data export endpoints
            if "/export" in request.url.path or "/download" in request.url.path:
                # Small request for potentially large data
                if not request.headers.get("range"):  # No range header = full download
                    return True

        return False

    async def _should_challenge(self, ip: str) -> bool:
        """Determine if IP should be challenged"""
        if not DDoSConfig.ENABLE_CHALLENGE:
            return False

        if ip in self.challenged_ips:
            return False

        # Challenge after threshold
        history = self.request_history[ip]
        if len(history) > DDoSConfig.CHALLENGE_THRESHOLD:
            return True

        return False

    async def _issue_challenge(
        self, ip: str, request: Request
    ) -> Tuple[bool, str, Dict]:
        """Issue challenge to suspicious IP"""
        # Check if challenge response is provided
        challenge_response = request.headers.get("X-Challenge-Response")

        if challenge_response:
            # Verify challenge response
            if await self._verify_challenge(ip, challenge_response):
                self.challenged_ips.add(ip)
                return True, None, None
            else:
                return False, "invalid_challenge", {"ip": ip}

        # Issue new challenge
        challenge = self._generate_challenge(ip)
        return False, "challenge_required", {"ip": ip, "challenge": challenge}

    def _generate_challenge(self, ip: str) -> str:
        """Generate challenge for IP"""
        timestamp = int(time.time())
        data = f"{ip}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    async def _verify_challenge(self, ip: str, response: str) -> bool:
        """Verify challenge response"""
        # Simple verification - in production, use more sophisticated challenges
        expected = self._generate_challenge(ip)
        return response == expected

    async def _record_request(self, ip: str):
        """Record request for analysis"""
        current_time = time.time()
        self.request_history[ip].append(current_time)

        # Update patterns
        self.request_patterns[ip].append(current_time)

        # Clean old pattern data
        self.request_patterns[ip] = [
            t
            for t in self.request_patterns[ip]
            if current_time - t < DDoSConfig.PATTERN_WINDOW
        ]

    async def block_ip(self, ip: str, duration: int, reason: str):
        """Block an IP address"""
        self.blocked_ips.add(ip)

        if self.redis_client:
            try:
                await self.redis_client.setex(f"ddos:blocked:{ip}", duration, reason)
            except Exception as e:
                logger.error(f"Error blocking IP: {e}")

        logger.warning(f"Blocked IP {ip} for {duration}s: {reason}")

        # Log security event
        await audit_logger.log_security_event(
            event_type=AuditEventType.INTRUSION_ATTEMPT,
            description=f"IP blocked for DDoS: {reason}",
            ip_address=ip,
            severity=AuditSeverity.HIGH,
            threat_indicators=["ddos", reason],
            metadata={"block_duration": duration},
        )

    def release_connection(self, ip: str):
        """Release connection count for IP"""
        if ip in self.connection_count:
            self.connection_count[ip] = max(0, self.connection_count[ip] - 1)


# ============================================================================
# DDoS Protection Middleware
# ============================================================================


class DDoSProtectionMiddleware(BaseHTTPMiddleware):
    """DDoS protection middleware"""

    def __init__(
        self, app, redis_client: Optional[redis.Redis] = None, enabled: bool = True
    ):
        super().__init__(app)
        self.detector = DDoSDetector(redis_client)
        self.enabled = enabled
        self.redis_client = redis_client

    async def dispatch(self, request: Request, call_next):
        """Process request through DDoS protection"""

        if not self.enabled:
            return await call_next(request)

        # Skip protection for health checks
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)

        client_ip = self.detector._get_client_ip(request)

        try:
            # Check for DDoS patterns
            is_allowed, block_reason, metadata = await self.detector.check_request(
                request
            )

            if not is_allowed:
                # Determine block duration based on severity
                if block_reason == "challenge_required":
                    # Return challenge
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Too Many Requests",
                            "message": "Please complete the challenge",
                            "challenge": metadata.get("challenge"),
                        },
                        headers={
                            "X-Challenge": metadata.get("challenge", ""),
                            "Retry-After": "10",
                        },
                    )

                # Determine block duration
                duration = DDoSConfig.TEMP_BLOCK_DURATION
                if "flood" in block_reason or "attack" in block_reason:
                    duration = DDoSConfig.MEDIUM_BLOCK_DURATION
                if block_reason == "ip_blocked":
                    duration = DDoSConfig.PERMANENT_BLOCK_DURATION

                # Block the IP
                await self.detector.block_ip(client_ip, duration, block_reason)

                # Return 429 Too Many Requests
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Too Many Requests",
                        "message": "Request rate limit exceeded",
                        "retry_after": duration,
                    },
                    headers={
                        "Retry-After": str(duration),
                        "X-RateLimit-Limit": str(DDoSConfig.MAX_REQUESTS_PER_MINUTE),
                        "X-RateLimit-Reset": str(int(time.time()) + duration),
                    },
                )

            # Process request
            response = await call_next(request)

            # Release connection
            self.detector.release_connection(client_ip)

            # Add rate limit headers
            if hasattr(response, "headers"):
                response.headers["X-RateLimit-Limit"] = str(
                    DDoSConfig.MAX_REQUESTS_PER_MINUTE
                )
                response.headers["X-RateLimit-Remaining"] = str(
                    DDoSConfig.MAX_REQUESTS_PER_MINUTE
                    - len(self.detector.request_history.get(client_ip, []))
                )

            return response

        except Exception as e:
            logger.error(f"DDoS protection error: {e}")
            # Release connection on error
            self.detector.release_connection(client_ip)
            # On error, allow request but log
            return await call_next(request)


# ============================================================================
# DDoS Mitigation Service
# ============================================================================


class DDoSMitigationService:
    """Advanced DDoS mitigation service"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.detector = DDoSDetector(redis_client)

    async def analyze_traffic_patterns(self) -> Dict:
        """Analyze current traffic patterns"""
        analysis = {
            "total_ips": len(self.detector.request_history),
            "blocked_ips": len(self.detector.blocked_ips),
            "active_connections": sum(self.detector.connection_count.values()),
            "patterns_detected": [],
            "threat_level": "low",
        }

        # Analyze patterns
        for ip, history in self.detector.request_history.items():
            if len(history) > DDoSConfig.PATTERN_THRESHOLD:
                analysis["patterns_detected"].append(
                    {"ip": ip, "requests": len(history), "pattern": "high_volume"}
                )

        # Determine threat level
        if analysis["blocked_ips"] > 10:
            analysis["threat_level"] = "high"
        elif analysis["blocked_ips"] > 5:
            analysis["threat_level"] = "medium"

        return analysis

    async def get_blocked_ips(self) -> List[Dict]:
        """Get list of blocked IPs"""
        blocked = []

        for ip in self.detector.blocked_ips:
            blocked.append(
                {
                    "ip": ip,
                    "blocked_at": datetime.utcnow().isoformat(),
                    "reason": "DDoS protection",
                }
            )

        # Get from Redis
        if self.redis_client:
            try:
                keys = await self.redis_client.keys("ddos:blocked:*")
                for key in keys:
                    ip = key.decode().split(":")[-1]
                    reason = await self.redis_client.get(key)
                    blocked.append(
                        {
                            "ip": ip,
                            "blocked_at": datetime.utcnow().isoformat(),
                            "reason": reason.decode() if reason else "Unknown",
                        }
                    )
            except Exception as e:
                logger.error(f"Error getting blocked IPs: {e}")

        return blocked

    async def unblock_ip(self, ip: str):
        """Manually unblock an IP"""
        self.detector.blocked_ips.discard(ip)

        if self.redis_client:
            try:
                await self.redis_client.delete(f"ddos:blocked:{ip}")
            except Exception as e:
                logger.error(f"Error unblocking IP: {e}")

        logger.info(f"Manually unblocked IP: {ip}")

    async def enable_emergency_mode(self):
        """Enable emergency DDoS protection mode"""
        # Reduce all limits
        DDoSConfig.MAX_REQUESTS_PER_SECOND = 1
        DDoSConfig.MAX_REQUESTS_PER_MINUTE = 10
        DDoSConfig.MAX_REQUESTS_PER_HOUR = 100
        DDoSConfig.ENABLE_CHALLENGE = True

        logger.warning("Emergency DDoS protection mode enabled")

        await audit_logger.log_security_event(
            event_type=AuditEventType.SECURITY_ALERT,
            description="Emergency DDoS protection mode enabled",
            ip_address="system",
            severity=AuditSeverity.CRITICAL,
            threat_indicators=["ddos", "emergency_mode"],
        )

    async def disable_emergency_mode(self):
        """Disable emergency mode"""
        # Restore normal limits
        DDoSConfig.MAX_REQUESTS_PER_SECOND = 10
        DDoSConfig.MAX_REQUESTS_PER_MINUTE = 300
        DDoSConfig.MAX_REQUESTS_PER_HOUR = 5000

        logger.info("Emergency DDoS protection mode disabled")
