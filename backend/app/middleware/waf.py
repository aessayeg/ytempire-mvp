"""
Web Application Firewall (WAF) middleware for YTEmpire.
Protects against common web attacks including SQL injection, XSS, and more.
"""

import re
import hashlib
import json
import time
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
import logging
from urllib.parse import unquote

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import Headers
import redis.asyncio as redis

from app.core.audit_logging import audit_logger, AuditEventType, AuditSeverity

logger = logging.getLogger(__name__)

# ============================================================================
# WAF Rules and Patterns
# ============================================================================


class WAFRules:
    """WAF detection rules and patterns"""

    # SQL Injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|FROM|WHERE|HAVING|GROUP BY|ORDER BY)\b)",
        r"(--|\#|\/\*|\*\/)",  # SQL comments
        r"(\bOR\b\s*\d+\s*=\s*\d+)",  # OR 1=1
        r"(\bAND\b\s*\d+\s*=\s*\d+)",  # AND 1=1
        r"(';|';--|';\s*\/\*)",  # Common injection endings
        r"(\bSLEEP\s*\(|\bBENCHMARK\s*\()",  # Time-based attacks
        r"(xp_cmdshell|sp_executesql)",  # MSSQL specific
        r"(\bINTO\s+(OUTFILE|DUMPFILE)\b)",  # File operations
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"(<script[^>]*>.*?</script>)",  # Script tags
        r"(javascript:|vbscript:|livescript:)",  # Script protocols
        r"(on\w+\s*=)",  # Event handlers
        r"(<iframe[^>]*>.*?</iframe>)",  # Iframes
        r"(<object[^>]*>.*?</object>)",  # Objects
        r"(<embed[^>]*>.*?</embed>)",  # Embeds
        r"(eval\s*\()",  # Eval functions
        r"(alert\s*\(|confirm\s*\(|prompt\s*\()",  # JS dialogs
        r"(document\.(cookie|write|location))",  # Document manipulation
        r"(window\.(location|open))",  # Window manipulation
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"(\.\./|\.\.\%2[fF]/)",  # Directory traversal
        r"(/etc/passwd|/etc/shadow)",  # Unix system files
        r"(C:\\|C:\%5[cC])",  # Windows paths
        r"(\%00|\x00)",  # Null bytes
        r"(\.\.\\|\.\.\%5[cC])",  # Windows traversal
    ]

    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"(;|\||&&|\|\||`)",  # Command separators
        r"(\$\(.*\))",  # Command substitution
        r"(>\s*/dev/null)",  # Output redirection
        r"(nc\s+-|\bnc\b.*\s+-e)",  # Netcat
        r"(bash\s+-i|/bin/sh)",  # Shell invocation
        r"(wget\s+|curl\s+)",  # Remote file download
    ]

    # XXE patterns
    XXE_PATTERNS = [
        r"(<!DOCTYPE[^>]*\[)",  # DOCTYPE with entities
        r"(<!ENTITY[^>]*>)",  # Entity declarations
        r"(SYSTEM\s+[\"']file:)",  # File protocol
        r"(SYSTEM\s+[\"']http:)",  # HTTP protocol
    ]

    # LDAP injection patterns
    LDAP_INJECTION_PATTERNS = [
        r"(\*\||\|\*)",  # LDAP wildcards
        r"(\)\(|\(\))",  # Parenthesis manipulation
        r"(cn=|ou=|dc=)",  # LDAP attributes
    ]

    # NoSQL injection patterns
    NOSQL_INJECTION_PATTERNS = [
        r"(\$where|\$regex|\$ne|\$gt|\$lt)",  # MongoDB operators
        r"({.*:.*})",  # JSON-like structures in params
        r"(\[.*\])",  # Array structures in params
    ]

    # Suspicious user agents
    SUSPICIOUS_USER_AGENTS = [
        r"(sqlmap|nikto|nessus|metasploit|nmap|masscan)",  # Security tools
        r"(bot|crawler|spider|scraper)",  # Automated tools
        r"(curl|wget|python-requests|httpie)",  # Command line tools
        r"(scanner|vulnerability|exploit)",  # Scanning tools
    ]

    # Suspicious file extensions
    DANGEROUS_FILE_EXTENSIONS = {
        ".exe",
        ".dll",
        ".scr",
        ".bat",
        ".cmd",
        ".com",
        ".pif",  # Windows executables
        ".sh",
        ".bash",
        ".zsh",  # Unix scripts
        ".app",
        ".deb",
        ".rpm",  # Application packages
        ".jsp",
        ".asp",
        ".aspx",
        ".php",
        ".cgi",  # Server scripts
        ".swf",
        ".jar",  # Java/Flash
    }

    # Rate limiting rules (requests per minute)
    RATE_LIMITS = {
        "global": 1000,  # Global rate limit
        "auth": 10,  # Authentication endpoints
        "api": 100,  # API endpoints
        "upload": 20,  # File upload endpoints
        "export": 5,  # Data export endpoints
    }


# ============================================================================
# WAF Engine
# ============================================================================


class WAFEngine:
    """Core WAF detection and prevention engine"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.compiled_patterns = self._compile_patterns()
        self.blocked_ips: Set[str] = set()
        self.whitelist_ips: Set[str] = set()
        self.blacklist_patterns: List[re.Pattern] = []

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for performance"""
        return {
            "sql_injection": [
                re.compile(p, re.IGNORECASE) for p in WAFRules.SQL_INJECTION_PATTERNS
            ],
            "xss": [re.compile(p, re.IGNORECASE) for p in WAFRules.XSS_PATTERNS],
            "path_traversal": [
                re.compile(p, re.IGNORECASE) for p in WAFRules.PATH_TRAVERSAL_PATTERNS
            ],
            "command_injection": [
                re.compile(p, re.IGNORECASE)
                for p in WAFRules.COMMAND_INJECTION_PATTERNS
            ],
            "xxe": [re.compile(p, re.IGNORECASE) for p in WAFRules.XXE_PATTERNS],
            "ldap_injection": [
                re.compile(p, re.IGNORECASE) for p in WAFRules.LDAP_INJECTION_PATTERNS
            ],
            "nosql_injection": [
                re.compile(p, re.IGNORECASE) for p in WAFRules.NOSQL_INJECTION_PATTERNS
            ],
            "user_agent": [
                re.compile(p, re.IGNORECASE) for p in WAFRules.SUSPICIOUS_USER_AGENTS
            ],
        }

    async def check_request(
        self, request: Request
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check request for threats
        Returns: (is_safe, threat_type, threat_details)
        """

        # Check if IP is whitelisted
        client_ip = request.client.host if request.client else "unknown"
        if client_ip in self.whitelist_ips:
            return True, None, None

        # Check if IP is blocked
        if await self._is_ip_blocked(client_ip):
            return False, "blocked_ip", f"IP {client_ip} is blocked"

        # Check rate limiting
        is_within_limit, limit_type = await self._check_rate_limit(request)
        if not is_within_limit:
            return False, "rate_limit", f"Rate limit exceeded for {limit_type}"

        # Check User-Agent
        user_agent = request.headers.get("user-agent", "")
        threat = self._check_user_agent(user_agent)
        if threat:
            return False, "suspicious_user_agent", threat

        # Check URL path
        threat = self._check_path(request.url.path)
        if threat:
            return False, threat[0], threat[1]

        # Check query parameters
        threat = await self._check_query_params(request)
        if threat:
            return False, threat[0], threat[1]

        # Check headers
        threat = self._check_headers(request.headers)
        if threat:
            return False, threat[0], threat[1]

        # Check request body (if applicable)
        if request.method in ["POST", "PUT", "PATCH"]:
            threat = await self._check_body(request)
            if threat:
                return False, threat[0], threat[1]

        return True, None, None

    def _check_user_agent(self, user_agent: str) -> Optional[str]:
        """Check user agent for suspicious patterns"""
        for pattern in self.compiled_patterns["user_agent"]:
            if pattern.search(user_agent):
                return f"Suspicious user agent detected: {user_agent[:100]}"
        return None

    def _check_path(self, path: str) -> Optional[Tuple[str, str]]:
        """Check URL path for attacks"""
        decoded_path = unquote(path)

        # Check for path traversal
        for pattern in self.compiled_patterns["path_traversal"]:
            if pattern.search(decoded_path):
                return "path_traversal", f"Path traversal attempt in URL: {path[:100]}"

        # Check for suspicious file extensions
        for ext in WAFRules.DANGEROUS_FILE_EXTENSIONS:
            if decoded_path.lower().endswith(ext):
                return "dangerous_file", f"Dangerous file extension: {ext}"

        return None

    async def _check_query_params(self, request: Request) -> Optional[Tuple[str, str]]:
        """Check query parameters for attacks"""
        query_string = str(request.url.query)
        if not query_string:
            return None

        decoded_query = unquote(query_string)

        # Check each attack type
        checks = [
            ("sql_injection", "SQL injection"),
            ("xss", "XSS"),
            ("command_injection", "Command injection"),
            ("ldap_injection", "LDAP injection"),
            ("nosql_injection", "NoSQL injection"),
        ]

        for check_type, threat_name in checks:
            for pattern in self.compiled_patterns[check_type]:
                if pattern.search(decoded_query):
                    return check_type, f"{threat_name} attempt in query parameters"

        return None

    def _check_headers(self, headers: Headers) -> Optional[Tuple[str, str]]:
        """Check headers for attacks"""
        # Check for header injection
        dangerous_headers = ["X-Forwarded-Host", "X-Original-URL", "X-Rewrite-URL"]

        for header in dangerous_headers:
            if header in headers:
                value = headers[header]
                # Check for injection in header value
                for pattern in self.compiled_patterns["xss"]:
                    if pattern.search(value):
                        return (
                            "header_injection",
                            f"Potential header injection in {header}",
                        )

        # Check for HTTP response splitting
        for name, value in headers.items():
            if "\r" in value or "\n" in value:
                return (
                    "response_splitting",
                    f"HTTP response splitting attempt in header {name}",
                )

        return None

    async def _check_body(self, request: Request) -> Optional[Tuple[str, str]]:
        """Check request body for attacks"""
        try:
            # Get content type
            content_type = request.headers.get("content-type", "")

            # Read body
            body = await request.body()
            if not body:
                return None

            body_text = body.decode("utf-8", errors="ignore")

            # Check for XXE in XML content
            if "xml" in content_type.lower():
                for pattern in self.compiled_patterns["xxe"]:
                    if pattern.search(body_text):
                        return "xxe", "XXE attack attempt in request body"

            # Check for various injection attacks
            checks = [
                ("sql_injection", "SQL injection"),
                ("xss", "XSS"),
                ("command_injection", "Command injection"),
                ("nosql_injection", "NoSQL injection"),
            ]

            for check_type, threat_name in checks:
                for pattern in self.compiled_patterns[check_type]:
                    if pattern.search(body_text):
                        return check_type, f"{threat_name} attempt in request body"

        except Exception as e:
            logger.error(f"Error checking request body: {e}")

        return None

    async def _check_rate_limit(self, request: Request) -> Tuple[bool, Optional[str]]:
        """Check rate limiting"""
        if not self.redis_client:
            return True, None

        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path

        # Determine rate limit type
        limit_type = "api"
        limit = WAFRules.RATE_LIMITS.get(limit_type, 100)

        if "/auth" in path:
            limit_type = "auth"
            limit = WAFRules.RATE_LIMITS["auth"]
        elif "/upload" in path:
            limit_type = "upload"
            limit = WAFRules.RATE_LIMITS["upload"]
        elif "/export" in path:
            limit_type = "export"
            limit = WAFRules.RATE_LIMITS["export"]

        # Check rate limit
        key = f"waf:rate:{limit_type}:{client_ip}"

        try:
            current = await self.redis_client.get(key)

            if current is None:
                await self.redis_client.setex(key, 60, 1)
                return True, None

            count = int(current)
            if count >= limit:
                return False, limit_type

            await self.redis_client.incr(key)
            return True, None

        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return True, None

    async def _is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        if ip in self.blocked_ips:
            return True

        if self.redis_client:
            try:
                blocked = await self.redis_client.get(f"waf:blocked:{ip}")
                return blocked is not None
            except Exception as e:
                logger.error(f"Error checking blocked IP: {e}")

        return False

    async def block_ip(self, ip: str, duration: int = 3600, reason: str = ""):
        """Block an IP address"""
        self.blocked_ips.add(ip)

        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"waf:blocked:{ip}",
                    duration,
                    json.dumps(
                        {
                            "reason": reason,
                            "blocked_at": datetime.utcnow().isoformat(),
                            "duration": duration,
                        }
                    ),
                )
            except Exception as e:
                logger.error(f"Error blocking IP: {e}")

        logger.warning(f"Blocked IP {ip} for {duration} seconds: {reason}")

    async def unblock_ip(self, ip: str):
        """Unblock an IP address"""
        self.blocked_ips.discard(ip)

        if self.redis_client:
            try:
                await self.redis_client.delete(f"waf:blocked:{ip}")
            except Exception as e:
                logger.error(f"Error unblocking IP: {e}")

        logger.info(f"Unblocked IP {ip}")


# ============================================================================
# WAF Middleware
# ============================================================================


class WAFMiddleware(BaseHTTPMiddleware):
    """WAF middleware for FastAPI"""

    def __init__(
        self, app, redis_client: Optional[redis.Redis] = None, enabled: bool = True
    ):
        super().__init__(app)
        self.waf_engine = WAFEngine(redis_client)
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next):
        """Process request through WAF"""

        if not self.enabled:
            return await call_next(request)

        # Skip WAF for health checks and metrics
        if request.url.path in ["/health", "/metrics", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        try:
            # Check request through WAF
            start_time = time.time()
            is_safe, threat_type, threat_details = await self.waf_engine.check_request(
                request
            )
            check_time = time.time() - start_time

            if not is_safe:
                # Log security event
                await audit_logger.log_security_event(
                    event_type=self._get_audit_event_type(threat_type),
                    description=threat_details or f"WAF blocked: {threat_type}",
                    ip_address=client_ip,
                    severity=AuditSeverity.HIGH,
                    threat_indicators=[threat_type],
                    metadata={
                        "path": str(request.url.path),
                        "method": request.method,
                        "user_agent": request.headers.get("user-agent", ""),
                        "check_time_ms": check_time * 1000,
                    },
                )

                # Block IP for repeated offenses
                if threat_type in ["sql_injection", "xss", "command_injection"]:
                    await self._check_and_block_repeat_offender(client_ip, threat_type)

                # Return 403 Forbidden
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Forbidden",
                        "message": "Request blocked by security policy",
                        "request_id": request.headers.get("x-request-id", ""),
                    },
                )

            # Process request
            response = await call_next(request)

            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Content-Security-Policy"] = "default-src 'self'"

            return response

        except Exception as e:
            logger.error(f"WAF error: {e}")
            # On error, allow request but log
            return await call_next(request)

    def _get_audit_event_type(self, threat_type: str) -> AuditEventType:
        """Map threat type to audit event type"""
        mapping = {
            "sql_injection": AuditEventType.SQL_INJECTION_ATTEMPT,
            "xss": AuditEventType.XSS_ATTEMPT,
            "blocked_ip": AuditEventType.INTRUSION_ATTEMPT,
            "rate_limit": AuditEventType.API_RATE_LIMIT,
            "suspicious_user_agent": AuditEventType.SUSPICIOUS_ACTIVITY,
            "path_traversal": AuditEventType.INTRUSION_ATTEMPT,
            "command_injection": AuditEventType.INTRUSION_ATTEMPT,
            "xxe": AuditEventType.INTRUSION_ATTEMPT,
            "ldap_injection": AuditEventType.INTRUSION_ATTEMPT,
            "nosql_injection": AuditEventType.INTRUSION_ATTEMPT,
        }
        return mapping.get(threat_type, AuditEventType.SECURITY_ALERT)

    async def _check_and_block_repeat_offender(self, ip: str, threat_type: str):
        """Block IP after repeated offenses"""
        if not self.waf_engine.redis_client:
            return

        try:
            # Track offense count
            key = f"waf:offenses:{ip}"
            count = await self.waf_engine.redis_client.incr(key)
            await self.waf_engine.redis_client.expire(key, 3600)  # Reset after 1 hour

            # Block after 3 offenses
            if count >= 3:
                await self.waf_engine.block_ip(
                    ip,
                    duration=86400,  # 24 hours
                    reason=f"Repeated {threat_type} attempts",
                )

                # Log critical security event
                await audit_logger.log_security_event(
                    event_type=AuditEventType.BRUTE_FORCE_ATTEMPT,
                    description=f"IP blocked for repeated {threat_type} attempts",
                    ip_address=ip,
                    severity=AuditSeverity.CRITICAL,
                    threat_indicators=[threat_type, "repeat_offender"],
                    metadata={"offense_count": count},
                )

        except Exception as e:
            logger.error(f"Error tracking repeat offender: {e}")
