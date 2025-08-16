"""
Input Validation and Sanitization Middleware
Prevents injection attacks and validates all user inputs
"""
import re
import html
import json
import logging
from typing import Any, Dict, List, Optional, Union
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import sqlparse
from urllib.parse import unquote, urlparse
import bleach

logger = logging.getLogger(__name__)


class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive input validation and sanitization middleware
    Prevents SQL injection, XSS, command injection, and other attacks
    """

    def __init__(
        self,
        app: ASGIApp,
        max_field_length: int = 10000,
        max_json_depth: int = 10,
        block_file_extensions: List[str] = None,
    ):
        super().__init__(app)
        self.max_field_length = max_field_length
        self.max_json_depth = max_json_depth
        self.block_file_extensions = block_file_extensions or [
            ".exe",
            ".dll",
            ".bat",
            ".cmd",
            ".com",
            ".pif",
            ".scr",
            ".vbs",
            ".js",
            ".jar",
            ".zip",
            ".rar",
        ]

        # SQL injection patterns
        self.sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|FROM|WHERE|JOIN|ORDER BY|GROUP BY|HAVING)\b)",
            r"(--|#|\/\*|\*\/)",  # SQL comments
            r"(\bOR\b\s*\d+\s*=\s*\d+)",  # OR 1=1
            r"(\bAND\b\s*\d+\s*=\s*\d+)",  # AND 1=1
            r"(xp_cmdshell|sp_execute|sp_executesql)",  # SQL Server specific
            r"(WAITFOR\s+DELAY|BENCHMARK|SLEEP)",  # Time-based attacks
        ]

        # XSS patterns
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",  # Event handlers
            r"<iframe[^>]*>",
            r"<embed[^>]*>",
            r"<object[^>]*>",
            r"eval\s*\(",
            r"expression\s*\(",
        ]

        # Command injection patterns
        self.command_patterns = [
            r"[;&|`$]",  # Shell metacharacters
            r"\$\(",  # Command substitution
            r"\.\./",  # Directory traversal
            r"\\x[0-9a-fA-F]{2}",  # Hex encoding
        ]

        # Path traversal patterns
        self.path_traversal_patterns = [
            r"\.\./",
            r"\.\\/",
            r"%2e%2e/",
            r"%252e%252e/",
            r"\.\.\\",
        ]

    async def dispatch(self, request: Request, call_next):
        try:
            # Validate request path
            self._validate_path(request.url.path)

            # Validate query parameters
            if request.url.query:
                self._validate_query_params(request.url.query)

            # Validate headers
            self._validate_headers(dict(request.headers))

            # Validate body for POST/PUT/PATCH requests
            if request.method in ["POST", "PUT", "PATCH"]:
                # Read body
                body = await request.body()

                if body:
                    content_type = request.headers.get("content-type", "")

                    if "application/json" in content_type:
                        # Validate JSON body
                        try:
                            json_body = json.loads(body)
                            self._validate_json(json_body)
                            # Sanitize JSON
                            sanitized_body = self._sanitize_json(json_body)
                            # Store sanitized body for downstream use
                            request.state.sanitized_body = sanitized_body
                        except json.JSONDecodeError:
                            raise HTTPException(
                                status_code=400, detail="Invalid JSON format"
                            )

                    elif "application/x-www-form-urlencoded" in content_type:
                        # Validate form data
                        form_data = unquote(body.decode())
                        self._validate_form_data(form_data)

                    elif "multipart/form-data" in content_type:
                        # Validate file uploads
                        self._validate_multipart(body, content_type)

            # Process request
            response = await call_next(request)

            return response

        except HTTPException as e:
            # Re-raise HTTP exceptions
            raise e
        except Exception as e:
            logger.error(f"Input validation error: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid input detected")

    def _validate_path(self, path: str):
        """Validate request path for traversal attacks"""
        # Check for path traversal
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, path, re.IGNORECASE):
                raise HTTPException(status_code=400, detail="Path traversal detected")

        # Check for suspicious file extensions
        for ext in self.block_file_extensions:
            if path.lower().endswith(ext):
                raise HTTPException(
                    status_code=403, detail=f"File type {ext} not allowed"
                )

    def _validate_query_params(self, query: str):
        """Validate query parameters"""
        # URL decode
        decoded_query = unquote(query)

        # Check for SQL injection
        for pattern in self.sql_patterns:
            if re.search(pattern, decoded_query, re.IGNORECASE):
                logger.warning(
                    f"SQL injection attempt detected in query: {query[:100]}"
                )
                raise HTTPException(status_code=400, detail="Invalid query parameters")

        # Check for XSS
        for pattern in self.xss_patterns:
            if re.search(pattern, decoded_query, re.IGNORECASE):
                logger.warning(f"XSS attempt detected in query: {query[:100]}")
                raise HTTPException(status_code=400, detail="Invalid query parameters")

    def _validate_headers(self, headers: Dict[str, str]):
        """Validate request headers"""
        dangerous_headers = [
            "X-Forwarded-Host",  # Can be used for cache poisoning
            "X-Original-URL",  # Can bypass access controls
            "X-Rewrite-URL",  # Can bypass access controls
        ]

        for header, value in headers.items():
            # Check header length
            if len(value) > self.max_field_length:
                raise HTTPException(status_code=431, detail="Request header too large")

            # Check for injection in critical headers
            if header.lower() in ["host", "referer", "origin"]:
                # Validate URL format
                if header.lower() == "host":
                    if not self._is_valid_host(value):
                        raise HTTPException(
                            status_code=400, detail="Invalid host header"
                        )

            # Check for command injection in headers
            for pattern in self.command_patterns:
                if re.search(pattern, value):
                    logger.warning(f"Command injection attempt in header {header}")
                    raise HTTPException(status_code=400, detail="Invalid header value")

    def _validate_json(self, data: Any, depth: int = 0):
        """Recursively validate JSON data"""
        if depth > self.max_json_depth:
            raise HTTPException(status_code=400, detail="JSON nesting too deep")

        if isinstance(data, dict):
            for key, value in data.items():
                # Validate key
                self._validate_string(str(key))
                # Recursively validate value
                self._validate_json(value, depth + 1)

        elif isinstance(data, list):
            for item in data:
                self._validate_json(item, depth + 1)

        elif isinstance(data, str):
            self._validate_string(data)

    def _validate_string(self, value: str):
        """Validate string for injection attacks"""
        if len(value) > self.max_field_length:
            raise HTTPException(status_code=400, detail="Input too long")

        # Check for SQL injection
        for pattern in self.sql_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                # Allow some legitimate SQL keywords in specific fields
                if not self._is_legitimate_sql(value):
                    logger.warning(f"SQL injection attempt: {value[:100]}")
                    raise HTTPException(
                        status_code=400, detail="Invalid input detected"
                    )

        # Check for XSS
        for pattern in self.xss_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"XSS attempt: {value[:100]}")
                raise HTTPException(status_code=400, detail="Invalid input detected")

        # Check for command injection
        for pattern in self.command_patterns:
            if re.search(pattern, value):
                # Allow some legitimate uses (e.g., in descriptions)
                if not self._is_legitimate_command_char(value):
                    logger.warning(f"Command injection attempt: {value[:100]}")
                    raise HTTPException(
                        status_code=400, detail="Invalid input detected"
                    )

    def _validate_form_data(self, form_data: str):
        """Validate form data"""
        # Parse form fields
        fields = form_data.split("&")
        for field in fields:
            if "=" in field:
                key, value = field.split("=", 1)
                self._validate_string(unquote(key))
                self._validate_string(unquote(value))

    def _validate_multipart(self, body: bytes, content_type: str):
        """Validate multipart form data (file uploads)"""
        # Basic validation - in production, use a proper multipart parser
        if len(body) > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(status_code=413, detail="File too large")

    def _sanitize_json(self, data: Any) -> Any:
        """Recursively sanitize JSON data"""
        if isinstance(data, dict):
            return {
                self._sanitize_string(str(k)): self._sanitize_json(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._sanitize_json(item) for item in data]
        elif isinstance(data, str):
            return self._sanitize_string(data)
        else:
            return data

    def _sanitize_string(self, value: str) -> str:
        """Sanitize string input"""
        # Remove null bytes
        value = value.replace("\x00", "")

        # HTML entity encoding for special characters
        value = html.escape(value)

        # Use bleach for additional HTML sanitization
        allowed_tags = ["p", "br", "strong", "em", "u", "a"]
        allowed_attrs = {"a": ["href", "title"]}
        value = bleach.clean(
            value, tags=allowed_tags, attributes=allowed_attrs, strip=True
        )

        # Remove any remaining suspicious patterns
        value = re.sub(r"javascript:", "", value, flags=re.IGNORECASE)
        value = re.sub(r"on\w+\s*=", "", value, flags=re.IGNORECASE)

        return value

    def _is_valid_host(self, host: str) -> bool:
        """Validate host header format"""
        # Remove port if present
        if ":" in host:
            host = host.split(":")[0]

        # Check for valid domain/IP format
        domain_pattern = r"^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?$"
        ip_pattern = r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"

        return bool(re.match(domain_pattern, host) or re.match(ip_pattern, host))

    def _is_legitimate_sql(self, value: str) -> bool:
        """Check if SQL keywords are legitimate (e.g., in a description field)"""
        # Allow SQL keywords in certain contexts
        # This is a simplified check - adjust based on your needs
        legitimate_contexts = [
            "learn sql",
            "sql tutorial",
            "database select",
            "update your",
            "delete account",
        ]

        value_lower = value.lower()
        return any(context in value_lower for context in legitimate_contexts)

    def _is_legitimate_command_char(self, value: str) -> bool:
        """Check if command characters are legitimate"""
        # Allow some command characters in certain contexts
        # e.g., in code examples, descriptions
        legitimate_contexts = [
            "example:",
            "code:",
            "command:",
            "bash",
            "shell",
            "terminal",
        ]

        value_lower = value.lower()
        return any(context in value_lower for context in legitimate_contexts)


class SQLInjectionProtection:
    """
    Additional SQL injection protection utilities
    """

    @staticmethod
    def sanitize_sql_identifier(identifier: str) -> str:
        """Sanitize SQL identifiers (table names, column names)"""
        # Only allow alphanumeric and underscore
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
            raise ValueError("Invalid SQL identifier")
        return identifier

    @staticmethod
    def sanitize_sql_value(value: Any) -> str:
        """Sanitize SQL values for safe interpolation"""
        if value is None:
            return "NULL"
        elif isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            # Escape single quotes
            escaped = value.replace("'", "''")
            return f"'{escaped}'"
        else:
            raise ValueError(f"Unsupported value type: {type(value)}")

    @staticmethod
    def validate_sql_query(query: str) -> bool:
        """Validate SQL query for safety"""
        # Parse SQL to check for dangerous operations
        parsed = sqlparse.parse(query)

        if not parsed:
            return False

        for statement in parsed:
            # Check for multiple statements (prevent stacked queries)
            if len(parsed) > 1:
                return False

            # Check for dangerous keywords
            dangerous_keywords = [
                "DROP",
                "TRUNCATE",
                "DELETE",
                "EXEC",
                "EXECUTE",
                "XP_CMDSHELL",
                "SP_EXECUTE",
                "SHUTDOWN",
                "GRANT",
                "REVOKE",
            ]

            tokens = statement.tokens
            for token in tokens:
                if token.ttype is sqlparse.tokens.Keyword:
                    if token.value.upper() in dangerous_keywords:
                        return False

        return True
