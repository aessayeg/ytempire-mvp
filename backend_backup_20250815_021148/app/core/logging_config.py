"""
Structured logging configuration for YTEmpire
"""
import logging
import logging.config
import json
import sys
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
import traceback

from pythonjsonlogger import jsonlogger
from app.core.config import settings


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging"""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]):
        """Add custom fields to log record"""
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add log level
        log_record['level'] = record.levelname
        
        # Add logger name
        log_record['logger'] = record.name
        
        # Add module and function info
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add environment
        log_record['environment'] = settings.ENVIRONMENT
        
        # Add trace ID if available
        if hasattr(record, 'trace_id'):
            log_record['trace_id'] = record.trace_id
            
        # Add user ID if available
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id
            
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }


class ContextFilter(logging.Filter):
    """Filter to add context information to log records"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context fields to log record"""
        # Add default trace_id if not present
        if not hasattr(record, 'trace_id'):
            record.trace_id = 'no-trace-id'
            
        # Add default user_id if not present
        if not hasattr(record, 'user_id'):
            record.user_id = 'system'
            
        return True


class SensitiveDataFilter(logging.Filter):
    """Filter to remove sensitive data from logs"""
    
    SENSITIVE_FIELDS = [
        'password', 'token', 'api_key', 'secret', 'authorization',
        'credit_card', 'ssn', 'email', 'phone'
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Remove sensitive data from log record"""
        # Check message for sensitive data
        if hasattr(record, 'msg'):
            for field in self.SENSITIVE_FIELDS:
                if field in str(record.msg).lower():
                    # Redact sensitive data
                    record.msg = self._redact_sensitive_data(str(record.msg), field)
                    
        # Check extra fields
        for field in self.SENSITIVE_FIELDS:
            if hasattr(record, field):
                setattr(record, field, '***REDACTED***')
                
        return True
    
    def _redact_sensitive_data(self, message: str, field: str) -> str:
        """Redact sensitive data from message"""
        import re
        # Simple pattern to find key-value pairs
        pattern = rf'{field}["\']?\s*[:=]\s*["\']?([^"\'\s,}}]+)'
        return re.sub(pattern, f'{field}=***REDACTED***', message, flags=re.IGNORECASE)


def setup_logging():
    """Setup structured logging configuration"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'json': {
                '()': CustomJsonFormatter,
                'format': '%(timestamp)s %(level)s %(name)s %(message)s'
            },
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        },
        'filters': {
            'context': {
                '()': ContextFilter
            },
            'sensitive': {
                '()': SensitiveDataFilter
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'json' if settings.ENVIRONMENT == 'production' else 'standard',
                'stream': sys.stdout,
                'filters': ['context', 'sensitive']
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'json',
                'filename': 'logs/ytempire.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'filters': ['context', 'sensitive']
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'json',
                'filename': 'logs/ytempire_errors.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'filters': ['context', 'sensitive']
            },
            'access_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json',
                'filename': 'logs/access.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'filters': ['context']
            },
            'performance_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json',
                'filename': 'logs/performance.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'filters': ['context']
            }
        },
        'loggers': {
            'app': {
                'level': 'DEBUG',
                'handlers': ['console', 'file', 'error_file'],
                'propagate': False
            },
            'app.api': {
                'level': 'INFO',
                'handlers': ['console', 'file', 'access_file'],
                'propagate': False
            },
            'app.services': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'app.ml': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'app.performance': {
                'level': 'INFO',
                'handlers': ['performance_file'],
                'propagate': False
            },
            'uvicorn': {
                'level': 'INFO',
                'handlers': ['console', 'access_file'],
                'propagate': False
            },
            'uvicorn.error': {
                'level': 'ERROR',
                'handlers': ['console', 'error_file'],
                'propagate': False
            },
            'uvicorn.access': {
                'level': 'INFO',
                'handlers': ['access_file'],
                'propagate': False
            },
            'sqlalchemy': {
                'level': 'WARNING',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'sqlalchemy.engine': {
                'level': 'WARNING' if settings.ENVIRONMENT == 'production' else 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console', 'file']
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured successfully",
        extra={
            "environment": settings.ENVIRONMENT,
            "log_level": config['root']['level'],
            "handlers": list(config['handlers'].keys())
        }
    )


class LoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter to add context to all log messages"""
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any] = None):
        super().__init__(logger, extra or {})
        
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message and add context"""
        # Add trace_id from context if available
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
            
        # Merge adapter extra with kwargs extra
        kwargs['extra'].update(self.extra)
        
        return msg, kwargs


def get_logger(name: str, **context) -> LoggerAdapter:
    """Get a logger with context"""
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, context)


class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger('app.performance')
        
    def log_api_performance(
        self,
        endpoint: str,
        method: str,
        response_time_ms: float,
        status_code: int,
        **kwargs
    ):
        """Log API performance metrics"""
        self.logger.info(
            "API Performance",
            extra={
                "metric_type": "api_performance",
                "endpoint": endpoint,
                "method": method,
                "response_time_ms": response_time_ms,
                "status_code": status_code,
                **kwargs
            }
        )
        
    def log_database_performance(
        self,
        operation: str,
        table: str,
        execution_time_ms: float,
        rows_affected: int = 0,
        **kwargs
    ):
        """Log database performance metrics"""
        self.logger.info(
            "Database Performance",
            extra={
                "metric_type": "database_performance",
                "operation": operation,
                "table": table,
                "execution_time_ms": execution_time_ms,
                "rows_affected": rows_affected,
                **kwargs
            }
        )
        
    def log_ml_performance(
        self,
        model: str,
        operation: str,
        processing_time_ms: float,
        input_size: int = 0,
        output_size: int = 0,
        **kwargs
    ):
        """Log ML model performance metrics"""
        self.logger.info(
            "ML Performance",
            extra={
                "metric_type": "ml_performance",
                "model": model,
                "operation": operation,
                "processing_time_ms": processing_time_ms,
                "input_size": input_size,
                "output_size": output_size,
                **kwargs
            }
        )


# Global performance logger instance
performance_logger = PerformanceLogger()