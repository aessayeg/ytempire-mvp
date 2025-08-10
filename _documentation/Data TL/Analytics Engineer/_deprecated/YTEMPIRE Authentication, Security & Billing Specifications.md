# YTEMPIRE Authentication, Security & Billing Specifications

## Document Control
- **Version**: 1.0
- **Date**: January 2025
- **Author**: CTO / Technical Director
- **Audience**: Analytics Engineer, Backend Team, Frontend Team
- **Status**: FINAL - Ready for Implementation

---

## 1. Authentication System Architecture

### 1.1 JWT Implementation Details

```python
class JWTAuthenticationSystem:
    """
    Complete JWT implementation for YTEMPIRE
    """
    
    # Token configuration
    ACCESS_TOKEN_EXPIRE_MINUTES = 15
    REFRESH_TOKEN_EXPIRE_DAYS = 30
    ALGORITHM = "HS256"
    
    # Token payload structure
    TOKEN_PAYLOAD = {
        'user_id': 'UUID',
        'email': 'string',
        'roles': ['user', 'admin'],
        'permissions': ['create_channel', 'view_analytics'],
        'subscription_tier': 'beta',
        'iat': 'timestamp',
        'exp': 'timestamp',
        'jti': 'UUID'  # Unique token ID for revocation
    }
    
    async def create_access_token(self, user_data: dict) -> str:
        """
        Create short-lived access token
        """
        expires_delta = timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)
        expire = datetime.utcnow() + expires_delta
        
        payload = {
            'user_id': str(user_data['user_id']),
            'email': user_data['email'],
            'roles': user_data.get('roles', ['user']),
            'permissions': self.get_user_permissions(user_data),
            'subscription_tier': user_data['subscription_tier'],
            'exp': expire,
            'iat': datetime.utcnow(),
            'jti': str(uuid.uuid4()),
            'type': 'access'
        }
        
        encoded_jwt = jwt.encode(payload, self.SECRET_KEY, algorithm=self.ALGORITHM)
        
        # Store JTI for potential revocation
        await self.redis.setex(
            f"jwt:access:{payload['jti']}", 
            self.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_data['user_id']
        )
        
        return encoded_jwt
    
    async def create_refresh_token(self, user_data: dict) -> str:
        """
        Create long-lived refresh token
        """
        expires_delta = timedelta(days=self.REFRESH_TOKEN_EXPIRE_DAYS)
        expire = datetime.utcnow() + expires_delta
        
        payload = {
            'user_id': str(user_data['user_id']),
            'exp': expire,
            'iat': datetime.utcnow(),
            'jti': str(uuid.uuid4()),
            'type': 'refresh'
        }
        
        encoded_jwt = jwt.encode(payload, self.REFRESH_KEY, algorithm=self.ALGORITHM)
        
        # Store refresh token hash in database
        token_hash = hashlib.sha256(encoded_jwt.encode()).hexdigest()
        await self.store_refresh_token(user_data['user_id'], token_hash, expire)
        
        return encoded_jwt
    
    async def verify_token(self, token: str, token_type: str = 'access') -> dict:
        """
        Verify and decode JWT token
        """
        try:
            # Select appropriate key
            key = self.SECRET_KEY if token_type == 'access' else self.REFRESH_KEY
            
            # Decode token
            payload = jwt.decode(token, key, algorithms=[self.ALGORITHM])
            
            # Verify token type
            if payload.get('type') != token_type:
                raise SecurityError("Invalid token type")
            
            # Check if token is revoked
            if await self.is_token_revoked(payload['jti']):
                raise SecurityError("Token has been revoked")
            
            # Additional security checks for access tokens
            if token_type == 'access':
                # Verify user still active
                if not await self.is_user_active(payload['user_id']):
                    raise SecurityError("User account inactive")
                
                # Verify permissions still valid
                current_permissions = await self.get_user_permissions({'user_id': payload['user_id']})
                if set(payload['permissions']) != set(current_permissions):
                    raise SecurityError("Permissions have changed")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.JWTError:
            raise AuthenticationError("Invalid token")
```

### 1.2 Session Management

```python
class SessionManager:
    """
    Manage user sessions with Redis
    """
    
    SESSION_PREFIX = "session:"
    ACTIVE_SESSIONS_SET = "active_sessions"
    
    async def create_session(self, user_id: str, device_info: dict) -> str:
        """
        Create new user session
        """
        session_id = str(uuid.uuid4())
        
        session_data = {
            'user_id': user_id,
            'session_id': session_id,
            'created_at': datetime.utcnow().isoformat(),
            'last_activity': datetime.utcnow().isoformat(),
            'device_info': device_info,
            'ip_address': device_info.get('ip_address'),
            'user_agent': device_info.get('user_agent')
        }
        
        # Store session in Redis
        await self.redis.hset(
            f"{self.SESSION_PREFIX}{session_id}",
            mapping=session_data
        )
        
        # Set expiration
        await self.redis.expire(
            f"{self.SESSION_PREFIX}{session_id}",
            86400  # 24 hours
        )
        
        # Add to active sessions set
        await self.redis.sadd(self.ACTIVE_SESSIONS_SET, session_id)
        
        # Track user's sessions
        await self.redis.sadd(f"user_sessions:{user_id}", session_id)
        
        return session_id
    
    async def validate_session(self, session_id: str) -> dict:
        """
        Validate and update session
        """
        session_key = f"{self.SESSION_PREFIX}{session_id}"
        
        # Get session data
        session_data = await self.redis.hgetall(session_key)
        
        if not session_data:
            raise SessionError("Invalid session")
        
        # Update last activity
        await self.redis.hset(
            session_key,
            'last_activity',
            datetime.utcnow().isoformat()
        )
        
        # Extend expiration
        await self.redis.expire(session_key, 86400)
        
        return session_data
    
    async def terminate_session(self, session_id: str):
        """
        Terminate user session
        """
        session_key = f"{self.SESSION_PREFIX}{session_id}"
        
        # Get session data for user_id
        session_data = await self.redis.hgetall(session_key)
        
        if session_data:
            user_id = session_data['user_id']
            
            # Remove from user's sessions
            await self.redis.srem(f"user_sessions:{user_id}", session_id)
        
        # Remove session
        await self.redis.delete(session_key)
        await self.redis.srem(self.ACTIVE_SESSIONS_SET, session_id)
```

### 1.3 Multi-Factor Authentication (MFA)

```python
class MFASystem:
    """
    Multi-factor authentication implementation
    """
    
    async def setup_totp(self, user_id: str) -> dict:
        """
        Set up TOTP (Time-based One-Time Password) for user
        """
        # Generate secret
        secret = pyotp.random_base32()
        
        # Store encrypted secret
        encrypted_secret = self.encrypt(secret)
        await self.db.execute(
            "UPDATE users SET totp_secret = $1, mfa_enabled = true WHERE id = $2",
            encrypted_secret, user_id
        )
        
        # Generate QR code
        user = await self.get_user(user_id)
        provisioning_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user['email'],
            issuer_name='YTEMPIRE'
        )
        
        qr_code = qrcode.make(provisioning_uri)
        qr_buffer = io.BytesIO()
        qr_code.save(qr_buffer, format='PNG')
        
        return {
            'secret': secret,
            'qr_code': base64.b64encode(qr_buffer.getvalue()).decode(),
            'backup_codes': await self.generate_backup_codes(user_id)
        }
    
    async def verify_totp(self, user_id: str, token: str) -> bool:
        """
        Verify TOTP token
        """
        # Get encrypted secret
        result = await self.db.fetchone(
            "SELECT totp_secret FROM users WHERE id = $1 AND mfa_enabled = true",
            user_id
        )
        
        if not result:
            return False
        
        # Decrypt secret
        secret = self.decrypt(result['totp_secret'])
        
        # Verify token
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)  # Allow 30 second window
    
    async def generate_backup_codes(self, user_id: str) -> list:
        """
        Generate backup codes for account recovery
        """
        codes = []
        hashed_codes = []
        
        for _ in range(10):
            code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            codes.append(code)
            hashed_codes.append(hashlib.sha256(code.encode()).hexdigest())
        
        # Store hashed codes
        await self.db.execute(
            "UPDATE users SET backup_codes = $1 WHERE id = $2",
            json.dumps(hashed_codes), user_id
        )
        
        return codes

---

## 2. Security Implementation

### 2.1 API Key Management

```python
class APIKeyManager:
    """
    Secure API key generation and management
    """
    
    KEY_PREFIX = "yte_"  # YTEMPIRE prefix
    KEY_LENGTH = 32
    
    async def generate_api_key(self, user_id: str, key_name: str) -> dict:
        """
        Generate new API key for user
        """
        # Generate cryptographically secure key
        key_bytes = secrets.token_bytes(self.KEY_LENGTH)
        key_string = base64.urlsafe_b64encode(key_bytes).decode('utf-8').rstrip('=')
        api_key = f"{self.KEY_PREFIX}{key_string}"
        
        # Hash key for storage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Store key metadata
        key_data = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'key_hash': key_hash,
            'key_prefix': api_key[:12] + '...',  # Store prefix for identification
            'name': key_name,
            'permissions': await self.get_default_permissions(user_id),
            'rate_limit': await self.get_rate_limit(user_id),
            'created_at': datetime.utcnow(),
            'last_used': None,
            'expires_at': datetime.utcnow() + timedelta(days=365),
            'status': 'active'
        }
        
        await self.db.execute(
            """
            INSERT INTO api_keys (id, user_id, key_hash, key_prefix, name, 
                                 permissions, rate_limit, created_at, expires_at, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            key_data['id'], key_data['user_id'], key_data['key_hash'],
            key_data['key_prefix'], key_data['name'], json.dumps(key_data['permissions']),
            key_data['rate_limit'], key_data['created_at'], key_data['expires_at'],
            key_data['status']
        )
        
        # Return key only once
        return {
            'api_key': api_key,
            'key_id': key_data['id'],
            'expires_at': key_data['expires_at'].isoformat(),
            'warning': 'Store this key securely. It will not be shown again.'
        }
    
    async def validate_api_key(self, api_key: str) -> dict:
        """
        Validate API key and return permissions
        """
        # Check format
        if not api_key.startswith(self.KEY_PREFIX):
            raise SecurityError("Invalid API key format")
        
        # Hash the key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Look up key
        result = await self.db.fetchone(
            """
            SELECT k.*, u.subscription_tier, u.status as user_status
            FROM api_keys k
            JOIN users u ON k.user_id = u.id
            WHERE k.key_hash = $1 AND k.status = 'active'
            """,
            key_hash
        )
        
        if not result:
            raise SecurityError("Invalid API key")
        
        # Check expiration
        if result['expires_at'] < datetime.utcnow():
            raise SecurityError("API key expired")
        
        # Check user status
        if result['user_status'] != 'active':
            raise SecurityError("User account inactive")
        
        # Update last used
        await self.db.execute(
            "UPDATE api_keys SET last_used = $1, usage_count = usage_count + 1 WHERE id = $2",
            datetime.utcnow(), result['id']
        )
        
        return {
            'user_id': result['user_id'],
            'key_id': result['id'],
            'permissions': json.loads(result['permissions']),
            'rate_limit': result['rate_limit'],
            'subscription_tier': result['subscription_tier']
        }
    
    async def rotate_api_key(self, user_id: str, old_key_id: str) -> dict:
        """
        Rotate API key with grace period
        """
        # Get old key details
        old_key = await self.db.fetchone(
            "SELECT * FROM api_keys WHERE id = $1 AND user_id = $2",
            old_key_id, user_id
        )
        
        if not old_key:
            raise SecurityError("Key not found")
        
        # Generate new key
        new_key = await self.generate_api_key(user_id, old_key['name'] + ' (rotated)')
        
        # Set old key to expire in 7 days
        await self.db.execute(
            """
            UPDATE api_keys 
            SET status = 'rotating', 
                expires_at = $1,
                rotation_key_id = $2
            WHERE id = $3
            """,
            datetime.utcnow() + timedelta(days=7),
            new_key['key_id'],
            old_key_id
        )
        
        return new_key
```

### 2.2 Row-Level Security Implementation

```python
class RowLevelSecurity:
    """
    PostgreSQL RLS implementation for multi-tenant security
    """
    
    async def setup_rls(self):
        """
        Set up row-level security policies
        """
        # Enable RLS on all user data tables
        tables = ['channels', 'videos', 'analytics', 'billing_records']
        
        for table in tables:
            await self.db.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        
        # Create security policies
        policies = {
            'channels': """
                CREATE POLICY tenant_isolation ON channels
                FOR ALL
                USING (user_id = current_setting('app.current_user')::uuid)
                WITH CHECK (user_id = current_setting('app.current_user')::uuid)
            """,
            
            'videos': """
                CREATE POLICY tenant_isolation ON videos
                FOR ALL
                USING (
                    channel_id IN (
                        SELECT id FROM channels 
                        WHERE user_id = current_setting('app.current_user')::uuid
                    )
                )
            """,
            
            'analytics': """
                CREATE POLICY tenant_isolation ON analytics
                FOR SELECT
                USING (
                    channel_id IN (
                        SELECT id FROM channels 
                        WHERE user_id = current_setting('app.current_user')::uuid
                    )
                )
            """,
            
            'billing_records': """
                CREATE POLICY tenant_isolation ON billing_records
                FOR ALL
                USING (user_id = current_setting('app.current_user')::uuid)
            """
        }
        
        for table, policy in policies.items():
            await self.db.execute(policy)
    
    async def set_user_context(self, connection, user_id: str):
        """
        Set user context for RLS
        """
        await connection.execute(
            "SET LOCAL app.current_user TO $1",
            user_id
        )
```

### 2.3 Encryption and Key Storage

```python
class EncryptionService:
    """
    Handle encryption for sensitive data
    """
    
    def __init__(self):
        # Load master key from environment
        self.master_key = base64.b64decode(os.environ['MASTER_ENCRYPTION_KEY'])
        self.cipher_suite = Fernet(self.master_key)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """
        Encrypt sensitive data like API keys, tokens
        """
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data
        """
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    async def encrypt_user_data(self, user_id: str, data_type: str, data: dict) -> str:
        """
        Encrypt user-specific data with derived key
        """
        # Derive user-specific key
        user_key = self.derive_user_key(user_id, data_type)
        user_cipher = Fernet(user_key)
        
        # Encrypt data
        json_data = json.dumps(data)
        encrypted = user_cipher.encrypt(json_data.encode())
        
        return base64.b64encode(encrypted).decode()
    
    def derive_user_key(self, user_id: str, data_type: str) -> bytes:
        """
        Derive user-specific encryption key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=f"{user_id}:{data_type}".encode(),
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(self.master_key))
```

---

## 3. Billing and Subscription Management

### 3.1 Stripe Integration

```python
class StripeIntegration:
    """
    Complete Stripe billing integration
    """
    
    def __init__(self):
        stripe.api_key = os.environ['STRIPE_SECRET_KEY']
        self.webhook_secret = os.environ['STRIPE_WEBHOOK_SECRET']
    
    # Subscription plans
    SUBSCRIPTION_PLANS = {
        'beta': {
            'price_id': 'price_1234567890abcdef',
            'name': 'Beta Plan',
            'price': 0,  # Free during beta
            'channels': 5,
            'videos_per_day': 10,
            'features': ['basic_analytics', 'email_support']
        },
        'starter': {
            'price_id': 'price_starter123456',
            'name': 'Starter Plan',
            'price': 297,  # $297/month
            'channels': 5,
            'videos_per_day': 10,
            'features': ['advanced_analytics', 'priority_support', 'api_access']
        },
        'growth': {
            'price_id': 'price_growth123456',
            'name': 'Growth Plan',
            'price': 997,  # $997/month
            'channels': 25,
            'videos_per_day': 50,
            'features': ['all_starter_features', 'custom_voices', 'white_label']
        },
        'enterprise': {
            'price_id': 'price_enterprise123',
            'name': 'Enterprise Plan',
            'price': 2997,  # $2,997/month
            'channels': 100,
            'videos_per_day': 200,
            'features': ['all_features', 'dedicated_support', 'sla']
        }
    }
    
    async def create_customer(self, user_data: dict) -> str:
        """
        Create Stripe customer for new user
        """
        customer = stripe.Customer.create(
            email=user_data['email'],
            name=user_data.get('name'),
            metadata={
                'user_id': str(user_data['user_id']),
                'signup_date': datetime.utcnow().isoformat()
            }
        )
        
        # Store Stripe customer ID
        await self.db.execute(
            "UPDATE users SET stripe_customer_id = $1 WHERE id = $2",
            customer.id, user_data['user_id']
        )
        
        return customer.id
    
    async def create_subscription(self, user_id: str, plan_key: str) -> dict:
        """
        Create subscription for user
        """
        # Get user's Stripe customer ID
        user = await self.get_user(user_id)
        customer_id = user['stripe_customer_id']
        
        if not customer_id:
            customer_id = await self.create_customer({'user_id': user_id, 'email': user['email']})
        
        # Get plan details
        plan = self.SUBSCRIPTION_PLANS[plan_key]
        
        # Create subscription
        subscription = stripe.Subscription.create(
            customer=customer_id,
            items=[{'price': plan['price_id']}],
            payment_behavior='default_incomplete',
            payment_settings={'save_default_payment_method': 'on_subscription'},
            expand=['latest_invoice.payment_intent'],
            metadata={
                'user_id': str(user_id),
                'plan_key': plan_key
            }
        )
        
        # Store subscription details
        await self.store_subscription(user_id, subscription)
        
        return {
            'subscription_id': subscription.id,
            'client_secret': subscription.latest_invoice.payment_intent.client_secret,
            'status': subscription.status
        }
    
    async def handle_webhook(self, payload: bytes, sig_header: str) -> dict:
        """
        Handle Stripe webhooks
        """
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, self.webhook_secret
            )
        except ValueError:
            raise WebhookError("Invalid payload")
        except stripe.error.SignatureVerificationError:
            raise WebhookError("Invalid signature")
        
        # Handle different event types
        handlers = {
            'customer.subscription.created': self.handle_subscription_created,
            'customer.subscription.updated': self.handle_subscription_updated,
            'customer.subscription.deleted': self.handle_subscription_deleted,
            'invoice.payment_succeeded': self.handle_payment_succeeded,
            'invoice.payment_failed': self.handle_payment_failed,
            'customer.subscription.trial_will_end': self.handle_trial_ending
        }
        
        handler = handlers.get(event['type'])
        if handler:
            await handler(event['data']['object'])
        
        return {'status': 'success', 'event_type': event['type']}
```

### 3.2 Usage-Based Billing

```python
class UsageBillingSystem:
    """
    Track and bill for usage beyond plan limits
    """
    
    OVERAGE_RATES = {
        'extra_video': 0.50,  # $0.50 per video over limit
        'extra_channel': 50.00,  # $50/month per extra channel
        'premium_voice': 0.10,  # $0.10 extra per premium voice
        'rush_processing': 1.00  # $1.00 per rush video
    }
    
    async def track_usage(self, user_id: str, usage_type: str, quantity: int = 1):
        """
        Track usage for billing
        """
        # Get current billing period
        billing_period = await self.get_current_billing_period(user_id)
        
        # Record usage
        await self.db.execute(
            """
            INSERT INTO usage_records (user_id, billing_period_id, usage_type, 
                                      quantity, unit_price, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (user_id, billing_period_id, usage_type, date_trunc('day', timestamp))
            DO UPDATE SET quantity = usage_records.quantity + $4
            """,
            user_id, billing_period['id'], usage_type, quantity,
            self.OVERAGE_RATES.get(usage_type, 0), datetime.utcnow()
        )
        
        # Check if user is approaching limits
        await self.check_usage_alerts(user_id, billing_period)
    
    async def calculate_current_usage(self, user_id: str) -> dict:
        """
        Calculate current usage and costs
        """
        billing_period = await self.get_current_billing_period(user_id)
        
        # Get base plan
        plan = await self.get_user_plan(user_id)
        
        # Calculate usage
        usage = await self.db.fetch(
            """
            SELECT 
                usage_type,
                SUM(quantity) as total_quantity,
                SUM(quantity * unit_price) as total_cost
            FROM usage_records
            WHERE user_id = $1 AND billing_period_id = $2
            GROUP BY usage_type
            """,
            user_id, billing_period['id']
        )
        
        # Calculate totals
        total_videos = 0
        total_overage_cost = 0
        
        for record in usage:
            if record['usage_type'] == 'video_generated':
                total_videos = record['total_quantity']
                # Calculate overage
                overage = max(0, total_videos - plan['videos_per_day'] * 30)
                if overage > 0:
                    total_overage_cost += overage * self.OVERAGE_RATES['extra_video']
            else:
                total_overage_cost += record['total_cost']
        
        return {
            'billing_period': billing_period,
            'plan': plan,
            'usage': {
                'videos_used': total_videos,
                'videos_limit': plan['videos_per_day'] * 30,
                'channels_used': await self.count_active_channels(user_id),
                'channels_limit': plan['channels']
            },
            'overage_cost': total_overage_cost,
            'total_cost': plan['price'] + total_overage_cost
        }
```

### 3.3 Invoice Generation

```python
class InvoiceSystem:
    """
    Generate and manage invoices
    """
    
    async def generate_invoice(self, user_id: str, billing_period_id: str) -> dict:
        """
        Generate invoice for billing period
        """
        # Get user and plan details
        user = await self.get_user(user_id)
        plan = await self.get_user_plan(user_id)
        usage = await self.calculate_period_usage(user_id, billing_period_id)
        
        # Create invoice record
        invoice = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'billing_period_id': billing_period_id,
            'invoice_number': await self.generate_invoice_number(),
            'issue_date': datetime.utcnow(),
            'due_date': datetime.utcnow() + timedelta(days=7),
            'status': 'draft',
            'line_items': []
        }
        
        # Add base subscription
        invoice['line_items'].append({
            'description': f"{plan['name']} - Monthly Subscription",
            'quantity': 1,
            'unit_price': plan['price'],
            'total': plan['price']
        })
        
        # Add usage-based items
        if usage['video_overage'] > 0:
            invoice['line_items'].append({
                'description': f"Additional Videos ({usage['video_overage']} videos)",
                'quantity': usage['video_overage'],
                'unit_price': self.OVERAGE_RATES['extra_video'],
                'total': usage['video_overage'] * self.OVERAGE_RATES['extra_video']
            })
        
        # Calculate totals
        subtotal = sum(item['total'] for item in invoice['line_items'])
        tax = subtotal * 0.0  # Configure based on location
        
        invoice['subtotal'] = subtotal
        invoice['tax'] = tax
        invoice['total'] = subtotal + tax
        
        # Store invoice
        await self.store_invoice(invoice)
        
        # Generate PDF
        pdf_url = await self.generate_invoice_pdf(invoice)
        invoice['pdf_url'] = pdf_url
        
        return invoice
```

---

## 4. Security Headers and Middleware

### 4.1 Security Headers Configuration

```python
class SecurityMiddleware:
    """
    Apply security headers to all responses
    """
    
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' https://js.stripe.com; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' https://api.stripe.com https://api.youtube.com; frame-src https://js.stripe.com",
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'camera=(), microphone=(), geolocation=()'
    }
    
    async def __call__(self, request: Request, call_next):
        # Process request
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.SECURITY_HEADERS.items():
            response.headers[header] = value
        
        # Remove sensitive headers
        response.headers.pop('X-Powered-By', None)
        response.headers.pop('Server', None)
        
        return response
```

### 4.2 Rate Limiting Implementation

```python
class RateLimiter:
    """
    Implement rate limiting per user/IP
    """
    
    RATE_LIMITS = {
        'api': {'requests': 1000, 'window': 3600},  # 1000 per hour
        'auth': {'requests': 10, 'window': 300},    # 10 per 5 minutes
        'video_generation': {'requests': 50, 'window': 3600},  # 50 per hour
        'webhook': {'requests': 100, 'window': 60}   # 100 per minute
    }
    
    async def check_rate_limit(self, identifier: str, endpoint_type: str) -> bool:
        """
        Check if request is within rate limits
        """
        limits = self.RATE_LIMITS.get(endpoint_type, self.RATE_LIMITS['api'])
        key = f"rate_limit:{endpoint_type}:{identifier}"
        
        # Get current count
        current = await self.redis.incr(key)
        
        # Set expiry on first request
        if current == 1:
            await self.redis.expire(key, limits['window'])
        
        # Check limit
        if current > limits['requests']:
            ttl = await self.redis.ttl(key)
            raise RateLimitError(
                f"Rate limit exceeded. Try again in {ttl} seconds.",
                retry_after=ttl
            )
        
        return True
```

---

## 5. Audit Logging and Compliance

### 5.1 Comprehensive Audit System

```python
class AuditLogger:
    """
    Log all security-relevant events
    """
    
    AUDIT_EVENTS = {
        'auth.login': 'User login attempt',
        'auth.logout': 'User logout',
        'auth.failed': 'Failed authentication',
        'api.key_created': 'API key created',
        'api.key_used': 'API key used',
        'billing.subscription_created': 'Subscription created',
        'billing.payment_failed': 'Payment failed',
        'security.permission_denied': 'Permission denied',
        'data.export': 'Data exported',
        'data.deleted': 'Data deleted'
    }
    
    async def log_event(self, event_type: str, context: dict):
        """
        Log audit event
        """
        event = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow(),
            'event_type': event_type,
            'user_id': context.get('user_id'),
            'ip_address': context.get('ip_address'),
            'user_agent': context.get('user_agent'),
            'resource': context.get('resource'),
            'action': context.get('action'),
            'result': context.get('result', 'success'),
            'metadata': context.get('metadata', {})
        }
        
        # Store in database
        await self.db.execute(
            """
            INSERT INTO audit_logs (id, timestamp, event_type, user_id, 
                                   ip_address, user_agent, resource, 
                                   action, result, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            event['id'], event['timestamp'], event['event_type'],
            event['user_id'], event['ip_address'], event['user_agent'],
            event['resource'], event['action'], event['result'],
            json.dumps(event['metadata'])
        )
        
        # Send to SIEM if configured
        if self.siem_enabled:
            await self.send_to_siem(event)
```

---

## 6. Integration Points for Analytics

### 6.1 Security Metrics Schema

```sql
-- Security metrics for analytics dashboards
CREATE TABLE security_metrics (
    timestamp TIMESTAMP NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    value NUMERIC NOT NULL,
    metadata JSONB,
    PRIMARY KEY (timestamp, metric_type)
);

-- Create hypertable for time-series data
SELECT create_hypertable('security_metrics', 'timestamp');

-- Sample metrics to track
INSERT INTO security_metrics (timestamp, metric_type, value, metadata) VALUES
(NOW(), 'auth.success_rate', 0.98, '{"period": "hour"}'),
(NOW(), 'api.rate_limit_hits', 45, '{"endpoint": "/api/videos"}'),
(NOW(), 'security.blocked_requests', 12, '{"reason": "invalid_token"}');
```

### 6.2 Billing Analytics Schema

```sql
-- Billing analytics tables
CREATE TABLE billing_analytics (
    user_id UUID NOT NULL,
    date DATE NOT NULL,
    plan_name VARCHAR(50),
    base_revenue DECIMAL(10,2),
    overage_revenue DECIMAL(10,2),
    total_revenue DECIMAL(10,2),
    video_count INTEGER,
    channel_count INTEGER,
    api_calls INTEGER,
    PRIMARY KEY (user_id, date)
);

-- Revenue aggregation view
CREATE VIEW daily_revenue_summary AS
SELECT 
    date,
    COUNT(DISTINCT user_id) as active_users,
    SUM(base_revenue) as base_revenue,
    SUM(overage_revenue) as overage_revenue,
    SUM(total_revenue) as total_revenue,
    AVG(total_revenue) as arpu,
    SUM(video_count) as total_videos
FROM billing_analytics
GROUP BY date;
```

---

## Next Steps for Analytics Engineer

1. **Implement authentication tracking**:
   - Session duration metrics
   - Login success/failure rates
   - MFA adoption tracking

2. **Set up billing dashboards**:
   - MRR/ARR tracking
   - Churn analysis
   - Usage vs. limits visualization
   - Payment failure monitoring

3. **Create security monitoring**:
   - Failed authentication attempts
   - API key usage patterns
   - Rate limit violations
   - Suspicious activity detection

4. **Build compliance reports**:
   - Audit log summaries
   - Data access reports
   - User permission audits

This document provides comprehensive authentication, security, and billing specifications needed for the MVP implementation.