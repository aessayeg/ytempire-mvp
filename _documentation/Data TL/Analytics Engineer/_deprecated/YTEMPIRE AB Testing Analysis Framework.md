# YTEMPIRE A/B Testing Analysis Framework

## Document Control
- **Version**: 1.0
- **Last Updated**: January 2025
- **Owner**: Analytics Engineering Team
- **Audience**: Analytics Engineers, Data Scientists, Product Managers

---

## 1. Executive Summary

This document provides a comprehensive framework for designing, implementing, analyzing, and reporting A/B tests at YTEMPIRE. It covers statistical methodologies, implementation patterns, analysis queries, and best practices for running experiments at scale across our content empire.

---

## 2. A/B Testing Framework Overview

### 2.1 Testing Philosophy

At YTEMPIRE, we approach A/B testing with the following principles:

1. **Data-Driven Decisions**: Every significant change should be tested
2. **Statistical Rigor**: Proper sample sizes and significance testing
3. **Business Impact Focus**: Tests should map to business KPIs
4. **Rapid Iteration**: Quick test cycles with automated analysis
5. **Learning Culture**: Document and share all test results

### 2.2 Test Categories

```yaml
test_categories:
  content_optimization:
    - title_variations
    - thumbnail_designs
    - video_length
    - content_style
    - description_formats
    
  algorithm_optimization:
    - recommendation_models
    - trend_detection_algorithms
    - pricing_strategies
    - personalization_rules
    
  user_experience:
    - ui_layouts
    - feature_rollouts
    - onboarding_flows
    - notification_strategies
    
  monetization:
    - ad_placement
    - sponsorship_formats
    - membership_tiers
    - pricing_models
```

---

## 3. Statistical Foundation

### 3.1 Sample Size Calculation

```python
import numpy as np
from scipy import stats
import math

class SampleSizeCalculator:
    """
    Calculate required sample size for A/B tests
    """
    
    def __init__(self):
        self.confidence_levels = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        
    def calculate_sample_size(self, 
                            baseline_rate, 
                            minimum_detectable_effect, 
                            confidence_level=0.95,
                            power=0.80,
                            test_type='two-sided'):
        """
        Calculate required sample size per variant
        
        Args:
            baseline_rate: Current conversion rate (0-1)
            minimum_detectable_effect: Relative change to detect (e.g., 0.05 for 5%)
            confidence_level: Statistical confidence (typically 0.95)
            power: Statistical power (typically 0.80)
            test_type: 'two-sided' or 'one-sided'
        
        Returns:
            Required sample size per variant
        """
        
        # Get z-scores
        z_alpha = self.confidence_levels[confidence_level]
        z_beta = stats.norm.ppf(power)
        
        # Adjust for test type
        if test_type == 'two-sided':
            z_alpha = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        
        # Calculate effect size
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)
        
        # Pooled probability
        p_pool = (p1 + p2) / 2
        
        # Sample size formula
        n = (2 * p_pool * (1 - p_pool) * (z_alpha + z_beta)**2) / (p1 - p2)**2
        
        return math.ceil(n)
    
    def calculate_test_duration(self, 
                               required_sample_size,
                               daily_traffic,
                               variants_count=2,
                               allocation_percent=100):
        """
        Calculate how long the test needs to run
        
        Args:
            required_sample_size: Sample size per variant
            daily_traffic: Average daily traffic
            variants_count: Number of test variants (including control)
            allocation_percent: Percentage of traffic in test (0-100)
        
        Returns:
            Required test duration in days
        """
        
        traffic_per_variant = (daily_traffic * allocation_percent / 100) / variants_count
        days_required = required_sample_size / traffic_per_variant
        
        return math.ceil(days_required)
    
    def validate_sample_size(self, 
                           current_sample_size,
                           required_sample_size,
                           current_days,
                           projected_days):
        """
        Validate if test has sufficient sample size
        
        Returns:
            Dictionary with validation status and recommendations
        """
        
        progress_percent = (current_sample_size / required_sample_size) * 100
        
        return {
            'is_valid': current_sample_size >= required_sample_size,
            'progress_percent': progress_percent,
            'samples_needed': max(0, required_sample_size - current_sample_size),
            'days_remaining': max(0, projected_days - current_days),
            'recommendation': self._get_recommendation(progress_percent, current_days)
        }
    
    def _get_recommendation(self, progress_percent, days_elapsed):
        """Generate recommendation based on test progress"""
        
        if progress_percent >= 100:
            return "Test has sufficient data - ready for analysis"
        elif progress_percent >= 80:
            return f"Almost ready - {100 - progress_percent:.0f}% more data needed"
        elif progress_percent >= 50 and days_elapsed < 3:
            return "On track - continue running test"
        elif progress_percent < 50 and days_elapsed > 7:
            return "Consider increasing traffic allocation or extending test"
        else:
            return "Continue collecting data"

# Example usage
calculator = SampleSizeCalculator()

# Thumbnail CTR test
baseline_ctr = 0.05  # 5% CTR
mde = 0.10  # Want to detect 10% relative improvement
sample_size = calculator.calculate_sample_size(baseline_ctr, mde)
print(f"Required sample size per variant: {sample_size}")

# Calculate duration
daily_impressions = 50000
test_duration = calculator.calculate_test_duration(sample_size, daily_impressions)
print(f"Test should run for {test_duration} days")
```

### 3.2 Statistical Significance Testing

```python
class SignificanceTester:
    """
    Perform statistical significance tests for A/B experiments
    """
    
    def __init__(self):
        self.test_results = {}
        
    def chi_square_test(self, control_data, variant_data):
        """
        Chi-square test for conversion rate differences
        
        Args:
            control_data: Dict with 'successes' and 'trials'
            variant_data: Dict with 'successes' and 'trials'
        
        Returns:
            Test results dictionary
        """
        
        # Create contingency table
        observed = np.array([
            [control_data['successes'], control_data['trials'] - control_data['successes']],
            [variant_data['successes'], variant_data['trials'] - variant_data['successes']]
        ])
        
        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)
        
        # Calculate conversion rates
        control_rate = control_data['successes'] / control_data['trials']
        variant_rate = variant_data['successes'] / variant_data['trials']
        
        # Calculate lift
        absolute_lift = variant_rate - control_rate
        relative_lift = (variant_rate - control_rate) / control_rate
        
        # Calculate confidence interval for lift
        se_diff = self._calculate_standard_error(control_data, variant_data)
        ci_lower = absolute_lift - 1.96 * se_diff
        ci_upper = absolute_lift + 1.96 * se_diff
        
        return {
            'control_rate': control_rate,
            'variant_rate': variant_rate,
            'absolute_lift': absolute_lift,
            'relative_lift': relative_lift,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'confidence_interval': {
                'lower': ci_lower,
                'upper': ci_upper
            },
            'chi_square': chi2,
            'test_type': 'chi_square'
        }
    
    def t_test_continuous(self, control_values, variant_values):
        """
        T-test for continuous metrics (e.g., watch time, revenue)
        
        Args:
            control_values: Array of control group values
            variant_values: Array of variant group values
        
        Returns:
            Test results dictionary
        """
        
        # Perform Welch's t-test (doesn't assume equal variances)
        t_stat, p_value = stats.ttest_ind(
            variant_values, 
            control_values, 
            equal_var=False
        )
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.std(control_values)**2 + np.std(variant_values)**2) / 2
        )
        cohens_d = (np.mean(variant_values) - np.mean(control_values)) / pooled_std
        
        # Calculate confidence interval
        diff_mean = np.mean(variant_values) - np.mean(control_values)
        se_diff = np.sqrt(
            np.var(control_values)/len(control_values) + 
            np.var(variant_values)/len(variant_values)
        )
        ci_lower = diff_mean - 1.96 * se_diff
        ci_upper = diff_mean + 1.96 * se_diff
        
        return {
            'control_mean': np.mean(control_values),
            'variant_mean': np.mean(variant_values),
            'absolute_difference': diff_mean,
            'relative_difference': diff_mean / np.mean(control_values),
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'confidence_interval': {
                'lower': ci_lower,
                'upper': ci_upper
            },
            't_statistic': t_stat,
            'cohens_d': cohens_d,
            'effect_size': self._interpret_cohens_d(cohens_d),
            'test_type': 't_test'
        }
    
    def sequential_testing(self, data_stream, alpha_spending_function='obrien_fleming'):
        """
        Sequential testing for early stopping
        
        Args:
            data_stream: Stream of test data
            alpha_spending_function: Method for alpha spending
        
        Returns:
            Sequential test results
        """
        
        # Implement group sequential testing
        # This is a simplified version - production would use gsDesign package
        
        interim_results = []
        cumulative_alpha_spent = 0
        
        for i, data_point in enumerate(data_stream):
            # Calculate current alpha to spend
            if alpha_spending_function == 'obrien_fleming':
                alpha_current = 0.05 * (2 * (1 - stats.norm.cdf(
                    stats.norm.ppf(1 - 0.025) / np.sqrt(i + 1)
                )))
            else:  # Pocock
                alpha_current = 0.05 / len(data_stream)
            
            # Perform test at this interim point
            test_result = self.chi_square_test(
                data_point['control'],
                data_point['variant']
            )
            
            interim_results.append({
                'interim_point': i + 1,
                'sample_size': data_point['control']['trials'],
                'p_value': test_result['p_value'],
                'alpha_threshold': alpha_current,
                'decision': 'stop' if test_result['p_value'] < alpha_current else 'continue',
                'cumulative_alpha_spent': cumulative_alpha_spent + alpha_current
            })
            
            if test_result['p_value'] < alpha_current:
                break
        
        return interim_results
    
    def _calculate_standard_error(self, control_data, variant_data):
        """Calculate standard error for difference in proportions"""
        
        p1 = control_data['successes'] / control_data['trials']
        p2 = variant_data['successes'] / variant_data['trials']
        n1 = control_data['trials']
        n2 = variant_data['trials']
        
        return np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    
    def _interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size"""
        
        if abs(d) < 0.2:
            return 'negligible'
        elif abs(d) < 0.5:
            return 'small'
        elif abs(d) < 0.8:
            return 'medium'
        else:
            return 'large'
```

### 3.3 Multiple Testing Correction

```python
class MultipleTestingCorrection:
    """
    Handle multiple comparisons in A/B testing
    """
    
    def __init__(self):
        self.correction_methods = [
            'bonferroni',
            'holm',
            'benjamini_hochberg',
            'none'
        ]
    
    def correct_p_values(self, p_values, method='benjamini_hochberg', alpha=0.05):
        """
        Apply multiple testing correction
        
        Args:
            p_values: List of p-values from multiple tests
            method: Correction method to use
            alpha: Significance level
        
        Returns:
            Corrected p-values and rejection decisions
        """
        
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]
        
        if method == 'bonferroni':
            adjusted_p = np.minimum(sorted_p * n, 1.0)
            reject = adjusted_p < alpha
            
        elif method == 'holm':
            adjusted_p = []
            for i, p in enumerate(sorted_p):
                adjusted_p.append(min(p * (n - i), 1.0))
            adjusted_p = np.maximum.accumulate(adjusted_p)
            reject = adjusted_p < alpha
            
        elif method == 'benjamini_hochberg':
            adjusted_p = []
            for i, p in enumerate(sorted_p):
                adjusted_p.append(min(p * n / (i + 1), 1.0))
            # Ensure monotonicity
            for i in range(n - 2, -1, -1):
                adjusted_p[i] = min(adjusted_p[i], adjusted_p[i + 1])
            reject = np.array(adjusted_p) < alpha
            
        else:  # No correction
            adjusted_p = sorted_p
            reject = sorted_p < alpha
        
        # Reorder to match original
        original_order = np.argsort(sorted_indices)
        
        return {
            'original_p_values': p_values,
            'adjusted_p_values': np.array(adjusted_p)[original_order],
            'reject_null': reject[original_order],
            'method': method,
            'alpha': alpha,
            'num_rejections': np.sum(reject)
        }
```

---

## 4. Test Implementation Framework

### 4.1 Database Schema for A/B Tests

```sql
-- A/B Test Configuration
CREATE TABLE ab_tests (
    test_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    test_name VARCHAR(255) NOT NULL,
    test_type VARCHAR(50) NOT NULL, -- 'split', 'multivariate', 'bandit'
    status VARCHAR(50) DEFAULT 'draft', -- 'draft', 'running', 'completed', 'stopped'
    
    -- Test details
    hypothesis TEXT,
    primary_metric VARCHAR(100) NOT NULL,
    secondary_metrics JSONB,
    
    -- Configuration
    traffic_allocation DECIMAL(5,2) DEFAULT 100.0, -- Percentage of traffic
    min_sample_size INTEGER,
    max_duration_days INTEGER DEFAULT 30,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    
    -- Results
    winner_variant VARCHAR(50),
    decision_notes TEXT,
    
    -- Metadata
    created_by VARCHAR(100),
    tags TEXT[]
);

-- Test Variants
CREATE TABLE ab_test_variants (
    variant_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    test_id UUID REFERENCES ab_tests(test_id),
    variant_name VARCHAR(50) NOT NULL,
    variant_type VARCHAR(20) DEFAULT 'treatment', -- 'control', 'treatment'
    
    -- Configuration
    traffic_split DECIMAL(5,2), -- Percentage for this variant
    configuration JSONB, -- Variant-specific config
    
    -- Tracking
    is_active BOOLEAN DEFAULT true,
    
    UNIQUE(test_id, variant_name)
);

-- User Assignment
CREATE TABLE ab_test_assignments (
    assignment_id BIGSERIAL PRIMARY KEY,
    test_id UUID REFERENCES ab_tests(test_id),
    variant_id UUID REFERENCES ab_test_variants(variant_id),
    
    -- Entity being tested
    entity_type VARCHAR(50), -- 'video', 'channel', 'user'
    entity_id VARCHAR(100),
    
    -- Assignment details
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    assignment_reason VARCHAR(50), -- 'random', 'forced', 'excluded'
    
    -- Indexes for performance
    INDEX idx_test_entity (test_id, entity_type, entity_id),
    INDEX idx_assigned_at (assigned_at)
);

-- Test Metrics
CREATE TABLE ab_test_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    test_id UUID REFERENCES ab_tests(test_id),
    variant_id UUID REFERENCES ab_test_variants(variant_id),
    
    -- Metric data
    metric_name VARCHAR(100),
    metric_value DECIMAL(20,6),
    metric_count INTEGER, -- For rate metrics
    metric_denominator INTEGER, -- For rate metrics
    
    -- Aggregation
    aggregation_period VARCHAR(20), -- 'hourly', 'daily', 'cumulative'
    period_start TIMESTAMP,
    period_end TIMESTAMP,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_test_variant_metric (test_id, variant_id, metric_name),
    INDEX idx_period (period_start, period_end)
);

-- Test Events (for custom analysis)
CREATE TABLE ab_test_events (
    event_id BIGSERIAL PRIMARY KEY,
    test_id UUID REFERENCES ab_tests(test_id),
    variant_id UUID REFERENCES ab_test_variants(variant_id),
    
    -- Event details
    entity_type VARCHAR(50),
    entity_id VARCHAR(100),
    event_type VARCHAR(100),
    event_value JSONB,
    event_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Partitioning by date for performance
    INDEX idx_test_event_time (test_id, event_timestamp)
) PARTITION BY RANGE (event_timestamp);

-- Create monthly partitions
CREATE TABLE ab_test_events_2025_01 PARTITION OF ab_test_events
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
```

### 4.2 Assignment Logic

```python
import hashlib
import random
from typing import Dict, Optional

class ABTestAssigner:
    """
    Handle user/entity assignment to test variants
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.assignment_cache = {}
    
    def assign_to_test(self, 
                      test_id: str,
                      entity_type: str,
                      entity_id: str,
                      force_variant: Optional[str] = None) -> Dict:
        """
        Assign an entity to a test variant
        
        Args:
            test_id: UUID of the test
            entity_type: Type of entity (video, channel, user)
            entity_id: ID of the entity
            force_variant: Force assignment to specific variant
        
        Returns:
            Assignment details
        """
        
        # Check cache first
        cache_key = f"{test_id}:{entity_type}:{entity_id}"
        if cache_key in self.assignment_cache:
            return self.assignment_cache[cache_key]
        
        # Check existing assignment
        existing = self._get_existing_assignment(test_id, entity_type, entity_id)
        if existing:
            self.assignment_cache[cache_key] = existing
            return existing
        
        # Get test configuration
        test_config = self._get_test_config(test_id)
        if not test_config or test_config['status'] != 'running':
            return {'assigned': False, 'reason': 'test_not_active'}
        
        # Check traffic allocation
        if not self._should_include_in_test(test_config['traffic_allocation']):
            return {'assigned': False, 'reason': 'traffic_excluded'}
        
        # Assign to variant
        if force_variant:
            variant = self._get_variant_by_name(test_id, force_variant)
            assignment_reason = 'forced'
        else:
            variant = self._select_variant(test_id, entity_id, test_config['variants'])
            assignment_reason = 'random'
        
        # Store assignment
        assignment = self._store_assignment(
            test_id, variant['variant_id'],
            entity_type, entity_id,
            assignment_reason
        )
        
        self.assignment_cache[cache_key] = assignment
        return assignment
    
    def _select_variant(self, test_id: str, entity_id: str, variants: list) -> Dict:
        """
        Select variant using deterministic hashing
        """
        
        # Create stable hash
        hash_input = f"{test_id}:{entity_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Map to percentile (0-100)
        percentile = (hash_value % 10000) / 100
        
        # Select variant based on traffic split
        cumulative = 0
        for variant in variants:
            cumulative += variant['traffic_split']
            if percentile < cumulative:
                return variant
        
        # Fallback to last variant
        return variants[-1]
    
    def _should_include_in_test(self, traffic_allocation: float) -> bool:
        """
        Determine if entity should be included based on traffic allocation
        """
        return random.random() * 100 < traffic_allocation
    
    def get_variant_for_entity(self, test_id: str, entity_type: str, entity_id: str) -> Optional[str]:
        """
        Get assigned variant for an entity
        """
        
        assignment = self.assign_to_test(test_id, entity_type, entity_id)
        return assignment.get('variant_name') if assignment.get('assigned') else None
```

### 4.3 Test Configuration Management

```sql
-- Function to create a new A/B test
CREATE OR REPLACE FUNCTION create_ab_test(
    p_test_name VARCHAR,
    p_hypothesis TEXT,
    p_primary_metric VARCHAR,
    p_variants JSONB,
    p_created_by VARCHAR
) RETURNS UUID AS $
DECLARE
    v_test_id UUID;
    v_variant JSONB;
    v_total_split DECIMAL := 0;
BEGIN
    -- Validate traffic splits sum to 100
    FOR v_variant IN SELECT * FROM jsonb_array_elements(p_variants)
    LOOP
        v_total_split := v_total_split + (v_variant->>'traffic_split')::DECIMAL;
    END LOOP;
    
    IF v_total_split != 100 THEN
        RAISE EXCEPTION 'Traffic splits must sum to 100%%, got %%', v_total_split;
    END IF;
    
    -- Create test
    INSERT INTO ab_tests (
        test_name, hypothesis, primary_metric, created_by
    ) VALUES (
        p_test_name, p_hypothesis, p_primary_metric, p_created_by
    ) RETURNING test_id INTO v_test_id;
    
    -- Create variants
    FOR v_variant IN SELECT * FROM jsonb_array_elements(p_variants)
    LOOP
        INSERT INTO ab_test_variants (
            test_id, 
            variant_name, 
            variant_type,
            traffic_split,
            configuration
        ) VALUES (
            v_test_id,
            v_variant->>'name',
            v_variant->>'type',
            (v_variant->>'traffic_split')::DECIMAL,
            v_variant->'config'
        );
    END LOOP;
    
    RETURN v_test_id;
END;
$ LANGUAGE plpgsql;

-- Function to start a test
CREATE OR REPLACE FUNCTION start_ab_test(p_test_id UUID) RETURNS BOOLEAN AS $
BEGIN
    -- Validate test is ready
    IF NOT EXISTS (
        SELECT 1 FROM ab_tests 
        WHERE test_id = p_test_id 
        AND status = 'draft'
    ) THEN
        RAISE EXCEPTION 'Test not in draft status';
    END IF;
    
    -- Update status
    UPDATE ab_tests 
    SET status = 'running', started_at = CURRENT_TIMESTAMP
    WHERE test_id = p_test_id;
    
    -- Log event
    INSERT INTO ab_test_events (test_id, event_type, event_value)
    VALUES (p_test_id, 'test_started', jsonb_build_object('timestamp', CURRENT_TIMESTAMP));
    
    RETURN TRUE;
END;
$ LANGUAGE plpgsql;
```

---

## 5. Analysis Queries

### 5.1 Test Performance Summary

```sql
-- Comprehensive test performance analysis
CREATE OR REPLACE VIEW v_ab_test_performance AS
WITH variant_metrics AS (
    SELECT 
        t.test_id,
        t.test_name,
        t.primary_metric,
        v.variant_id,
        v.variant_name,
        v.variant_type,
        
        -- Aggregate metrics
        COUNT(DISTINCT a.entity_id) AS sample_size,
        
        -- For conversion metrics
        SUM(CASE WHEN m.metric_name = t.primary_metric THEN m.metric_count ELSE 0 END) AS conversions,
        SUM(CASE WHEN m.metric_name = t.primary_metric THEN m.metric_denominator ELSE 0 END) AS trials,
        
        -- For continuous metrics
        AVG(CASE WHEN m.metric_name = t.primary_metric THEN m.metric_value END) AS avg_value,
        STDDEV(CASE WHEN m.metric_name = t.primary_metric THEN m.metric_value END) AS std_value,
        
        -- Time in test
        MIN(a.assigned_at) AS first_assignment,
        MAX(a.assigned_at) AS last_assignment,
        EXTRACT(EPOCH FROM (MAX(a.assigned_at) - MIN(a.assigned_at))) / 86400 AS test_duration_days
        
    FROM ab_tests t
    JOIN ab_test_variants v ON t.test_id = v.test_id
    LEFT JOIN ab_test_assignments a ON v.variant_id = a.variant_id
    LEFT JOIN ab_test_metrics m ON v.variant_id = m.variant_id
    WHERE t.status IN ('running', 'completed')
    GROUP BY t.test_id, t.test_name, t.primary_metric, 
             v.variant_id, v.variant_name, v.variant_type
),
statistical_analysis AS (
    SELECT 
        test_id,
        test_name,
        primary_metric,
        
        -- Control metrics
        MAX(CASE WHEN variant_type = 'control' THEN sample_size END) AS control_sample_size,
        MAX(CASE WHEN variant_type = 'control' THEN conversions END) AS control_conversions,
        MAX(CASE WHEN variant_type = 'control' THEN trials END) AS control_trials,
        MAX(CASE WHEN variant_type = 'control' THEN avg_value END) AS control_avg_value,
        
        -- Treatment metrics (assuming single treatment for simplicity)
        MAX(CASE WHEN variant_type = 'treatment' THEN sample_size END) AS treatment_sample_size,
        MAX(CASE WHEN variant_type = 'treatment' THEN conversions END) AS treatment_conversions,
        MAX(CASE WHEN variant_type = 'treatment' THEN trials END) AS treatment_trials,
        MAX(CASE WHEN variant_type = 'treatment' THEN avg_value END) AS treatment_avg_value,
        
        -- Test duration
        MAX(test_duration_days) AS test_duration_days
        
    FROM variant_metrics
    GROUP BY test_id, test_name, primary_metric
)
SELECT 
    test_id,
    test_name,
    primary_metric,
    
    -- Sample sizes
    control_sample_size,
    treatment_sample_size,
    control_sample_size + treatment_sample_size AS total_sample_size,
    
    -- Conversion rates (for binary metrics)
    CASE 
        WHEN control_trials > 0 THEN control_conversions::FLOAT / control_trials 
        ELSE NULL 
    END AS control_conversion_rate,
    
    CASE 
        WHEN treatment_trials > 0 THEN treatment_conversions::FLOAT / treatment_trials 
        ELSE NULL 
    END AS treatment_conversion_rate,
    
    -- Lift calculation
    CASE 
        WHEN control_trials > 0 AND treatment_trials > 0 THEN
            ((treatment_conversions::FLOAT / treatment_trials) - 
             (control_conversions::FLOAT / control_trials)) / 
            (control_conversions::FLOAT / control_trials) * 100
        ELSE NULL 
    END AS relative_lift_percent,
    
    -- Continuous metric differences
    treatment_avg_value - control_avg_value AS absolute_difference,
    CASE 
        WHEN control_avg_value > 0 THEN 
            (treatment_avg_value - control_avg_value) / control_avg_value * 100
        ELSE NULL 
    END AS relative_difference_percent,
    
    -- Test duration
    test_duration_days,
    
    -- Statistical significance (simplified - would use proper function in production)
    CASE 
        WHEN control_trials > 30 AND treatment_trials > 30 THEN
            CASE 
                WHEN ABS(
                    (treatment_conversions::FLOAT / treatment_trials) - 
                    (control_conversions::FLOAT / control_trials)
                ) > 1.96 * SQRT(
                    (control_conversions::FLOAT / control_trials) * 
                    (1 - control_conversions::FLOAT / control_trials) / control_trials +
                    (treatment_conversions::FLOAT / treatment_trials) * 
                    (1 - treatment_conversions::FLOAT / treatment_trials) / treatment_trials
                ) THEN 'Significant'
                ELSE 'Not Significant'
            END
        ELSE 'Insufficient Data'
    END AS significance_status
    
FROM statistical_analysis;
```

### 5.2 Time Series Analysis

```sql
-- Time series view of test performance
CREATE OR REPLACE FUNCTION analyze_test_time_series(
    p_test_id UUID,
    p_interval VARCHAR DEFAULT 'daily'
) RETURNS TABLE (
    period_start TIMESTAMP,
    variant_name VARCHAR,
    sample_size INTEGER,
    conversion_rate DECIMAL,
    cumulative_conversions INTEGER,
    cumulative_trials INTEGER,
    cumulative_conversion_rate DECIMAL,
    confidence_lower DECIMAL,
    confidence_upper DECIMAL
) AS $
BEGIN
    RETURN QUERY
    WITH time_periods AS (
        SELECT 
            DATE_TRUNC(p_interval, a.assigned_at) AS period_start,
            v.variant_name,
            v.variant_id,
            COUNT(DISTINCT a.entity_id) AS period_sample_size
        FROM ab_test_assignments a
        JOIN ab_test_variants v ON a.variant_id = v.variant_id
        WHERE a.test_id = p_test_id
        GROUP BY DATE_TRUNC(p_interval, a.assigned_at), v.variant_name, v.variant_id
    ),
    period_metrics AS (
        SELECT 
            tp.period_start,
            tp.variant_name,
            tp.variant_id,
            tp.period_sample_size,
            COALESCE(SUM(m.metric_count), 0) AS conversions,
            COALESCE(SUM(m.metric_denominator), 0) AS trials,
            CASE 
                WHEN COALESCE(SUM(m.metric_denominator), 0) > 0 
                THEN SUM(m.metric_count)::FLOAT / SUM(m.metric_denominator)
                ELSE 0 
            END AS conversion_rate
        FROM time_periods tp
        LEFT JOIN ab_test_metrics m ON 
            tp.variant_id = m.variant_id AND
            m.period_start >= tp.period_start AND
            m.period_start < tp.period_start + 
                CASE p_interval 
                    WHEN 'hourly' THEN INTERVAL '1 hour'
                    WHEN 'daily' THEN INTERVAL '1 day'
                    WHEN 'weekly' THEN INTERVAL '1 week'
                END
        GROUP BY tp.period_start, tp.variant_name, tp.variant_id, tp.period_sample_size
    ),
    cumulative_metrics AS (
        SELECT 
            period_start,
            variant_name,
            period_sample_size,
            conversion_rate,
            SUM(conversions) OVER (PARTITION BY variant_name ORDER BY period_start) AS cumulative_conversions,
            SUM(trials) OVER (PARTITION BY variant_name ORDER BY period_start) AS cumulative_trials,
            SUM(conversions) OVER (PARTITION BY variant_name ORDER BY period_start)::FLOAT / 
                NULLIF(SUM(trials) OVER (PARTITION BY variant_name ORDER BY period_start), 0) AS cumulative_conversion_rate
        FROM period_metrics
    )
    SELECT 
        period_start,
        variant_name,
        period_sample_size AS sample_size,
        conversion_rate::DECIMAL(10,4),
        cumulative_conversions::INTEGER,
        cumulative_trials::INTEGER,
        cumulative_conversion_rate::DECIMAL(10,4),
        -- Wilson confidence interval
        (cumulative_conversion_rate - 1.96 * SQRT(
            cumulative_conversion_rate * (1 - cumulative_conversion_rate) / NULLIF(cumulative_trials, 0)
        ))::DECIMAL(10,4) AS confidence_lower,
        (cumulative_conversion_rate + 1.96 * SQRT(
            cumulative_conversion_rate * (1 - cumulative_conversion_rate) / NULLIF(cumulative_trials, 0)
        ))::DECIMAL(10,4) AS confidence_upper
    FROM cumulative_metrics
    ORDER BY period_start, variant_name;
END;
$ LANGUAGE plpgsql;
```

### 5.3 Segment Analysis

```sql
-- Analyze test performance by segments
CREATE OR REPLACE FUNCTION analyze_test_segments(
    p_test_id UUID,
    p_segment_type VARCHAR -- 'channel', 'category', 'audience', etc.
) RETURNS TABLE (
    segment_value VARCHAR,
    variant_name VARCHAR,
    sample_size INTEGER,
    conversion_rate DECIMAL,
    relative_lift DECIMAL,
    p_value DECIMAL,
    is_significant BOOLEAN
) AS $
BEGIN
    RETURN QUERY
    WITH segment_data AS (
        SELECT 
            CASE p_segment_type
                WHEN 'channel' THEN c.niche
                WHEN 'category' THEN v.category
                WHEN 'audience' THEN v.audience_segment
                ELSE 'unknown'
            END AS segment_value,
            av.variant_name,
            av.variant_type,
            COUNT(DISTINCT a.entity_id) AS sample_size,
            SUM(m.metric_count) AS conversions,
            SUM(m.metric_denominator) AS trials
        FROM ab_test_assignments a
        JOIN ab_test_variants av ON a.variant_id = av.variant_id
        LEFT JOIN ab_test_metrics m ON a.variant_id = m.variant_id
        LEFT JOIN videos v ON a.entity_id = v.video_id AND a.entity_type = 'video'
        LEFT JOIN channels c ON v.channel_id = c.channel_id
        WHERE a.test_id = p_test_id
        GROUP BY 1, av.variant_name, av.variant_type
    ),
    segment_analysis AS (
        SELECT 
            segment_value,
            variant_name,
            sample_size,
            conversions::FLOAT / NULLIF(trials, 0) AS conversion_rate,
            
            -- Calculate lift vs control
            CASE 
                WHEN variant_type = 'treatment' THEN
                    (conversions::FLOAT / NULLIF(trials, 0) - 
                     LAG(conversions::FLOAT / NULLIF(trials, 0)) OVER (PARTITION BY segment_value ORDER BY variant_type)) /
                    NULLIF(LAG(conversions::FLOAT / NULLIF(trials, 0)) OVER (PARTITION BY segment_value ORDER BY variant_type), 0)
                ELSE NULL
            END AS relative_lift,
            
            -- Simplified p-value calculation (would use proper stats function)
            CASE 
                WHEN trials > 30 THEN
                    2 * (1 - normal_cdf(ABS(
                        (conversions::FLOAT / trials - 
                         LAG(conversions::FLOAT / trials) OVER (PARTITION BY segment_value ORDER BY variant_type)) /
                        SQRT(
                            (conversions::FLOAT / trials * (1 - conversions::FLOAT / trials) / trials) +
                            (LAG(conversions::FLOAT / trials) OVER (PARTITION BY segment_value ORDER BY variant_type) * 
                             (1 - LAG(conversions::FLOAT / trials) OVER (PARTITION BY segment_value ORDER BY variant_type)) /
                             LAG(trials) OVER (PARTITION BY segment_value ORDER BY variant_type))
                        )
                    )))
                ELSE NULL
            END AS p_value
            
        FROM segment_data
    )
    SELECT 
        segment_value,
        variant_name,
        sample_size,
        conversion_rate::DECIMAL(10,4),
        relative_lift::DECIMAL(10,4),
        p_value::DECIMAL(10,6),
        p_value < 0.05 AS is_significant
    FROM segment_analysis
    WHERE segment_value IS NOT NULL
    ORDER BY segment_value, variant_name;
END;
$ LANGUAGE plpgsql;

-- Helper function for normal CDF (simplified)
CREATE OR REPLACE FUNCTION normal_cdf(z FLOAT) RETURNS FLOAT AS $
BEGIN
    -- Simplified approximation of normal CDF
    RETURN 0.5 * (1 + erf(z / SQRT(2)));
END;
$ LANGUAGE plpgsql;
```

---

## 6. Reporting Framework

### 6.1 Test Results Report Generation

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

class ABTestReporter:
    """
    Generate comprehensive A/B test reports
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.report_components = []
        
    def generate_test_report(self, test_id: str, output_format='html'):
        """
        Generate comprehensive test report
        
        Args:
            test_id: UUID of the test
            output_format: 'html', 'pdf', or 'json'
        
        Returns:
            Report in specified format
        """
        
        # Gather all report components
        test_info = self._get_test_info(test_id)
        overall_results = self._get_overall_results(test_id)
        time_series = self._get_time_series_analysis(test_id)
        segment_analysis = self._get_segment_analysis(test_id)
        
        # Statistical analysis
        significance_test = self._perform_significance_test(test_id)
        power_analysis = self._perform_power_analysis(test_id)
        
        # Generate visualizations
        visualizations = self._create_visualizations(
            test_id, overall_results, time_series
        )
        
        # Compile report
        report = {
            'test_info': test_info,
            'executive_summary': self._create_executive_summary(
                test_info, overall_results, significance_test
            ),
            'overall_results': overall_results,
            'statistical_analysis': significance_test,
            'power_analysis': power_analysis,
            'time_series_analysis': time_series,
            'segment_analysis': segment_analysis,
            'visualizations': visualizations,
            'recommendations': self._generate_recommendations(
                overall_results, significance_test, segment_analysis
            ),
            'technical_appendix': self._create_technical_appendix(test_id)
        }
        
        # Format output
        if output_format == 'html':
            return self._format_as_html(report)
        elif output_format == 'pdf':
            return self._format_as_pdf(report)
        else:
            return report
    
    def _create_executive_summary(self, test_info, results, significance):
        """Create executive summary of test results"""
        
        summary = {
            'test_name': test_info['test_name'],
            'test_duration': f"{test_info['duration_days']} days",
            'total_sample_size': results['total_sample_size'],
            
            'winner': 'Treatment' if results['relative_lift'] > 0 and significance['is_significant'] else 'No Clear Winner',
            'lift': f"{results['relative_lift']:.1f}%",
            'confidence': f"{(1 - significance['p_value']) * 100:.1f}%",
            
            'business_impact': self._calculate_business_impact(results),
            
            'recommendation': self._get_recommendation(results, significance)
        }
        
        return summary
    
    def _create_visualizations(self, test_id, overall_results, time_series):
        """Create test result visualizations"""
        
        visualizations = {}
        
        # 1. Overall Results Bar Chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        variants = ['Control', 'Treatment']
        rates = [
            overall_results['control_conversion_rate'],
            overall_results['treatment_conversion_rate']
        ]
        
        bars = ax.bar(variants, rates)
        ax.set_ylabel('Conversion Rate')
        ax.set_title('A/B Test Results: Conversion Rate by Variant')
        
        # Add value labels
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.2%}', ha='center', va='bottom')
        
        visualizations['overall_results'] = self._fig_to_base64(fig)
        plt.close()
        
        # 2. Time Series Plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        for variant in time_series['variant_name'].unique():
            variant_data = time_series[time_series['variant_name'] == variant]
            ax.plot(
                variant_data['period_start'],
                variant_data['cumulative_conversion_rate'],
                label=variant,
                marker='o'
            )
            
            # Add confidence intervals
            ax.fill_between(
                variant_data['period_start'],
                variant_data['confidence_lower'],
                variant_data['confidence_upper'],
                alpha=0.2
            )
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Conversion Rate')
        ax.set_title('Conversion Rate Over Time')
        ax.legend()
        
        visualizations['time_series'] = self._fig_to_base64(fig)
        plt.close()
        
        # 3. Statistical Power Curve
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        sample_sizes = range(100, 10000, 100)
        powers = [self._calculate_power_at_sample_size(n, overall_results) for n in sample_sizes]
        
        ax.plot(sample_sizes, powers)
        ax.axhline(y=0.8, color='r', linestyle='--', label='Target Power (80%)')
        ax.set_xlabel('Sample Size per Variant')
        ax.set_ylabel('Statistical Power')
        ax.set_title('Statistical Power Analysis')
        ax.legend()
        
        visualizations['power_curve'] = self._fig_to_base64(fig)
        plt.close()
        
        return visualizations
    
    def _generate_recommendations(self, results, significance, segments):
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Primary recommendation
        if significance['is_significant'] and results['relative_lift'] > 0:
            recommendations.append({
                'priority': 'high',
                'action': 'implement_winner',
                'description': f"Implement the treatment variant across all channels. Expected lift: {results['relative_lift']:.1f}%",
                'estimated_impact': self._calculate_business_impact(results)
            })
        elif not significance['is_significant'] and results['sample_size'] < results['required_sample_size']:
            recommendations.append({
                'priority': 'medium',
                'action': 'continue_test',
                'description': f"Continue test for {results['additional_days_needed']} more days to reach statistical significance",
                'estimated_impact': 'Conclusive results'
            })
        
        # Segment-specific recommendations
        if segments:
            high_performing_segments = [
                s for s in segments 
                if s['is_significant'] and s['relative_lift'] > results['relative_lift']
            ]
            
            if high_performing_segments:
                recommendations.append({
                    'priority': 'medium',
                    'action': 'targeted_rollout',
                    'description': f"Consider targeted rollout to high-performing segments: {', '.join([s['segment'] for s in high_performing_segments])}",
                    'estimated_impact': 'Higher lift in specific segments'
                })
        
        return recommendations
```

### 6.2 Automated Test Monitoring

```sql
-- Automated test monitoring and alerting
CREATE OR REPLACE FUNCTION monitor_ab_tests() RETURNS TABLE (
    test_id UUID,
    test_name VARCHAR,
    alert_type VARCHAR,
    alert_message TEXT,
    alert_severity VARCHAR
) AS $
BEGIN
    RETURN QUERY
    
    -- Check for tests reaching sample size
    SELECT 
        t.test_id,
        t.test_name,
        'sample_size_reached'::VARCHAR AS alert_type,
        format('Test has reached required sample size (%s). Ready for analysis.', t.min_sample_size) AS alert_message,
        'info'::VARCHAR AS alert_severity
    FROM ab_tests t
    JOIN (
        SELECT test_id, COUNT(DISTINCT entity_id) AS current_sample_size
        FROM ab_test_assignments
        GROUP BY test_id
    ) a ON t.test_id = a.test_id
    WHERE t.status = 'running'
        AND a.current_sample_size >= t.min_sample_size
    
    UNION ALL
    
    -- Check for tests running too long
    SELECT 
        test_id,
        test_name,
        'max_duration_exceeded'::VARCHAR,
        format('Test has been running for %s days (max: %s)', 
               EXTRACT(DAY FROM CURRENT_TIMESTAMP - started_at),
               max_duration_days
        ),
        'warning'::VARCHAR
    FROM ab_tests
    WHERE status = 'running'
        AND CURRENT_TIMESTAMP > started_at + (max_duration_days || ' days')::INTERVAL
    
    UNION ALL
    
    -- Check for sample ratio mismatch
    SELECT 
        t.test_id,
        t.test_name,
        'sample_ratio_mismatch'::VARCHAR,
        format('Sample ratio mismatch detected. Expected: %s, Actual: %s',
               STRING_AGG(v.traffic_split::TEXT, ':' ORDER BY v.variant_name),
               STRING_AGG(ROUND(100.0 * COUNT(a.assignment_id) / SUM(COUNT(a.assignment_id)) OVER (PARTITION BY t.test_id), 1)::TEXT, ':' ORDER BY v.variant_name)
        ),
        'critical'::VARCHAR
    FROM ab_tests t
    JOIN ab_test_variants v ON t.test_id = v.test_id
    LEFT JOIN ab_test_assignments a ON v.variant_id = a.variant_id
    WHERE t.status = 'running'
    GROUP BY t.test_id, t.test_name, v.variant_name, v.traffic_split
    HAVING ABS(v.traffic_split - (100.0 * COUNT(a.assignment_id) / SUM(COUNT(a.assignment_id)) OVER (PARTITION BY t.test_id))) > 5;
    
END;
$ LANGUAGE plpgsql;

-- Schedule monitoring
SELECT cron.schedule('monitor-ab-tests', '*/30 * * * *', 'SELECT monitor_ab_tests()');
```

---

## 7. Best Practices and Guidelines

### 7.1 Test Design Checklist

```yaml
test_design_checklist:
  before_testing:
    - [ ] Clear hypothesis defined
    - [ ] Primary metric identified
    - [ ] Success criteria established
    - [ ] Sample size calculated
    - [ ] Test duration estimated
    - [ ] Segments identified for analysis
    - [ ] Potential confounders considered
    
  implementation:
    - [ ] Random assignment verified
    - [ ] Control and treatment properly isolated
    - [ ] Tracking implemented correctly
    - [ ] No data leakage between variants
    - [ ] Assignment logging working
    
  during_test:
    - [ ] Monitor sample ratio mismatch
    - [ ] Check for technical issues
    - [ ] Avoid peeking at results
    - [ ] Document any incidents
    - [ ] Maintain test integrity
    
  analysis:
    - [ ] Wait for full sample size
    - [ ] Check statistical assumptions
    - [ ] Perform segment analysis
    - [ ] Consider practical significance
    - [ ] Document learnings
    
  post_test:
    - [ ] Make implementation decision
    - [ ] Plan rollout strategy
    - [ ] Set up monitoring
    - [ ] Share results broadly
    - [ ] Archive test data
```

### 7.2 Common Pitfalls and Solutions

| Pitfall | Impact | Solution |
|---------|---------|----------|
| **Peeking** | Inflated false positive rate | Use sequential testing methods |
| **Multiple testing** | Increased Type I error | Apply correction methods |
| **Selection bias** | Invalid results | Ensure proper randomization |
| **Novelty effects** | Overestimated impact | Run tests longer, analyze cohorts |
| **Interaction effects** | Confounded results | Avoid running overlapping tests |
| **Sample ratio mismatch** | Biased assignment | Monitor and investigate causes |
| **Ignoring segments** | Missed insights | Always perform segment analysis |

### 7.3 Test Velocity Guidelines

```python
class TestVelocityOptimizer:
    """
    Optimize test velocity while maintaining quality
    """
    
    def calculate_test_capacity(self, 
                              total_traffic,
                              min_test_size,
                              test_duration_days,
                              overlap_allowed=0.2):
        """
        Calculate how many tests can run simultaneously
        
        Args:
            total_traffic: Daily traffic available
            min_test_size: Minimum sample size per test
            test_duration_days: Typical test duration
            overlap_allowed: Fraction of traffic that can be in multiple tests
        
        Returns:
            Maximum number of concurrent tests
        """
        
        # Traffic needed per test per day
        traffic_per_test_day = min_test_size / test_duration_days
        
        # Account for overlap
        effective_traffic = total_traffic * (1 + overlap_allowed)
        
        # Maximum concurrent tests
        max_concurrent = int(effective_traffic / traffic_per_test_day)
        
        return {
            'max_concurrent_tests': max_concurrent,
            'traffic_per_test': traffic_per_test_day,
            'utilization_rate': (max_concurrent * traffic_per_test_day) / total_traffic,
            'recommendations': self._get_velocity_recommendations(max_concurrent)
        }
    
    def _get_velocity_recommendations(self, max_concurrent):
        """Provide recommendations for test velocity"""
        
        if max_concurrent < 3:
            return "Consider reducing test sample sizes or duration to increase velocity"
        elif max_concurrent > 10:
            return "High test velocity - ensure proper test isolation and tracking"
        else:
            return "Optimal test velocity - maintain current approach"
```

---

## 8. Integration with Business Metrics

### 8.1 Business Impact Calculation

```sql
-- Calculate business impact of test results
CREATE OR REPLACE FUNCTION calculate_test_business_impact(
    p_test_id UUID
) RETURNS TABLE (
    metric_name VARCHAR,
    control_value DECIMAL,
    treatment_value DECIMAL,
    absolute_impact DECIMAL,
    relative_impact DECIMAL,
    annualized_impact DECIMAL,
    confidence_interval JSONB
) AS $
WITH test_metrics AS (
    SELECT 
        m.metric_name,
        v.variant_type,
        AVG(m.metric_value) AS avg_value,
        COUNT(*) AS sample_size,
        STDDEV(m.metric_value) AS std_value
    FROM ab_test_metrics m
    JOIN ab_test_variants v ON m.variant_id = v.variant_id
    WHERE m.test_id = p_test_id
    GROUP BY m.metric_name, v.variant_type
),
impact_calculation AS (
    SELECT 
        metric_name,
        MAX(CASE WHEN variant_type = 'control' THEN avg_value END) AS control_value,
        MAX(CASE WHEN variant_type = 'treatment' THEN avg_value END) AS treatment_value,
        MAX(CASE WHEN variant_type = 'control' THEN sample_size END) AS control_n,
        MAX(CASE WHEN variant_type = 'treatment' THEN sample_size END) AS treatment_n,
        MAX(CASE WHEN variant_type = 'control' THEN std_value END) AS control_std,
        MAX(CASE WHEN variant_type = 'treatment' THEN std_value END) AS treatment_std
    FROM test_metrics
    GROUP BY metric_name
)
SELECT 
    metric_name,
    control_value,
    treatment_value,
    treatment_value - control_value AS absolute_impact,
    (treatment_value - control_value) / NULLIF(control_value, 0) * 100 AS relative_impact,
    
    -- Annualized impact (assuming metric is daily)
    (treatment_value - control_value) * 365 AS annualized_impact,
    
    -- Confidence interval
    jsonb_build_object(
        'lower', (treatment_value - control_value) - 1.96 * SQRT(
            POWER(control_std, 2) / control_n + POWER(treatment_std, 2) / treatment_n
        ),
        'upper', (treatment_value - control_value) + 1.96 * SQRT(
            POWER(control_std, 2) / control_n + POWER(treatment_std, 2) / treatment_n
        )
    ) AS confidence_interval
    
FROM impact_calculation;
$ LANGUAGE sql;
```

### 8.2 ROI Analysis

```python
def calculate_test_roi(test_results, implementation_cost, time_horizon_days=365):
    """
    Calculate ROI of implementing test winner
    
    Args:
        test_results: Dictionary with test results
        implementation_cost: Cost to implement the change
        time_horizon_days: Days to calculate ROI over
    
    Returns:
        ROI analysis
    """
    
    # Calculate incremental value
    daily_incremental_value = (
        test_results['treatment_revenue'] - 
        test_results['control_revenue']
    ) * test_results['daily_scale_factor']
    
    total_incremental_value = daily_incremental_value * time_horizon_days
    
    # Calculate ROI
    roi = (total_incremental_value - implementation_cost) / implementation_cost
    
    # Payback period
    if daily_incremental_value > 0:
        payback_days = implementation_cost / daily_incremental_value
    else:
        payback_days = float('inf')
    
    return {
        'roi_percentage': roi * 100,
        'payback_period_days': payback_days,
        'total_value': total_incremental_value,
        'implementation_cost': implementation_cost,
        'break_even_date': datetime.now() + timedelta(days=payback_days),
        'recommendation': 'Implement' if roi > 0.2 else 'Do not implement'
    }
```

---

## Appendices

### Appendix A: Statistical Tables

| Confidence Level | Z-Score | T-Score (df=30) |
|-----------------|---------|-----------------|
| 90% | 1.645 | 1.697 |
| 95% | 1.96 | 2.042 |
| 99% | 2.576 | 2.750 |

### Appendix B: Sample Size Quick Reference

| Baseline Rate | MDE | Sample Size (per variant) |
|--------------|-----|--------------------------|
| 1% | 50% | 3,842 |
| 5% | 20% | 3,832 |
| 10% | 15% | 3,998 |
| 20% | 10% | 6,138 |

### Appendix C: Test Analysis SQL Functions

```sql
-- Collection of useful SQL functions for A/B testing

-- Wilson Score Confidence Interval
CREATE OR REPLACE FUNCTION wilson_ci(
    successes INTEGER,
    trials INTEGER,
    confidence FLOAT DEFAULT 0.95
) RETURNS JSONB AS $
DECLARE
    p FLOAT;
    z FLOAT;
    denominator FLOAT;
    centre FLOAT;
    offset FLOAT;
BEGIN
    IF trials = 0 THEN
        RETURN jsonb_build_object('lower', 0, 'upper', 0);
    END IF;
    
    p := successes::FLOAT / trials;
    z := CASE 
        WHEN confidence = 0.90 THEN 1.645
        WHEN confidence = 0.95 THEN 1.96
        WHEN confidence = 0.99 THEN 2.576
        ELSE 1.96
    END;
    
    denominator := 1 + z*z/trials;
    centre := (p + z*z/(2*trials)) / denominator;
    offset := z * SQRT(p*(1-p)/trials + z*z/(4*trials*trials)) / denominator;
    
    RETURN jsonb_build_object(
        'lower', GREATEST(0, centre - offset),
        'upper', LEAST(1, centre + offset)
    );
END;
$ LANGUAGE plpgsql;

-- Bayesian A/B Test Analysis
CREATE OR REPLACE FUNCTION bayesian_ab_test(
    control_successes INTEGER,
    control_trials INTEGER,
    treatment_successes INTEGER,
    treatment_trials INTEGER,
    prior_alpha FLOAT DEFAULT 1,
    prior_beta FLOAT DEFAULT 1
) RETURNS JSONB AS $
DECLARE
    control_alpha FLOAT;
    control_beta FLOAT;
    treatment_alpha FLOAT;
    treatment_beta FLOAT;
    probability_treatment_better FLOAT;
BEGIN
    -- Update posteriors
    control_alpha := prior_alpha + control_successes;
    control_beta := prior_beta + control_trials - control_successes;
    treatment_alpha := prior_alpha + treatment_successes;
    treatment_beta := prior_beta + treatment_trials - treatment_successes;
    
    -- Calculate probability treatment is better
    -- This is a simplified calculation
    probability_treatment_better := 
        treatment_alpha / (treatment_alpha + treatment_beta) >
        control_alpha / (control_alpha + control_beta);
    
    RETURN jsonb_build_object(
        'control_posterior_mean', control_alpha / (control_alpha + control_beta),
        'treatment_posterior_mean', treatment_alpha / (treatment_alpha + treatment_beta),
        'probability_treatment_better', probability_treatment_better,
        'expected_lift', 
            (treatment_alpha / (treatment_alpha + treatment_beta) - 
             control_alpha / (control_alpha + control_beta)) /
            (control_alpha / (control_alpha + control_beta))
    );
END;
$ LANGUAGE plpgsql;
```

---

*This document is maintained by the Analytics Engineering team. For questions about A/B testing methodology or implementation, please contact analytics-eng@ytempire.com or visit the #ab-testing Slack channel.*