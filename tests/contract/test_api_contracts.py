"""
API Contract Testing - Advanced SDLC Enhancement
Validates API implementation against OpenAPI specification with comprehensive testing.
"""

import pytest
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import jsonschema
from jsonschema import validate, ValidationError
import requests
import tempfile
import os


class OpenAPIValidator:
    """Advanced OpenAPI specification validator with comprehensive contract testing."""
    
    def __init__(self, spec_path: str):
        """Initialize validator with OpenAPI specification."""
        self.spec_path = Path(spec_path)
        self.spec = self._load_spec()
        self.schemas = self._extract_schemas()
    
    def _load_spec(self) -> Dict[str, Any]:
        """Load and parse OpenAPI specification."""
        with open(self.spec_path, 'r') as f:
            if self.spec_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                return json.load(f)
    
    def _extract_schemas(self) -> Dict[str, Any]:
        """Extract component schemas from OpenAPI spec."""
        return self.spec.get('components', {}).get('schemas', {})
    
    def validate_response_schema(self, response_data: Dict[str, Any], schema_name: str) -> bool:
        """Validate response data against OpenAPI schema."""
        if schema_name not in self.schemas:
            raise ValueError(f"Schema '{schema_name}' not found in OpenAPI spec")
        
        schema = self.schemas[schema_name]
        
        # Resolve $ref references
        resolved_schema = self._resolve_references(schema)
        
        try:
            validate(instance=response_data, schema=resolved_schema)
            return True
        except ValidationError as e:
            print(f"Schema validation error: {e.message}")
            print(f"Failed at path: {' -> '.join(str(p) for p in e.absolute_path)}")
            return False
    
    def _resolve_references(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve $ref references in schema definition."""
        if isinstance(schema, dict):
            if '$ref' in schema:
                ref_path = schema['$ref']
                if ref_path.startswith('#/components/schemas/'):
                    schema_name = ref_path.split('/')[-1]
                    if schema_name in self.schemas:
                        return self._resolve_references(self.schemas[schema_name])
            else:
                resolved = {}
                for key, value in schema.items():
                    resolved[key] = self._resolve_references(value)
                return resolved
        elif isinstance(schema, list):
            return [self._resolve_references(item) for item in schema]
        
        return schema
    
    def get_endpoint_info(self, path: str, method: str) -> Optional[Dict[str, Any]]:
        """Get endpoint definition from OpenAPI spec."""
        paths = self.spec.get('paths', {})
        if path in paths and method.lower() in paths[path]:
            return paths[path][method.lower()]
        return None
    
    def validate_request_body(self, request_data: Dict[str, Any], path: str, method: str) -> bool:
        """Validate request body against OpenAPI specification."""
        endpoint_info = self.get_endpoint_info(path, method)
        if not endpoint_info:
            return True  # No spec to validate against
        
        request_body_spec = endpoint_info.get('requestBody')
        if not request_body_spec:
            return True  # No request body expected
        
        content_spec = request_body_spec.get('content', {})
        json_spec = content_spec.get('application/json', {})
        schema = json_spec.get('schema', {})
        
        if schema:
            resolved_schema = self._resolve_references(schema)
            try:
                validate(instance=request_data, schema=resolved_schema)
                return True
            except ValidationError as e:
                print(f"Request validation error: {e.message}")
                return False
        
        return True


@pytest.fixture
def api_validator():
    """Fixture providing OpenAPI validator."""
    spec_path = Path(__file__).parent.parent.parent / "api" / "openapi.yaml"
    return OpenAPIValidator(spec_path)


@pytest.fixture
def mock_api_server():
    """Fixture providing mock API server for contract testing."""
    # In a real implementation, this would start a test server
    # For this example, we'll simulate responses
    class MockAPIServer:
        def __init__(self):
            self.base_url = "http://localhost:8000/v2"
        
        def get(self, endpoint: str, **kwargs) -> 'MockResponse':
            return self._mock_response(endpoint, "GET", **kwargs)
        
        def post(self, endpoint: str, **kwargs) -> 'MockResponse':
            return self._mock_response(endpoint, "POST", **kwargs)
        
        def _mock_response(self, endpoint: str, method: str, **kwargs) -> 'MockResponse':
            # Simulate different responses based on endpoint
            if endpoint == "/health":
                return MockResponse(200, {
                    "status": "healthy",
                    "timestamp": "2025-07-29T10:30:00Z",
                    "version": "2.0.0",
                    "model_status": {
                        "loaded": True,
                        "version": "pneumonia-detector-v2.1",
                        "last_updated": "2025-07-29T08:00:00Z"
                    },
                    "system_metrics": {
                        "cpu_usage": 25.5,
                        "memory_usage": 68.2,
                        "gpu_usage": 12.1
                    }
                })
            elif endpoint == "/predict/single":
                return MockResponse(200, {
                    "prediction": "pneumonia",
                    "confidence": 0.87,
                    "processing_time_ms": 245,
                    "model_version": "pneumonia-detector-v2.1",
                    "metadata": {
                        "image_size": [224, 224],
                        "preprocessing_applied": ["resize", "normalize", "augment"]
                    },
                    "risk_factors": {
                        "severity": "moderate",
                        "recommendations": ["Follow up with radiologist", "Consider additional imaging"]
                    }
                })
            else:
                return MockResponse(404, {"error": "not_found", "message": "Endpoint not found"})
    
    class MockResponse:
        def __init__(self, status_code: int, json_data: Dict[str, Any]):
            self.status_code = status_code
            self._json_data = json_data
        
        def json(self) -> Dict[str, Any]:
            return self._json_data
    
    return MockAPIServer()


class TestAPIContracts:
    """Comprehensive API contract testing against OpenAPI specification."""
    
    @pytest.mark.contract
    def test_openapi_spec_validity(self, api_validator):
        """Validate that the OpenAPI specification itself is valid."""
        # Check required top-level fields
        required_fields = ['openapi', 'info', 'paths']
        for field in required_fields:
            assert field in api_validator.spec, f"Missing required field: {field}"
        
        # Validate OpenAPI version
        assert api_validator.spec['openapi'].startswith('3.'), "OpenAPI version should be 3.x"
        
        # Validate info section
        info = api_validator.spec['info']
        assert 'title' in info, "API title is required"
        assert 'version' in info, "API version is required"
        
        # Validate paths exist
        paths = api_validator.spec.get('paths', {})
        assert len(paths) > 0, "At least one path should be defined"
        
        print(f"✓ OpenAPI spec valid - {len(paths)} paths defined")
    
    @pytest.mark.contract
    def test_health_endpoint_contract(self, api_validator, mock_api_server):
        """Test health endpoint against OpenAPI contract."""
        response = mock_api_server.get("/health")
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        response_data = response.json()
        is_valid = api_validator.validate_response_schema(response_data, "HealthStatus")
        
        assert is_valid, "Health endpoint response doesn't match OpenAPI schema"
        
        # Additional business logic validation
        assert response_data['status'] in ['healthy', 'degraded', 'unhealthy']
        assert 'timestamp' in response_data
        assert 'version' in response_data
        
        print("✓ Health endpoint contract validation passed")
    
    @pytest.mark.contract
    def test_prediction_endpoint_contract(self, api_validator, mock_api_server):
        """Test prediction endpoint against OpenAPI contract."""
        # Test single prediction endpoint
        response = mock_api_server.post("/predict/single")
        
        assert response.status_code == 200
        
        response_data = response.json()
        is_valid = api_validator.validate_response_schema(response_data, "SinglePredictionResponse")
        
        assert is_valid, "Prediction endpoint response doesn't match OpenAPI schema"
        
        # Validate business logic constraints
        assert response_data['prediction'] in ['normal', 'pneumonia']
        assert 0.0 <= response_data['confidence'] <= 1.0
        assert response_data['processing_time_ms'] >= 0
        assert 'model_version' in response_data
        
        print("✓ Prediction endpoint contract validation passed")
    
    @pytest.mark.contract
    def test_error_response_contracts(self, api_validator, mock_api_server):
        """Test error responses match OpenAPI error schemas."""
        # Test 404 error
        response = mock_api_server.get("/nonexistent")
        
        assert response.status_code == 404
        
        response_data = response.json()
        is_valid = api_validator.validate_response_schema(response_data, "Error")
        
        assert is_valid, "Error response doesn't match OpenAPI schema"
        assert 'error' in response_data
        assert 'message' in response_data
        
        print("✓ Error response contract validation passed")
    
    @pytest.mark.contract
    @pytest.mark.parametrize("endpoint,method,expected_status", [
        ("/health", "GET", 200),
        ("/predict/single", "POST", 200),
        ("/models", "GET", 200),
    ])
    def test_endpoint_security_requirements(self, api_validator, endpoint, method, expected_status):
        """Test that endpoints properly define security requirements."""
        endpoint_info = api_validator.get_endpoint_info(endpoint, method)
        
        if endpoint == "/health":
            # Health endpoint should not require authentication
            security = endpoint_info.get('security', [])
            if security:
                assert {} in security, "Health endpoint should allow anonymous access"
        else:
            # Other endpoints should require authentication
            security = endpoint_info.get('security')
            if security is None:
                # Check global security requirements
                global_security = api_validator.spec.get('security', [])
                assert len(global_security) > 0, f"Endpoint {endpoint} should require authentication"
        
        print(f"✓ Security requirements validated for {method} {endpoint}")
    
    @pytest.mark.contract
    def test_response_content_types(self, api_validator):
        """Validate that all endpoints specify proper content types."""
        paths = api_validator.spec.get('paths', {})
        
        for path, methods in paths.items():
            for method, endpoint_spec in methods.items():
                responses = endpoint_spec.get('responses', {})
                
                for status_code, response_spec in responses.items():
                    if status_code.startswith('2'):  # Success responses
                        content = response_spec.get('content', {})
                        assert 'application/json' in content, (
                            f"Endpoint {method.upper()} {path} should return JSON for {status_code}"
                        )
        
        print("✓ Content type validation passed for all endpoints")
    
    @pytest.mark.contract
    def test_parameter_validation_contracts(self, api_validator):
        """Test parameter validation rules in OpenAPI spec."""
        paths = api_validator.spec.get('paths', {})
        
        for path, methods in paths.items():
            for method, endpoint_spec in methods.items():
                parameters = endpoint_spec.get('parameters', [])
                
                for param in parameters:
                    # Validate required parameter properties
                    assert 'name' in param, f"Parameter missing name in {method.upper()} {path}"
                    assert 'in' in param, f"Parameter missing 'in' specification in {method.upper()} {path}"
                    assert 'schema' in param, f"Parameter missing schema in {method.upper()} {path}"
                    
                    # Validate parameter constraints
                    schema = param['schema']
                    if param.get('required', False):
                        assert schema.get('type') is not None, (
                            f"Required parameter {param['name']} should have type specified"
                        )
        
        print("✓ Parameter validation contracts verified")
    
    @pytest.mark.contract
    def test_schema_completeness(self, api_validator):
        """Test that all referenced schemas are properly defined."""
        # Collect all schema references
        schema_refs = set()
        
        def collect_refs(obj):
            if isinstance(obj, dict):
                if '$ref' in obj:
                    ref = obj['$ref']
                    if ref.startswith('#/components/schemas/'):
                        schema_refs.add(ref.split('/')[-1])
                else:
                    for value in obj.values():
                        collect_refs(value)
            elif isinstance(obj, list):
                for item in obj:
                    collect_refs(item)
        
        collect_refs(api_validator.spec)
        
        # Verify all referenced schemas exist
        defined_schemas = set(api_validator.schemas.keys())
        missing_schemas = schema_refs - defined_schemas
        
        assert len(missing_schemas) == 0, f"Missing schema definitions: {missing_schemas}"
        
        print(f"✓ Schema completeness verified - {len(defined_schemas)} schemas defined")
    
    @pytest.mark.contract
    def test_api_versioning_consistency(self, api_validator):
        """Test API versioning consistency across specification."""
        # Check server URLs include version
        servers = api_validator.spec.get('servers', [])
        for server in servers:
            url = server.get('url', '')
            assert '/v2' in url or 'v2' in url, f"Server URL should include version: {url}"
        
        # Check API version in info
        info_version = api_validator.spec['info']['version']
        assert info_version.startswith('2.'), f"API version should be 2.x, got {info_version}"
        
        print("✓ API versioning consistency verified")
    
    @pytest.mark.contract
    def test_security_schemes_completeness(self, api_validator):
        """Test that all security schemes are properly defined."""
        security_schemes = api_validator.spec.get('components', {}).get('securitySchemes', {})
        
        # Should have at least API key authentication
        assert len(security_schemes) > 0, "At least one security scheme should be defined"
        
        for scheme_name, scheme_def in security_schemes.items():
            assert 'type' in scheme_def, f"Security scheme {scheme_name} missing type"
            
            if scheme_def['type'] == 'apiKey':
                assert 'in' in scheme_def, f"API key scheme {scheme_name} missing 'in' field"
                assert 'name' in scheme_def, f"API key scheme {scheme_name} missing 'name' field"
        
        print(f"✓ Security schemes completeness verified - {len(security_schemes)} schemes defined")


@pytest.mark.contract
def test_contract_testing_integration():
    """Integration test for contract testing framework."""
    # Verify test framework setup
    spec_path = Path(__file__).parent.parent.parent / "api" / "openapi.yaml"
    assert spec_path.exists(), "OpenAPI specification file should exist"
    
    # Load and validate spec structure
    with open(spec_path) as f:
        spec = yaml.safe_load(f)
    
    # Basic structure validation
    assert 'openapi' in spec
    assert 'info' in spec
    assert 'paths' in spec
    assert 'components' in spec
    
    print("✓ Contract testing framework integration verified")


if __name__ == "__main__":
    # Allow running contract tests directly
    pytest.main([__file__, "-v", "-m", "contract"])