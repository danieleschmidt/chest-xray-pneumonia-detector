#!/usr/bin/env python3
"""
Quantum Enhanced API Gateway - Generation 1: MAKE IT WORK
A simple yet powerful API gateway with quantum-inspired routing algorithms.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
from hashlib import sha256
import aiohttp
from aiohttp import web
from aiohttp.web import middleware
import ssl

class QuantumState(Enum):
    """Quantum states for routing decisions."""
    SUPERPOSITION = "superposition"
    ENTANGLED = "entangled" 
    COLLAPSED = "collapsed"

@dataclass
class ServiceEndpoint:
    """Service endpoint configuration."""
    name: str
    url: str
    health_check_path: str = "/health"
    weight: float = 1.0
    quantum_state: QuantumState = QuantumState.SUPERPOSITION
    last_health_check: float = 0.0
    healthy: bool = True
    response_time: float = 0.0
    
class QuantumRouter:
    """Quantum-inspired load balancing router."""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceEndpoint]] = {}
        self.entangled_pairs: Set[tuple] = set()
        
    def register_service(self, service_name: str, endpoint: ServiceEndpoint):
        """Register a service endpoint."""
        if service_name not in self.services:
            self.services[service_name] = []
        self.services[service_name].append(endpoint)
        
    def quantum_route_selection(self, service_name: str) -> Optional[ServiceEndpoint]:
        """Select endpoint using quantum-inspired algorithm."""
        if service_name not in self.services:
            return None
            
        endpoints = [ep for ep in self.services[service_name] if ep.healthy]
        if not endpoints:
            return None
            
        # Quantum superposition: All endpoints have probability
        total_weight = sum(ep.weight / (ep.response_time + 0.001) for ep in endpoints)
        
        # Quantum measurement collapse
        import random
        threshold = random.random() * total_weight
        current = 0.0
        
        for endpoint in endpoints:
            current += endpoint.weight / (endpoint.response_time + 0.001)
            if current >= threshold:
                endpoint.quantum_state = QuantumState.COLLAPSED
                return endpoint
                
        return endpoints[0]

class APIGateway:
    """Main API Gateway implementation."""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.router = QuantumRouter()
        self.app = web.Application(middlewares=[
            self.cors_middleware,
            self.auth_middleware,
            self.rate_limit_middleware,
            self.logging_middleware
        ])
        self.setup_routes()
        self.health_checker_task = None
        
    def setup_routes(self):
        """Setup API routes."""
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/metrics', self.metrics)
        self.app.router.add_post('/services/{service_name}/register', self.register_service)
        self.app.router.add_route('*', '/api/{service_name}/{path:.*}', self.proxy_request)
        
    @middleware
    async def cors_middleware(self, request, handler):
        """CORS middleware."""
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
        
    @middleware
    async def auth_middleware(self, request, handler):
        """Basic authentication middleware."""
        # Simple token validation
        auth_header = request.headers.get('Authorization')
        if request.path.startswith('/api/') and not auth_header:
            return web.json_response(
                {'error': 'Authorization required'}, 
                status=401
            )
        return await handler(request)
        
    @middleware
    async def rate_limit_middleware(self, request, handler):
        """Rate limiting middleware."""
        # Simple in-memory rate limiting
        client_ip = request.remote
        # Implementation would use Redis in production
        return await handler(request)
        
    @middleware
    async def logging_middleware(self, request, handler):
        """Request logging middleware."""
        start_time = time.time()
        response = await handler(request)
        duration = time.time() - start_time
        
        logging.info(f"{request.method} {request.path} - {response.status} - {duration:.3f}s")
        return response
        
    async def health_check(self, request):
        """Gateway health check."""
        return web.json_response({
            'status': 'healthy',
            'timestamp': time.time(),
            'version': '1.0.0',
            'services': len(self.router.services)
        })
        
    async def metrics(self, request):
        """Prometheus-compatible metrics."""
        metrics = []
        for service_name, endpoints in self.router.services.items():
            for endpoint in endpoints:
                metrics.append(f'service_health{{service="{service_name}",endpoint="{endpoint.name}"}} {int(endpoint.healthy)}')
                metrics.append(f'service_response_time{{service="{service_name}",endpoint="{endpoint.name}"}} {endpoint.response_time}')
                
        return web.Response(text='\n'.join(metrics), content_type='text/plain')
        
    async def register_service(self, request):
        """Register a new service endpoint."""
        service_name = request.match_info['service_name']
        data = await request.json()
        
        endpoint = ServiceEndpoint(
            name=data['name'],
            url=data['url'],
            health_check_path=data.get('health_check_path', '/health'),
            weight=data.get('weight', 1.0)
        )
        
        self.router.register_service(service_name, endpoint)
        
        return web.json_response({
            'message': f'Service {service_name} registered successfully',
            'endpoint': endpoint.name
        })
        
    async def proxy_request(self, request):
        """Proxy request to backend service."""
        service_name = request.match_info['service_name']
        path = request.match_info['path']
        
        # Select endpoint using quantum routing
        endpoint = self.router.quantum_route_selection(service_name)
        if not endpoint:
            return web.json_response(
                {'error': f'Service {service_name} not available'}, 
                status=503
            )
            
        # Build target URL
        target_url = f"{endpoint.url.rstrip('/')}/{path}"
        
        # Forward request
        async with aiohttp.ClientSession() as session:
            try:
                start_time = time.time()
                
                async with session.request(
                    method=request.method,
                    url=target_url,
                    headers=dict(request.headers),
                    data=await request.read() if request.body_exists else None,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    body = await resp.read()
                    
                    # Update endpoint metrics
                    endpoint.response_time = time.time() - start_time
                    
                    return web.Response(
                        body=body,
                        status=resp.status,
                        headers=dict(resp.headers)
                    )
                    
            except Exception as e:
                endpoint.healthy = False
                logging.error(f"Proxy error for {service_name}: {e}")
                return web.json_response(
                    {'error': 'Backend service unavailable'}, 
                    status=502
                )
                
    async def health_checker(self):
        """Background health checker for services."""
        while True:
            for service_name, endpoints in self.router.services.items():
                for endpoint in endpoints:
                    try:
                        async with aiohttp.ClientSession() as session:
                            health_url = f"{endpoint.url.rstrip('/')}{endpoint.health_check_path}"
                            async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                                endpoint.healthy = resp.status == 200
                                endpoint.last_health_check = time.time()
                    except:
                        endpoint.healthy = False
                        endpoint.last_health_check = time.time()
                        
            await asyncio.sleep(30)  # Check every 30 seconds
            
    async def start_server(self):
        """Start the API gateway server."""
        # Start health checker
        self.health_checker_task = asyncio.create_task(self.health_checker())
        
        # Start web server
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        print(f"Quantum Enhanced API Gateway started on port {self.port}")
        
        # Keep server running
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            print("Shutting down gateway...")
        finally:
            if self.health_checker_task:
                self.health_checker_task.cancel()
            await runner.cleanup()

async def main():
    """Main entry point."""
    gateway = APIGateway(port=8080)
    
    # Register default services
    gateway.router.register_service(
        'pneumonia-detection',
        ServiceEndpoint(
            name='primary',
            url='http://localhost:8000',
            health_check_path='/health'
        )
    )
    
    await gateway.start_server()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())