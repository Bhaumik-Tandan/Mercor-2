from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import time
import uuid
import asyncio
import json
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASSIFICATION_SERVER_URL = "http://localhost:8001/classify"

app = FastAPI(
    title="Simple Intelligent Classification Proxy",
    description="Simple proxy with intelligent batching to minimize total processing time"
)

class ProxyRequest(BaseModel):
    """Request model for single text classification"""
    sequence: str

class ProxyResponse(BaseModel):
    """Response model containing classification result ('code' or 'not code')"""
    result: str

class SimpleBatchingProxy:
    def __init__(self):
        self.http_client = None
        self.pending_requests = {}  # request_id -> (sequence, future)
        self.batch_size = 5
        self.batch_timeout = 0.005  # 5ms to collect batch
        self.server_semaphore = asyncio.Semaphore(1)
        
    async def initialize(self):
        """Initialize HTTP client"""
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=1.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            http2=True
        )
        logger.info("SimpleBatchingProxy initialized")
    
    async def classify(self, sequence: str) -> str:
        """Classify a single sequence with intelligent batching"""
        request_id = str(uuid.uuid4())
        
        # Create a future for this request
        future = asyncio.Future()
        self.pending_requests[request_id] = (sequence, future)
        
        # Try to process immediately if we have enough requests
        if len(self.pending_requests) >= self.batch_size:
            await self._process_batch()
        else:
            # Schedule batch processing after timeout
            asyncio.create_task(self._process_batch_with_timeout())
        
        # Wait for result
        try:
            result = await asyncio.wait_for(future, timeout=5.0)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Request {request_id} timed out")
            return "timeout"
        finally:
            # Clean up
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
    
    async def _process_batch_with_timeout(self):
        """Process batch after timeout if we have requests"""
        await asyncio.sleep(self.batch_timeout)
        if self.pending_requests:
            await self._process_batch()
    
    async def _process_batch(self):
        """Process current batch of requests"""
        if not self.pending_requests:
            return
        
        # Take up to batch_size requests
        batch_items = list(self.pending_requests.items())[:self.batch_size]
        sequences = [item[1][0] for item in batch_items]
        request_ids = [item[0] for item in batch_items]
        
        # Remove from pending
        for request_id in request_ids:
            del self.pending_requests[request_id]
        
        # Process batch
        async with self.server_semaphore:
            try:
                response = await self.http_client.post(
                    CLASSIFICATION_SERVER_URL,
                    json={"sequences": sequences},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data["results"]
                    
                    # Set results for all requests in batch
                    for request_id, result in zip(request_ids, results):
                        # Find the original future and set result
                        for req_id, (seq, future) in list(self.pending_requests.items()):
                            if seq == sequences[request_ids.index(request_id)]:
                                if not future.done():
                                    future.set_result(result)
                                break
                        else:
                            # If not found in pending, it might be in the batch_items
                            for req_id, (seq, future) in batch_items:
                                if seq == sequences[request_ids.index(request_id)]:
                                    if not future.done():
                                        future.set_result(result)
                                    break
                    
                    logger.info(f"✅ Processed batch of {len(sequences)} requests")
                    
                elif response.status_code == 429:
                    # Rate limited - process requests individually
                    logger.warning("Rate limited, processing requests individually")
                    for request_id, (sequence, future) in batch_items:
                        if not future.done():
                            individual_result = await self._process_single(sequence)
                            future.set_result(individual_result)
                else:
                    # Error - set error for all requests
                    for request_id, (sequence, future) in batch_items:
                        if not future.done():
                            future.set_result("error")
                            
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Set error for all requests
                for request_id, (sequence, future) in batch_items:
                    if not future.done():
                        future.set_result("error")
    
    async def _process_single(self, sequence: str) -> str:
        """Process a single request"""
        try:
            response = await self.http_client.post(
                CLASSIFICATION_SERVER_URL,
                json={"sequences": [sequence]},
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["results"][0]
            else:
                return "error"
        except Exception as e:
            logger.error(f"Single request error: {e}")
            return "error"
    
    async def shutdown(self):
        """Shutdown the proxy"""
        if self.http_client:
            await self.http_client.aclose()

# Global proxy instance
proxy = SimpleBatchingProxy()

@app.on_event("startup")
async def startup_event():
    await proxy.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    await proxy.shutdown()

@app.post("/proxy_classify")
async def proxy_classify(req: ProxyRequest):
    """Proxy endpoint for classification requests"""
    result = await proxy.classify(req.sequence)
    return ProxyResponse(result=result)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pending_requests": len(proxy.pending_requests)
    }
